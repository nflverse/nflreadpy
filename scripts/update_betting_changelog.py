#!/usr/bin/env python3
"""Utility for keeping the betting changelog in sync with release bumps."""
from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
BETTING_CHANGELOG = REPO_ROOT / "betting" / "CHANGELOG.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
BETTING_ROOT = Path("src/nflreadpy/betting")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def read_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text())
    return data["project"]["version"]


def list_version_tags() -> list[str]:
    result = _git("tag", "--list", "v*", "--sort=-v:refname")
    if result.returncode != 0:
        raise RuntimeError(f"git tag lookup failed: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def determine_base_tag(version: str) -> str | None:
    tags = list_version_tags()
    if not tags:
        return None
    current_tag = f"v{version}"
    try:
        idx = tags.index(current_tag)
    except ValueError:
        return tags[0]
    next_idx = idx + 1
    if next_idx < len(tags):
        return tags[next_idx]
    return None


def collect_changed_modules(base_tag: str | None) -> list[str]:
    if base_tag:
        diff_cmd = [
            "diff",
            "--name-only",
            f"{base_tag}",
            "HEAD",
            "--",
            str(BETTING_ROOT),
        ]
        result = _git(*diff_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"git diff failed: {result.stderr.strip()}")
        paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    else:
        root = REPO_ROOT / BETTING_ROOT
        paths = [str(path.relative_to(REPO_ROOT)) for path in root.rglob("*.py")]
    modules: set[str] = set()
    for raw in paths:
        rel = Path(raw)
        abs_path = REPO_ROOT / rel
        if not abs_path.exists():
            # diff against tag may list deleted files; ignore them.
            continue
        if abs_path.is_dir():
            continue
        if abs_path.suffix not in {".py", ""}:
            continue
        parts = list(rel.parts[2:])  # drop "src", "nflreadpy"
        if not parts:
            continue
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        if not parts:
            continue
        modules.add("/".join(parts).removesuffix(".py"))
    return sorted(modules)


def ensure_unreleased_placeholder(text: str) -> tuple[str, int]:
    marker = "## Unreleased"
    try:
        start = text.index(marker)
    except ValueError as exc:  # pragma: no cover - guard for manual edits
        raise RuntimeError("`## Unreleased` section missing from betting changelog") from exc
    section_start = start + len(marker)
    next_heading = text.find("\n## ", section_start)
    if next_heading == -1:
        next_heading = len(text)
    placeholder = "\n\n- _No unreleased changes._\n\n"
    updated = text[:section_start] + placeholder + text[next_heading:]
    return updated, next_heading


def format_entry(version: str, modules: list[str], date: _dt.date | None = None) -> str:
    release_date = date or _dt.date.today()
    header = f"## v{version} ({release_date.isoformat()})"
    lines = [header, "", "### Changed"]
    for module in modules:
        dotted = module.replace("/", ".")
        lines.append(f"- Updated `{dotted}` components.")
    lines.append("")
    return "\n".join(lines)


def update_changelog(version: str, modules: list[str], apply: bool) -> bool:
    content = BETTING_CHANGELOG.read_text()
    header = f"## v{version}"
    if header in content:
        return True
    if not modules:
        return True
    updated, insertion_point = ensure_unreleased_placeholder(content)
    new_entry = format_entry(version, modules)
    new_content = updated[:insertion_point] + "\n" + new_entry + updated[insertion_point:]
    if apply:
        BETTING_CHANGELOG.write_text(new_content)
    return apply


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="apply updates in place instead of running in check mode",
    )
    parser.add_argument(
        "--version",
        help="override detected project version",
    )
    parser.add_argument(
        "--base-tag",
        help="override the git tag used for change detection",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    version = args.version or read_version()
    base_tag = args.base_tag if args.base_tag is not None else determine_base_tag(version)
    modules = collect_changed_modules(base_tag)
    apply = bool(args.write)
    changed = update_changelog(version, modules, apply)
    if changed:
        return 0
    if modules:
        missing = ", ".join(modules)
        print(
            "Betting changelog missing entry for version",
            version,
            "covering modules:",
            missing,
            file=sys.stderr,
        )
        print("Re-run with --write to append the entry automatically.", file=sys.stderr)
        return 1
    # Nothing changed in betting modules; no entry needed.
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
