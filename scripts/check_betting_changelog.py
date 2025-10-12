#!/usr/bin/env python3
"""Pre-commit helper ensuring betting changes are reflected in the changelog."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BETTING_CHANGELOG = REPO_ROOT / "betting" / "CHANGELOG.md"
BETTING_DIR = "src/nflreadpy/betting/"


def staged_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr.strip() or "failed to inspect staged files", file=sys.stderr)
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    files = staged_files()
    betting_touched = any(path.startswith(BETTING_DIR) for path in files)
    if not betting_touched:
        return 0
    if str(BETTING_CHANGELOG.relative_to(REPO_ROOT)) in files:
        return 0
    print(
        "Files under src/nflreadpy/betting/ changed without updating betting/CHANGELOG.md.",
        file=sys.stderr,
    )
    print(
        "Please stage a changelog update or run scripts/update_betting_changelog.py --write.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
