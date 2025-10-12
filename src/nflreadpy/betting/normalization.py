"""Entity and sportsbook name normalisation utilities.

The Bloomberg-style tool needs to join odds across multiple operators and
datasets.  Real feeds often disagree on spelling, casing, or abbreviations
for teams and players.  This module provides a lightweight normalisation
layer that maps raw identifiers into canonical forms so that analytics and
portfolio logic can compare apples with apples.

The implementation intentionally avoids any heavyweight dependencies.  The
team map covers every current NFL franchise with a set of common aliases and
fallbacks.  Player names are canonicalised via a slug function and an
extensible registry so that tests – and future real scrapers – can register
additional aliases at runtime.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    from rapidfuzz import fuzz as _rapidfuzz_fuzz  # type: ignore
    from rapidfuzz import process as _rapidfuzz_process  # type: ignore
except Exception:  # pragma: no cover - fall back to stdlib if missing
    _rapidfuzz_fuzz = None
    _rapidfuzz_process = None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


@dataclasses.dataclass(frozen=True)
class CanonicalEntry:
    """Canonical identifier metadata for fuzzy matching."""

    id: str
    name: str
    aliases: Sequence[str] = dataclasses.field(default_factory=tuple)

    @property
    def labels(self) -> Sequence[str]:
        labels: list[str] = [self.id, self.name]
        labels.extend(alias for alias in self.aliases if alias)
        return labels


@dataclasses.dataclass(frozen=True)
class CanonicalIdentifiers:
    """Grouped canonical identifiers for betting entities."""

    teams: Sequence[CanonicalEntry] = dataclasses.field(default_factory=tuple)
    players: Sequence[CanonicalEntry] = dataclasses.field(default_factory=tuple)
    sportsbooks: Sequence[CanonicalEntry] = dataclasses.field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> CanonicalIdentifiers:
        def build_entries(items: Sequence[Mapping[str, object]] | object) -> list[CanonicalEntry]:
            entries: list[CanonicalEntry] = []
            if not isinstance(items, Sequence):
                return entries
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                entry_id = str(item.get("id", "")).strip()
                name = str(item.get("name", "")).strip() or entry_id
                aliases = item.get("aliases", [])
                if not entry_id:
                    continue
                if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes, bytearray)):
                    alias_list = [str(alias).strip() for alias in aliases if str(alias).strip()]
                else:
                    alias_list = []
                entries.append(CanonicalEntry(entry_id, name, tuple(alias_list)))
            return entries

        return cls(
            teams=tuple(build_entries(data.get("teams", []))),
            players=tuple(build_entries(data.get("players", []))),
            sportsbooks=tuple(build_entries(data.get("sportsbooks", []))),
        )


_DEFAULT_IDENTIFIER_PATH = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "identifiers"
    / "betting_entities.json"
)


def load_canonical_identifiers(path: str | Path | None = None) -> CanonicalIdentifiers:
    """Load canonical identifier metadata from disk."""

    target = Path(path) if path else _DEFAULT_IDENTIFIER_PATH
    try:
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return CanonicalIdentifiers()
    if not isinstance(payload, Mapping):
        raise TypeError("Canonical identifiers file must contain a mapping")
    return CanonicalIdentifiers.from_mapping(payload)


def _extract_fuzzy_matches(value: str, choices: Sequence[str], limit: int = 2) -> list[tuple[str, float]]:
    if not choices or not value.strip():
        return []
    if _rapidfuzz_process and _rapidfuzz_fuzz:  # pragma: no branch - optional dependency
        results = _rapidfuzz_process.extract(  # type: ignore[arg-type]
            value,
            choices,
            scorer=_rapidfuzz_fuzz.WRatio,
            limit=limit,
        )
        return [(match, float(score)) for match, score, _ in results]
    scored = [
        (choice, difflib.SequenceMatcher(None, value, choice).ratio() * 100)
        for choice in choices
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


_DEFAULT_FUZZY_THRESHOLDS: Mapping[str, float] = {
    "team": 88.0,
    "player": 90.0,
    "sportsbook": 80.0,
}


@dataclasses.dataclass
class NameNormalizer:
    """Normalise sportsbook, team, and player identifiers."""

    identifiers_path: str | Path | None = None
    identifiers: CanonicalIdentifiers | None = None
    team_aliases: MutableMapping[str, str] | None = None
    player_aliases: MutableMapping[str, str] | None = None
    sportsbook_aliases: MutableMapping[str, str] | None = None
    fuzzy_matching_enabled: bool = False
    fuzzy_thresholds: Mapping[str, float] | None = None
    fuzzy_ambiguity_margin: float = 5.0

    def __post_init__(self) -> None:
        identifiers = self.identifiers or load_canonical_identifiers(self.identifiers_path)
        self.identifiers = identifiers
        self.team_aliases = dict(self.team_aliases or {})
        self.player_aliases = dict(self.player_aliases or {})
        self.sportsbook_aliases = dict(self.sportsbook_aliases or {})

        self._alias_maps: Dict[str, MutableMapping[str, str]] = {
            "team": self.team_aliases,
            "player": self.player_aliases,
            "sportsbook": self.sportsbook_aliases,
        }
        self._fuzzy_choices: Dict[str, Dict[str, str]] = {key: {} for key in self._alias_maps}

        self._ingest_entries("team", identifiers.teams)
        self._ingest_entries("player", identifiers.players)
        self._ingest_entries("sportsbook", identifiers.sportsbooks)

        thresholds = dict(_DEFAULT_FUZZY_THRESHOLDS)
        if self.fuzzy_thresholds:
            thresholds.update({k: float(v) for k, v in self.fuzzy_thresholds.items()})
        self._fuzzy_thresholds = thresholds

    def register_players(self, aliases: Mapping[str, str]) -> None:
        for raw, canonical in aliases.items():
            self._register_alias("player", raw, canonical)

    def canonical_team(self, value: str) -> str:
        slug = _slug(value)
        if slug in self.team_aliases:
            return self.team_aliases[slug]
        resolved = self._resolve_with_fuzzy("team", value)
        if resolved is not None:
            return resolved
        parts = value.split()
        if parts:
            last_slug = _slug(parts[-1])
            if last_slug in self.team_aliases:
                return self.team_aliases[last_slug]
        if len(value) <= 4:
            return value.upper()
        return value.title()

    def canonical_player(self, value: str) -> str:
        slug = _slug(value)
        if slug in self.player_aliases:
            return self.player_aliases[slug]
        resolved = self._resolve_with_fuzzy("player", value)
        if resolved is not None:
            return resolved
        canonical = " ".join(part.capitalize() for part in value.split())
        self._register_alias("player", value, canonical)
        return canonical

    def canonical_sportsbook(self, value: str) -> str:
        slug = _slug(value)
        if slug in self.sportsbook_aliases:
            return self.sportsbook_aliases[slug]
        resolved = self._resolve_with_fuzzy("sportsbook", value)
        if resolved is not None:
            return resolved
        canonical = value.lower().replace(" ", "_")
        self._register_alias("sportsbook", value, canonical)
        return canonical

    def normalise_quote(self, quote: "OddsQuote") -> "OddsQuote":
        from .scrapers.base import OddsQuote  # local import to avoid cycle

        team_or_player = quote.team_or_player
        if quote.entity_type == "team":
            team_or_player = self.canonical_team(team_or_player)
        elif quote.entity_type in {"player", "either", "leader"}:
            team_or_player = self.canonical_player(team_or_player)

        sportsbook = self.canonical_sportsbook(quote.sportsbook)
        extra: Dict[str, object]
        if quote.extra:
            extra = dict(quote.extra)
        else:
            extra = {}
        participants = extra.get("participants")
        if isinstance(participants, Iterable) and not isinstance(participants, str):
            extra["participants"] = [self.canonical_player(p) for p in participants]
        return dataclasses.replace(
            quote,
            sportsbook=sportsbook,
            team_or_player=team_or_player,
            extra=extra,
        )

    def _ingest_entries(self, domain: str, entries: Sequence[CanonicalEntry]) -> None:
        alias_map = self._alias_maps[domain]
        choices = self._fuzzy_choices[domain]
        for entry in entries:
            canonical = entry.id
            for label in entry.labels:
                label = label.strip()
                if not label:
                    continue
                slug = _slug(label)
                if slug and slug not in alias_map:
                    alias_map[slug] = canonical
                choices.setdefault(label, canonical)

    def _register_alias(self, domain: str, alias: str, canonical: str) -> None:
        alias = alias.strip()
        if not alias:
            return
        alias_map = self._alias_maps[domain]
        slug = _slug(alias)
        if slug:
            alias_map[slug] = canonical
        self._fuzzy_choices[domain][alias] = canonical

    def _resolve_with_fuzzy(self, domain: str, value: str) -> str | None:
        if not self.fuzzy_matching_enabled:
            return None
        choices = self._fuzzy_choices.get(domain)
        if not choices:
            return None
        matches = _extract_fuzzy_matches(value, list(choices.keys()), limit=2)
        if not matches:
            return None
        top_label, top_score = matches[0]
        threshold = self._fuzzy_thresholds.get(domain, 0.0)
        if top_score < threshold:
            return None
        if len(matches) > 1:
            second_label, second_score = matches[1]
            if top_score - second_score < self.fuzzy_ambiguity_margin:
                raise ValueError(
                    "Ambiguous "
                    f"{domain} '{value}' matched '{top_label}' ({top_score:.1f}) "
                    f"and '{second_label}' ({second_score:.1f}); provide a clearer name or "
                    "add an explicit alias."
                )
        canonical = choices[top_label]
        self._register_alias(domain, value, canonical)
        return canonical


@lru_cache()
def default_normalizer() -> NameNormalizer:
    return NameNormalizer()

