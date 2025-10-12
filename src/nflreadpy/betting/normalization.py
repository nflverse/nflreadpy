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
import re
from functools import lru_cache
from typing import Dict, Iterable, Mapping, MutableMapping

TEAM_ALIASES: Mapping[str, str] = {
    "ari": "ARI",
    "arizona": "ARI",
    "cardinals": "ARI",
    "atl": "ATL",
    "atlanta": "ATL",
    "falcons": "ATL",
    "bal": "BAL",
    "baltimore": "BAL",
    "ravens": "BAL",
    "buf": "BUF",
    "bills": "BUF",
    "car": "CAR",
    "carolina": "CAR",
    "panthers": "CAR",
    "chi": "CHI",
    "bears": "CHI",
    "cin": "CIN",
    "bengals": "CIN",
    "cle": "CLE",
    "browns": "CLE",
    "dal": "DAL",
    "cowboys": "DAL",
    "den": "DEN",
    "broncos": "DEN",
    "det": "DET",
    "lions": "DET",
    "gb": "GB",
    "gnb": "GB",
    "packers": "GB",
    "hou": "HOU",
    "texans": "HOU",
    "ind": "IND",
    "colts": "IND",
    "jax": "JAX",
    "jac": "JAX",
    "jaguars": "JAX",
    "kc": "KC",
    "kan": "KC",
    "chiefs": "KC",
    "lv": "LV",
    "rai": "LV",
    "raiders": "LV",
    "lac": "LAC",
    "chargers": "LAC",
    "lar": "LAR",
    "ram": "LAR",
    "rams": "LAR",
    "mia": "MIA",
    "dolphins": "MIA",
    "min": "MIN",
    "vikings": "MIN",
    "ne": "NE",
    "nwe": "NE",
    "patriots": "NE",
    "no": "NO",
    "nor": "NO",
    "saints": "NO",
    "nyg": "NYG",
    "giants": "NYG",
    "nyj": "NYJ",
    "jets": "NYJ",
    "phi": "PHI",
    "eagles": "PHI",
    "pit": "PIT",
    "steelers": "PIT",
    "sf": "SF",
    "sfo": "SF",
    "49ers": "SF",
    "sea": "SEA",
    "seahawks": "SEA",
    "tb": "TB",
    "tam": "TB",
    "buccaneers": "TB",
    "ten": "TEN",
    "titans": "TEN",
    "was": "WAS",
    "wft": "WAS",
    "commanders": "WAS",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


@dataclasses.dataclass
class NameNormalizer:
    """Normalise sportsbook, team, and player identifiers."""

    team_aliases: MutableMapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(TEAM_ALIASES)
    )
    player_aliases: MutableMapping[str, str] = dataclasses.field(default_factory=dict)
    sportsbook_aliases: MutableMapping[str, str] = dataclasses.field(
        default_factory=dict
    )

    def register_players(self, aliases: Mapping[str, str]) -> None:
        for raw, canonical in aliases.items():
            self.player_aliases[_slug(raw)] = canonical

    def canonical_team(self, value: str) -> str:
        slug = _slug(value)
        if slug in self.team_aliases:
            return self.team_aliases[slug]
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
        canonical = " ".join(part.capitalize() for part in value.split())
        self.player_aliases[slug] = canonical
        return canonical

    def canonical_sportsbook(self, value: str) -> str:
        slug = _slug(value)
        if slug in self.sportsbook_aliases:
            return self.sportsbook_aliases[slug]
        return value.lower().replace(" ", "_")

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


@lru_cache()
def default_normalizer() -> NameNormalizer:
    return NameNormalizer()

