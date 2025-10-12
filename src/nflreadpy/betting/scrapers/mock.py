"""Mock sportsbook scraper providing deterministic market coverage."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import random
from typing import List, Sequence

from .base import OddsQuote, SportsbookScraper, normalise_american_odds


@dataclasses.dataclass
class _Fixture:
    event_id: str
    home_team: str
    away_team: str
    home_moneyline: int
    spread: float
    total: float


class MockSportsbookScraper(SportsbookScraper):
    """Yield a wide variety of markets for integration testing."""

    name = "mockbook"

    _DEFAULT_FIXTURES: Sequence[_Fixture] = (
        _Fixture(
            event_id="2024-NE-NYJ",
            home_team="NE",
            away_team="NYJ",
            home_moneyline=-145,
            spread=-6.5,
            total=41.5,
        ),
        _Fixture(
            event_id="2024-DEN-KC",
            home_team="DEN",
            away_team="KC",
            home_moneyline=+120,
            spread=+3.0,
            total=46.5,
        ),
    )

    _SEED_FIXTURE_TEMPLATES: Sequence[tuple[str, str, str]] = (
        ("2024-NE-NYJ", "NE", "NYJ"),
        ("2024-DEN-KC", "DEN", "KC"),
        ("2024-BUF-MIA", "BUF", "MIA"),
        ("2024-PHI-DAL", "PHI", "DAL"),
        ("2024-DET-GB", "DET", "GB"),
    )

    def __init__(
        self,
        fixtures: Sequence[_Fixture] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.poll_interval_seconds = 5.0
        self._rng = random.Random(seed) if seed is not None else None
        if fixtures is not None:
            self._fixtures = list(fixtures)
        elif self._rng is not None:
            self._fixtures = self._generate_seeded_fixtures()
        else:
            self._fixtures = list(self._DEFAULT_FIXTURES)

    def _generate_seeded_fixtures(self) -> list[_Fixture]:
        assert self._rng is not None
        fixtures: list[_Fixture] = []
        for event_id, home, away in self._SEED_FIXTURE_TEMPLATES:
            home_moneyline = self._rng.choice(
                [-185, -170, -150, -135, -120, -110, -105, 100, 115, 130, 145, 165]
            )
            spread = self._rng.choice(
                [-9.5, -7.5, -6.5, -3.5, -1.5, 0.0, 1.5, 3.5, 6.5, 9.5]
            )
            total = 36.5 + self._rng.randrange(0, 16)
            fixtures.append(
                _Fixture(
                    event_id=event_id,
                    home_team=home,
                    away_team=away,
                    home_moneyline=home_moneyline,
                    spread=spread,
                    total=total,
                )
            )
        return fixtures

    async def _fetch_lines_impl(self) -> List[OddsQuote]:
        now = dt.datetime.now(dt.timezone.utc)
        quotes: List[OddsQuote] = []
        for fixture in self._fixtures:
            quotes.extend(self._moneyline_quotes(fixture, now))
            quotes.extend(self._spread_quotes(fixture, now))
            quotes.extend(self._total_quotes(fixture, now))
            quotes.extend(self._team_total_quotes(fixture, now))
            quotes.extend(self._player_prop_quotes(fixture, now))
            quotes.extend(self._scope_splits(fixture, now))
            quotes.extend(self._three_way_winners(fixture, now))
            quotes.extend(self._leader_markets(fixture, now))
            quotes.extend(self._combo_props(fixture, now))
        await asyncio.sleep(0)
        return quotes

    def _moneyline_quotes(self, fixture: _Fixture, timestamp: dt.datetime) -> List[OddsQuote]:
        home_price = fixture.home_moneyline
        away_price = -home_price if home_price < 0 else -(200 - home_price)
        return [
            OddsQuote(
                event_id=fixture.event_id,
                sportsbook=self.name,
                book_market_group="Game Lines",
                market="moneyline",
                scope="game",
                entity_type="team",
                team_or_player=fixture.home_team,
                side=None,
                line=None,
                american_odds=home_price,
                observed_at=timestamp,
                extra={"opponent": fixture.away_team},
            ),
            OddsQuote(
                event_id=fixture.event_id,
                sportsbook=self.name,
                book_market_group="Game Lines",
                market="moneyline",
                scope="game",
                entity_type="team",
                team_or_player=fixture.away_team,
                side=None,
                line=None,
                american_odds=away_price,
                observed_at=timestamp,
                extra={"opponent": fixture.home_team},
            ),
        ]

    def _spread_quotes(self, fixture: _Fixture, timestamp: dt.datetime) -> List[OddsQuote]:
        main = [
            OddsQuote(
                event_id=fixture.event_id,
                sportsbook=self.name,
                book_market_group="Game Lines",
                market="spread",
                scope="game",
                entity_type="team",
                team_or_player=fixture.home_team,
                side="fav" if fixture.spread < 0 else "dog",
                line=fixture.spread,
                american_odds=-110,
                observed_at=timestamp,
                extra={},
            ),
            OddsQuote(
                event_id=fixture.event_id,
                sportsbook=self.name,
                book_market_group="Game Lines",
                market="spread",
                scope="game",
                entity_type="team",
                team_or_player=fixture.away_team,
                side="dog" if fixture.spread < 0 else "fav",
                line=-fixture.spread,
                american_odds=-110,
                observed_at=timestamp,
                extra={},
            ),
        ]
        alt_lines = []
        for ladder in (-9.5, -3.5, 3.5, 9.5):
            price = normalise_american_odds(130 if ladder < 0 else 145)
            alt_lines.append(
                OddsQuote(
                    event_id=fixture.event_id,
                    sportsbook=self.name,
                    book_market_group="Alternate Spread",
                    market="spread_alt",
                    scope="game",
                    entity_type="team",
                    team_or_player=fixture.home_team,
                    side="fav" if ladder < 0 else "dog",
                    line=ladder,
                    american_odds=price,
                    observed_at=timestamp,
                    extra={"matrix_row": True},
                )
            )
            alt_lines.append(
                OddsQuote(
                    event_id=fixture.event_id,
                    sportsbook=self.name,
                    book_market_group="Alternate Spread",
                    market="spread_alt",
                    scope="game",
                    entity_type="team",
                    team_or_player=fixture.away_team,
                    side="dog" if ladder < 0 else "fav",
                    line=-ladder,
                    american_odds=-price,
                    observed_at=timestamp,
                    extra={"matrix_row": True},
                )
            )
        return main + alt_lines

    def _total_quotes(self, fixture: _Fixture, timestamp: dt.datetime) -> List[OddsQuote]:
        totals = []
        for side, price in (("over", -108), ("under", -112)):
            totals.append(
                OddsQuote(
                    event_id=fixture.event_id,
                    sportsbook=self.name,
                    book_market_group="Game Lines",
                    market="total",
                    scope="game",
                    entity_type="total",
                    team_or_player="Total",
                    side=side,
                    line=fixture.total,
                    american_odds=price,
                    observed_at=timestamp,
                    extra={},
                )
            )
        for ladder, price in ((38.5, +115), (45.5, -102), (51.5, +130)):
            totals.append(
                OddsQuote(
                    event_id=fixture.event_id,
                    sportsbook=self.name,
                    book_market_group="Alternate Totals",
                    market="total_alt",
                    scope="game",
                    entity_type="total",
                    team_or_player="Total",
                    side="over",
                    line=ladder,
                    american_odds=normalise_american_odds(price),
                    observed_at=timestamp,
                    extra={"matrix_row": True},
                )
            )
            totals.append(
                OddsQuote(
                    event_id=fixture.event_id,
                    sportsbook=self.name,
                    book_market_group="Alternate Totals",
                    market="total_alt",
                    scope="game",
                    entity_type="total",
                    team_or_player="Total",
                    side="under",
                    line=ladder,
                    american_odds=normalise_american_odds(-price),
                    observed_at=timestamp,
                    extra={"matrix_row": True},
                )
            )
        return totals

    def _team_total_quotes(self, fixture: _Fixture, timestamp: dt.datetime) -> List[OddsQuote]:
        home_mean = fixture.total / 2 + fixture.spread / 2
        away_mean = fixture.total - home_mean
        quotes = []
        for team, mean in (
            (fixture.home_team, home_mean),
            (fixture.away_team, away_mean),
        ):
            for side in ("over", "under"):
                quotes.append(
                    OddsQuote(
                        event_id=fixture.event_id,
                        sportsbook=self.name,
                        book_market_group="Team Totals",
                        market="team_total",
                        scope="game",
                        entity_type="team",
                        team_or_player=team,
                        side=side,
                        line=round(mean, 1),
                        american_odds=-105 if side == "over" else -115,
                        observed_at=timestamp,
                        extra={},
                    )
                )
        return quotes

    def _player_prop_quotes(self, fixture: _Fixture, timestamp: dt.datetime) -> List[OddsQuote]:
        players = {
            fixture.home_team: [
                ("Mac Jones", "passing_yards", 248.5, 0.58),
                ("Rhamondre Stevenson", "rushing_yards", 62.5, 0.55),
                ("Hunter Henry", "receptions", 4.5, 0.52),
            ],
            fixture.away_team: [
                ("Garrett Wilson", "receiving_yards", 71.5, 0.61),
                ("Breece Hall", "rushing_yards", 59.5, 0.57),
                ("C.J. Uzomah", "record_sack", 0.5, 0.35),
            ],
        }
        quotes: List[OddsQuote] = []
        for team, props in players.items():
            for player, market, line, model_prob in props:
                if market == "record_sack":
                    quotes.append(
                        OddsQuote(
                            event_id=fixture.event_id,
                            sportsbook=self.name,
                            book_market_group="Player Specials",
                            market="record_sack",
                            scope="game",
                            entity_type="player",
                            team_or_player=player,
                            side="yes",
                            line=0.5,
                            american_odds=+175,
                            observed_at=timestamp,
                            extra={
                                "projection_mean": model_prob,
                                "projection_distribution": "bernoulli",
                            },
                        )
                    )
                    quotes.append(
                        OddsQuote(
                            event_id=fixture.event_id,
                            sportsbook=self.name,
                            book_market_group="Player Specials",
                            market="record_sack",
                            scope="game",
                            entity_type="player",
                            team_or_player=player,
                            side="no",
                            line=0.5,
                            american_odds=-210,
                            observed_at=timestamp,
                            extra={
                                "projection_mean": model_prob,
                                "projection_distribution": "bernoulli",
                            },
                        )
                    )
                    continue
                for side, price in (("over", -115), ("under", -105)):
                    quotes.append(
                        OddsQuote(
                            event_id=fixture.event_id,
                            sportsbook=self.name,
                            book_market_group="Player Props",
                            market=market,
                            scope="game",
                            entity_type="player",
                            team_or_player=player,
                            side=side,
                            line=line,
                            american_odds=price,
                            observed_at=timestamp,
                            extra={
                                "projection_mean": line + (4 if side == "over" else -4),
                                "projection_stdev": max(8.0, line * 0.18),
                            },
                        )
                    )
                if market.endswith("yards"):
                    for ladder, price in ((line - 20, +145), (line + 20, +175)):
                        quotes.append(
                            OddsQuote(
                                event_id=fixture.event_id,
                                sportsbook=self.name,
                                book_market_group="Alt Ladders",
                                market=f"{market}_alt",
                                scope="game",
                                entity_type="player",
                                team_or_player=player,
                                side="over",
                                line=ladder,
                                american_odds=normalise_american_odds(price),
                                observed_at=timestamp,
                                extra={
                                    "projection_mean": line + 4,
                                    "projection_stdev": max(8.0, line * 0.18),
                                    "matrix_row": True,
                                },
                            )
                        )
                if market == "receptions":
                    for ladder, price in ((line - 1, +165), (line + 2, +200)):
                        quotes.append(
                            OddsQuote(
                                event_id=fixture.event_id,
                                sportsbook=self.name,
                                book_market_group="Alt Receptions",
                                market="receptions_alt",
                                scope="game",
                                entity_type="player",
                                team_or_player=player,
                                side="over",
                                line=ladder,
                                american_odds=normalise_american_odds(price),
                                observed_at=timestamp,
                                extra={
                                    "projection_mean": line + 0.6,
                                    "projection_stdev": max(1.2, line * 0.22),
                                    "matrix_row": True,
                                },
                            )
                        )
        quotes.append(
            OddsQuote(
                event_id=fixture.event_id,
                sportsbook=self.name,
                book_market_group="Either Player",
                market="longest_reception",
                scope="game",
                entity_type="either",
                team_or_player="Sutton/Dobbins",
                side="over",
                line=29.5,
                american_odds=+135,
                observed_at=timestamp,
                extra={"participants": ["Courtland Sutton", "J.K. Dobbins"], "projection_mean": 0.44},
            )
        )
        return quotes

    def _scope_splits(self, fixture: _Fixture, timestamp: dt.datetime) -> List[OddsQuote]:
        spreads: List[OddsQuote] = []
        for scope, factor in (("1h", 0.52), ("1q", 0.27)):
            spreads.append(
                OddsQuote(
                    event_id=fixture.event_id,
                    sportsbook=self.name,
                    book_market_group="Game Lines",
                    market="spread",
                    scope=scope,
                    entity_type="team",
                    team_or_player=fixture.home_team,
                    side="fav" if fixture.spread < 0 else "dog",
                    line=round(fixture.spread * factor, 1),
                    american_odds=-110,
                    observed_at=timestamp,
                    extra={},
                )
            )
            spreads.append(
                OddsQuote(
                    event_id=fixture.event_id,
                    sportsbook=self.name,
                    book_market_group="Game Lines",
                    market="total",
                    scope=scope,
                    entity_type="total",
                    team_or_player="Total",
                    side="over",
                    line=round(fixture.total * factor, 1),
                    american_odds=-105,
                    observed_at=timestamp,
                    extra={},
                )
            )
            spreads.append(
                OddsQuote(
                    event_id=fixture.event_id,
                    sportsbook=self.name,
                    book_market_group="Team Totals",
                    market="team_total",
                    scope=scope,
                    entity_type="team",
                    team_or_player=fixture.home_team,
                    side="over",
                    line=round((fixture.total / 2 + fixture.spread / 2) * factor, 1),
                    american_odds=-110,
                    observed_at=timestamp,
                    extra={},
                )
            )
            spreads.append(
                OddsQuote(
                    event_id=fixture.event_id,
                    sportsbook=self.name,
                    book_market_group="Team Totals",
                    market="team_total",
                    scope=scope,
                    entity_type="team",
                    team_or_player=fixture.away_team,
                    side="over",
                    line=round((fixture.total / 2 - fixture.spread / 2) * factor, 1),
                    american_odds=-110,
                    observed_at=timestamp,
                    extra={},
                )
            )
        return spreads

    def _three_way_winners(
        self, fixture: _Fixture, timestamp: dt.datetime
    ) -> List[OddsQuote]:
        outcomes = []
        for scope, factor in (("game", 1.0), ("1h", 0.52), ("1q", 0.27)):
            outcomes.extend(
                [
                    OddsQuote(
                        event_id=fixture.event_id,
                        sportsbook=self.name,
                        book_market_group="Game Lines",
                        market="winner_3_way",
                        scope=scope,
                        entity_type="team",
                        team_or_player=fixture.home_team,
                        side=None,
                        line=None,
                        american_odds=-110,
                        observed_at=timestamp,
                        extra={"factor": factor},
                    ),
                    OddsQuote(
                        event_id=fixture.event_id,
                        sportsbook=self.name,
                        book_market_group="Game Lines",
                        market="winner_3_way",
                        scope=scope,
                        entity_type="team",
                        team_or_player=fixture.away_team,
                        side=None,
                        line=None,
                        american_odds=+145,
                        observed_at=timestamp,
                        extra={"factor": factor},
                    ),
                    OddsQuote(
                        event_id=fixture.event_id,
                        sportsbook=self.name,
                        book_market_group="Game Lines",
                        market="winner_3_way",
                        scope=scope,
                        entity_type="team",
                        team_or_player="Tie",
                        side=None,
                        line=None,
                        american_odds=+350,
                        observed_at=timestamp,
                        extra={"factor": factor},
                    ),
                ]
            )
        return outcomes

    def _leader_markets(
        self, fixture: _Fixture, timestamp: dt.datetime
    ) -> List[OddsQuote]:
        leaders: List[OddsQuote] = []
        contenders = [
            "Mac Jones",
            "Patrick Mahomes",
            "Russell Wilson",
            "Garrett Wilson",
        ]
        for market, odds in (("passing_leader", +210), ("receiving_leader", +260)):
            for player in contenders:
                leaders.append(
                    OddsQuote(
                        event_id=fixture.event_id,
                        sportsbook=self.name,
                        book_market_group="Leader Markets",
                        market=market,
                        scope="game",
                        entity_type="leader",
                        team_or_player=player,
                        side=None,
                        line=None,
                        american_odds=normalise_american_odds(odds),
                        observed_at=timestamp,
                        extra={"participants": contenders, "projection_mean": 0.5},
                    )
                )
        return leaders

    def _combo_props(
        self, fixture: _Fixture, timestamp: dt.datetime
    ) -> List[OddsQuote]:
        combos: List[OddsQuote] = []
        combos.append(
            OddsQuote(
                event_id=fixture.event_id,
                sportsbook=self.name,
                book_market_group="Combo Props",
                market="passing_rushing_yards",
                scope="game",
                entity_type="player",
                team_or_player="Patrick Mahomes",
                side="over",
                line=324.5,
                american_odds=-105,
                observed_at=timestamp,
                extra={"projection_mean": 338.0, "projection_stdev": 32.0},
            )
        )
        combos.append(
            OddsQuote(
                event_id=fixture.event_id,
                sportsbook=self.name,
                book_market_group="Combo Props",
                market="two_plus_touchdowns",
                scope="game",
                entity_type="player",
                team_or_player="Courtland Sutton",
                side="yes",
                line=1.5,
                american_odds=+475,
                observed_at=timestamp,
                extra={"projection_mean": 0.28, "projection_distribution": "poisson"},
            )
        )
        return combos
