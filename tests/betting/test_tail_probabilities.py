from __future__ import annotations

import datetime as dt
import math
import random

import pytest

from nflreadpy.betting.analytics import EdgeDetector
from nflreadpy.betting.ingestion import IngestedOdds
from nflreadpy.betting.models import SimulationResult
from nflreadpy.betting.scrapers.base import OddsQuote
from nflreadpy.betting.tail import LadderPoint, fit_monotone_envelope


def test_fit_monotone_envelope_enforces_monotonicity() -> None:
    rng = random.Random(42)
    lines = [-8.0, -5.5, -3.0, -0.5, 2.0, 4.5]
    baseline = [0.18, 0.32, 0.45, 0.58, 0.69, 0.81]
    points = []
    for line, target in zip(lines, baseline):
        noisy = max(0.01, min(0.99, target + rng.uniform(-0.06, 0.06)))
        weight = rng.randint(15, 40)
        points.append(
            LadderPoint(
                line=line,
                win_probability=noisy,
                push_probability=0.0,
                weight=float(weight),
            )
        )
    fitted = fit_monotone_envelope(points, direction="increasing")
    assert all(fitted[i] <= fitted[i + 1] + 1e-9 for i in range(len(fitted) - 1))
    raw_error = sum((pt.win_probability - true) ** 2 for pt, true in zip(points, baseline))
    fitted_error = sum((prob - true) ** 2 for prob, true in zip(fitted, baseline))
    assert fitted_error <= raw_error + 1e-9


@pytest.mark.parametrize("side, direction", [("over", "decreasing"), ("under", "increasing")])
def test_tail_metadata_is_monotone_and_calibrated(side: str, direction: str) -> None:
    event_id = "BUF@KC"
    quotes = [
        IngestedOdds(
            event_id=event_id,
            sportsbook="Mock",
            book_market_group="Alternate Totals",
            market="total_alt",
            scope="game",
            entity_type="total",
            team_or_player="Game Total",
            side=side,
            line=line,
            american_odds=price,
            observed_at=dt.datetime(2024, 1, 21, tzinfo=dt.timezone.utc),
            extra={"matrix_row": True},
        )
        for line, price in ((18.0, +125), (40.0, -110), (70.0, +240))
    ]
    odds_quotes = [
        OddsQuote(
            event_id=q.event_id,
            sportsbook=q.sportsbook,
            book_market_group=q.book_market_group,
            market=q.market,
            scope="game",
            entity_type="total",
            team_or_player=q.team_or_player,
            side=q.side,
            line=q.line,
            american_odds=q.american_odds,
            observed_at=q.observed_at,
            extra=q.extra,
        )
        for q in quotes
    ]
    total_counts = {20: 2, 60: 1}
    margin_counts = {0: 3}
    simulation = SimulationResult(
        event_id=event_id,
        home_team="BUF",
        away_team="KC",
        iterations=sum(total_counts.values()),
        home_win_probability=0.5,
        away_win_probability=0.5,
        expected_margin=0.0,
        expected_total=sum(total * count for total, count in total_counts.items())
        / sum(total_counts.values()),
        margin_distribution=margin_counts,
        total_distribution=total_counts,
        home_score_distribution={10: 2, 30: 1},
        away_score_distribution={10: 2, 30: 1},
    )
    detector = EdgeDetector(value_threshold=-1.0)
    detector.detect(odds_quotes, [simulation])
    iterations = sum(total_counts.values())
    for record in quotes:
        payload = record.extra.get("tail_probability")
        assert isinstance(payload, dict)
        assert payload["direction"] == direction
        line = float(record.line or 0.0)
        if side == "over":
            wins = sum(count for total, count in total_counts.items() if total > line)
        else:
            wins = sum(count for total, count in total_counts.items() if total < line)
        expected_raw = wins / iterations
        assert math.isclose(payload["raw_win"], expected_raw, rel_tol=1e-9)
        assert payload["method"] == "beta_isotonic"
        assert payload["prior"] == {"alpha": 1.0, "beta": 1.0}
    sorted_quotes = sorted(quotes, key=lambda q: float(q.line or 0.0))
    final_probs = [record.extra["tail_probability"]["final_win"] for record in sorted_quotes]
    if direction == "decreasing":
        assert all(a >= b - 1e-9 for a, b in zip(final_probs, final_probs[1:]))
    else:
        assert all(a <= b + 1e-9 for a, b in zip(final_probs, final_probs[1:]))
    extreme_low = sorted_quotes[-1].extra["tail_probability"]
    extreme_high = sorted_quotes[0].extra["tail_probability"]
    assert 0.0 < extreme_low["adjusted_win"] < 0.5
    assert 0.5 < extreme_high["adjusted_win"] < 1.0
