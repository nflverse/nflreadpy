"""Reusable betting math helpers for odds and probabilities."""

from __future__ import annotations

from fractions import Fraction

OddsValue = int | float | str

__all__ = [
    "OddsValue",
    "normalise_american_odds",
    "american_to_decimal",
    "american_to_profit_multiplier",
    "decimal_to_american",
    "fractional_to_decimal",
    "decimal_to_fractional",
    "american_to_fractional",
    "fractional_to_american",
    "implied_probability_from_decimal",
    "implied_probability_from_fractional",
    "implied_probability_from_american",
    "implied_probability_to_decimal",
    "implied_probability_to_american",
    "implied_probability_to_fraction",
]


def normalise_american_odds(value: OddsValue) -> int:
    """Coerce American odds into a signed integer."""

    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    stripped = value.strip()
    if not stripped:
        raise ValueError("Empty odds value")
    if stripped[0] in {"+", "-"}:
        return int(stripped)
    return int(f"+{stripped}")


def american_to_decimal(value: OddsValue) -> float:
    """Convert an American price into European decimal odds."""

    price = normalise_american_odds(value)
    if price == 0:
        raise ValueError("American odds cannot be zero")
    if price > 0:
        return 1.0 + price / 100.0
    return 1.0 + 100.0 / -price


def american_to_profit_multiplier(value: OddsValue) -> float:
    """Return the net profit multiplier for a one-unit stake at American odds."""

    price = normalise_american_odds(value)
    if price == 0:
        raise ValueError("American odds cannot be zero")
    if price > 0:
        return price / 100.0
    return 100.0 / -price


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to their American representation."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must exceed 1.0")
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100.0))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def fractional_to_decimal(numerator: int, denominator: int) -> float:
    """Convert fractional odds to decimal form."""

    if denominator == 0:
        raise ValueError("Fractional denominator cannot be zero")
    if numerator < 0 or denominator < 0:
        raise ValueError("Fractional odds must be non-negative")
    fraction = Fraction(numerator, denominator)
    return 1.0 + float(fraction)


def decimal_to_fractional(decimal_odds: float, *, max_denominator: int = 512) -> tuple[int, int]:
    """Convert decimal odds to a simplified fractional representation."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must exceed 1.0")
    fraction = Fraction(decimal_odds - 1.0).limit_denominator(max_denominator)
    return fraction.numerator, fraction.denominator


def american_to_fractional(
    value: OddsValue, *, max_denominator: int = 512
) -> tuple[int, int]:
    """Convert American odds to fractional form."""

    decimal = american_to_decimal(value)
    return decimal_to_fractional(decimal, max_denominator=max_denominator)


def fractional_to_american(numerator: int, denominator: int) -> int:
    """Convert fractional odds to American format."""

    decimal = fractional_to_decimal(numerator, denominator)
    return decimal_to_american(decimal)


def implied_probability_from_decimal(decimal_odds: float) -> float:
    """Return the bookmaker's implied win probability from decimal odds."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must exceed 1.0")
    return 1.0 / decimal_odds


def implied_probability_from_fractional(numerator: int, denominator: int) -> float:
    """Return the implied probability from fractional odds."""

    decimal = fractional_to_decimal(numerator, denominator)
    return implied_probability_from_decimal(decimal)


def implied_probability_from_american(value: OddsValue) -> float:
    """Return the implied probability from American odds."""

    decimal = american_to_decimal(value)
    return implied_probability_from_decimal(decimal)


def implied_probability_to_decimal(probability: float) -> float:
    """Convert an implied probability into decimal odds."""

    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("Probability must be between 0 and 1 (exclusive)")
    return 1.0 / probability


def implied_probability_to_american(probability: float) -> int:
    """Convert an implied probability into American odds."""

    decimal = implied_probability_to_decimal(probability)
    return decimal_to_american(decimal)


def implied_probability_to_fraction(
    probability: float, *, max_denominator: int = 512
) -> tuple[int, int]:
    """Convert an implied probability into fractional odds."""

    decimal = implied_probability_to_decimal(probability)
    return decimal_to_fractional(decimal, max_denominator=max_denominator)
