"""Utilities for calibrating scoring scope scale factors.

This module centralises the logic for learning how game-level point totals
should be allocated across betting scopes such as halves and quarters.  The
calibration routine is intentionally lightweight – it relies on Polars to
aggregate historical scoring splits and produces a callable model that the
simulation engine can query when sampling outcomes for a particular scope.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Mapping

import polars as pl

ScopeKey = tuple[int | None, str]


def _canonical_scope(scope: str) -> str:
    """Normalise scope tokens to an internal canonical representation."""

    token = scope.strip().lower().replace(" ", "_")
    aliases = {
        "1h": "first_half",
        "first_half": "first_half",
        "2h": "second_half",
        "second_half": "second_half",
        "1q": "first_quarter",
        "first_quarter": "first_quarter",
        "2q": "second_quarter",
        "second_quarter": "second_quarter",
        "3q": "third_quarter",
        "third_quarter": "third_quarter",
        "4q": "fourth_quarter",
        "fourth_quarter": "fourth_quarter",
        "ot": "overtime",
        "game": "game",
        "total": "game",
    }
    return aliases.get(token, token)


@dataclasses.dataclass(slots=True)
class ScopeScalingEntry:
    """Represents a calibrated scaling factor for a specific scope."""

    scope: str
    factor: float
    samples: int
    season: int | None = None
    intercept: float = 0.0
    updated_at: dt.datetime | None = None


class ScopeScalingModel:
    """Callable container that stores learned scope scaling factors."""

    DEFAULT_FACTORS: Mapping[str, float] = {
        "game": 1.0,
        "first_half": 0.52,
        "second_half": 0.48,
        "first_quarter": 0.27,
        "second_quarter": 0.23,
        "third_quarter": 0.26,
        "fourth_quarter": 0.24,
        "overtime": 0.06,
    }

    def __init__(
        self,
        base_factors: Mapping[str, float] | None = None,
        *,
        seasonal_factors: Mapping[int, Mapping[str, float]] | None = None,
        sample_counts: Mapping[ScopeKey, int] | None = None,
        metadata: Mapping[str, Any] | None = None,
        default_factors: Mapping[str, float] | None = None,
    ) -> None:
        self._default_factors: Dict[str, float] = dict(default_factors or self.DEFAULT_FACTORS)
        self._base_factors: Dict[str, float] = {
            key: float(value)
            for key, value in (base_factors or {}).items()
        }
        self._seasonal_factors: Dict[int, Dict[str, float]] = {
            int(season): {scope: float(val) for scope, val in values.items()}
            for season, values in (seasonal_factors or {}).items()
        }
        self._sample_counts: Dict[ScopeKey, int] = {
            (season, scope): int(count)
            for (season, scope), count in (sample_counts or {}).items()
        }
        self._metadata: Dict[str, Any] = dict(metadata or {})

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "ScopeScalingModel":
        """Return a model seeded with the heuristics used historically."""

        return cls(base_factors=cls.DEFAULT_FACTORS)

    @staticmethod
    def canonical_scope(scope: str) -> str:
        """Expose the canonical scope normalisation used internally."""

        return _canonical_scope(scope)

    @classmethod
    def calibrate(
        cls,
        data: pl.DataFrame,
        *,
        scope_column: str = "scope",
        scope_points_column: str = "scope_points",
        total_points_column: str = "game_points",
        season_column: str | None = "season",
        weight_column: str | None = None,
        min_samples: int = 4,
        default_factors: Mapping[str, float] | None = None,
    ) -> "ScopeScalingModel":
        """Learn scale factors from historical scoring splits.

        Args:
            data: Polars frame containing historical observations.  Each row
                should include the total points scored in the game,
                the points for a specific scope (half/quarter), and the scope
                label itself.
            scope_column: Column containing the scope tokens.
            scope_points_column: Column with observed scope points.
            total_points_column: Column containing full game points.
            season_column: Optional column describing the season/year.  When
                provided the model tracks seasonal factors in addition to the
                overall aggregates.
            weight_column: Optional column containing weights for each
                observation.
            min_samples: Minimum number of observations required to record a
                factor for a group.
            default_factors: Optional baseline mapping that should be used as
                a fallback when insufficient data exists for a scope.

        Returns:
            A :class:`ScopeScalingModel` instance with calibrated factors.
        """

        if data.is_empty():
            return cls(default_factors=default_factors)

        required = {scope_column, scope_points_column, total_points_column}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Missing required columns for calibration: {', '.join(missing)}")

        frame = data.select(list(required | ({season_column} if season_column else set()) | ({weight_column} if weight_column else set())))

        frame = frame.with_columns(
            pl.col(scope_column)
            .cast(pl.Utf8)
            .map_elements(_canonical_scope, return_dtype=pl.Utf8)
            .alias("__scope"),
            pl.col(scope_points_column).cast(pl.Float64).alias("__scope_points"),
            pl.col(total_points_column).cast(pl.Float64).alias("__total_points"),
        )

        if season_column and season_column in frame.columns:
            frame = frame.with_columns(pl.col(season_column).cast(pl.Int64).alias("__season"))
        else:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Int64).alias("__season"))

        if weight_column and weight_column in frame.columns:
            frame = frame.with_columns(
                pl.col(weight_column)
                .cast(pl.Float64)
                .fill_null(1.0)
                .alias("__weight"),
            )
        else:
            frame = frame.with_columns(pl.lit(1.0).alias("__weight"))

        frame = frame.filter(pl.col("__total_points") > 0)
        frame = frame.with_columns(
            (pl.col("__scope_points") * pl.col("__weight")).alias("__weighted_scope"),
            (pl.col("__total_points") * pl.col("__weight")).alias("__weighted_total"),
        )

        aggregations = [
            pl.len().alias("samples"),
            pl.col("__weighted_scope").sum().alias("scope_sum"),
            pl.col("__weighted_total").sum().alias("total_sum"),
        ]

        by_scope = (
            frame.group_by("__scope")
            .agg(aggregations)
            .with_columns(
                pl.when(pl.col("total_sum") > 0)
                .then(pl.col("scope_sum") / pl.col("total_sum"))
                .otherwise(None)
                .alias("factor"),
            )
            .filter(pl.col("samples") >= min_samples)
        )

        base_factors: Dict[str, float] = {}
        sample_counts: Dict[ScopeKey, int] = {}
        for row in by_scope.iter_rows(named=True):
            factor = row.get("factor")
            scope = str(row["__scope"])
            if factor is not None:
                base_factors[scope] = float(factor)
                sample_counts[(None, scope)] = int(row["samples"])

        seasonal_factors: Dict[int, Dict[str, float]] = {}
        season_present = bool(
            frame.select(pl.col("__season").is_not_null().any()).item()
        )
        if season_present:
            by_season = (
                frame.group_by("__season", "__scope")
                .agg(aggregations)
                .with_columns(
                    pl.when(pl.col("total_sum") > 0)
                    .then(pl.col("scope_sum") / pl.col("total_sum"))
                    .otherwise(None)
                    .alias("factor"),
                )
                .filter(pl.col("samples") >= min_samples)
            )

            for row in by_season.iter_rows(named=True):
                season_value = row["__season"]
                if season_value is None:
                    continue
                factor = row.get("factor")
                scope = str(row["__scope"])
                if factor is None:
                    continue
                season_key = int(season_value)
                seasonal_factors.setdefault(season_key, {})[scope] = float(factor)
                sample_counts[(season_key, scope)] = int(row["samples"])

        metadata = {
            "calibrated_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "min_samples": min_samples,
        }

        return cls(
            base_factors=base_factors,
            seasonal_factors=seasonal_factors,
            sample_counts=sample_counts,
            metadata=metadata,
            default_factors=default_factors,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    def factor(self, scope: str, *, season: int | None = None) -> float:
        """Return the scale factor for ``scope`` and optional ``season``."""

        key = _canonical_scope(scope)
        if season is not None:
            season_mapping = self._seasonal_factors.get(int(season))
            if season_mapping and key in season_mapping:
                return season_mapping[key]
        if key in self._base_factors:
            return self._base_factors[key]
        return self._default_factors.get(key, 1.0)

    def __call__(self, scope: str, *, season: int | None = None) -> float:
        return self.factor(scope, season=season)

    def with_overrides(
        self,
        overrides: Mapping[str, float],
        *,
        season: int | None = None,
    ) -> "ScopeScalingModel":
        """Return a new model with additional overrides applied."""

        if not overrides:
            return self

        base = dict(self._base_factors)
        seasonal = {season_key: dict(values) for season_key, values in self._seasonal_factors.items()}
        samples = dict(self._sample_counts)

        def _apply(target: Dict[str, float]) -> None:
            for key, value in overrides.items():
                canonical = _canonical_scope(key)
                target[canonical] = float(value)
                samples[(season if season is not None else None, canonical)] = max(
                    samples.get((season if season is not None else None, canonical), 0),
                    0,
                )

        if season is None:
            _apply(base)
        else:
            mapping = seasonal.setdefault(int(season), {})
            _apply(mapping)

        return ScopeScalingModel(
            base_factors=base,
            seasonal_factors=seasonal,
            sample_counts=samples,
            metadata=self._metadata,
            default_factors=self._default_factors,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def to_frame(self) -> pl.DataFrame:
        """Materialise the model parameters as a Polars dataframe."""

        records: list[ScopeScalingEntry] = []

        for scope, factor in self._base_factors.items():
            records.append(
                ScopeScalingEntry(
                    scope=scope,
                    factor=factor,
                    samples=self._sample_counts.get((None, scope), 0),
                    season=None,
                )
            )

        for season, mapping in self._seasonal_factors.items():
            for scope, factor in mapping.items():
                records.append(
                    ScopeScalingEntry(
                        scope=scope,
                        factor=factor,
                        samples=self._sample_counts.get((season, scope), 0),
                        season=season,
                    )
                )

        if not records:
            return pl.DataFrame({
                "scope": list(self._default_factors.keys()),
                "factor": list(self._default_factors.values()),
                "samples": [0] * len(self._default_factors),
                "season": [None] * len(self._default_factors),
            })

        return pl.DataFrame(records)

    def save(self, path: str | Path) -> Path:
        """Persist the calibrated parameters to ``path`` as Parquet."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame()
        frame.write_parquet(destination)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        default_factors: Mapping[str, float] | None = None,
    ) -> "ScopeScalingModel":
        """Load a previously saved set of scope scaling parameters."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)

        frame = pl.read_parquet(source)
        required = {"scope", "factor"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"Persisted scope scaling parameters at {source} are missing columns: {', '.join(missing)}"
            )

        base: Dict[str, float] = {}
        seasonal: Dict[int, Dict[str, float]] = {}
        samples: Dict[ScopeKey, int] = {}
        for row in frame.iter_rows(named=True):
            scope = _canonical_scope(str(row["scope"]))
            factor = float(row["factor"])
            season_value = row.get("season")
            sample_count = int(row.get("samples") or 0)
            if season_value is None:
                base[scope] = factor
                samples[(None, scope)] = sample_count
            else:
                season_key = int(season_value)
                seasonal.setdefault(season_key, {})[scope] = factor
                samples[(season_key, scope)] = sample_count

        metadata = {
            "loaded_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "source": str(source),
        }

        return cls(
            base_factors=base,
            seasonal_factors=seasonal,
            sample_counts=samples,
            metadata=metadata,
            default_factors=default_factors,
        )


__all__ = ["ScopeScalingModel", "ScopeScalingEntry"]

