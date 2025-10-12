from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
)

import yaml
from pydantic import BaseModel, Field

ENVIRONMENT_VARIABLE = "NFLREADPY_BETTING_ENV"
EXTRA_CONFIG_VARIABLE = "NFLREADPY_BETTING_CONFIG"
ENV_OVERRIDE_PREFIX = "NFLREADPY_BETTING__"

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .alerts import AlertManager, AlertSink
    from .analytics import EdgeDetector
    from .ingestion import OddsIngestionService
    from .scrapers.base import SportsbookScraper
    from .quantum import PortfolioOptimizer


class ScraperRuntimeConfig(BaseModel):
    """Runtime attributes applied to instantiated scrapers."""

    poll_interval_seconds: float | None = None
    retry_attempts: int | None = None
    retry_backoff: float | None = None
    timeout_seconds: float | None = None


class ScraperConfig(BaseModel):
    """Configuration describing how to build a sportsbook scraper."""

    type: str
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)
    runtime: ScraperRuntimeConfig = Field(default_factory=ScraperRuntimeConfig)


class SchedulerConfig(BaseModel):
    """Settings controlling the ingestion scheduler defaults."""

    interval_seconds: float = 0.0
    jitter_seconds: float = 0.0
    retries: int = 3
    retry_backoff: float = 2.0


class IngestionConfig(BaseModel):
    """Persistence and freshness controls for odds ingestion."""

    storage_path: str = "betting_odds.sqlite3"
    stale_after_seconds: int = 600
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class IterationConfig(BaseModel):
    """Number of Monte Carlo iterations to run for each workflow."""

    simulate: int = 20_000
    scan: int = 10_000
    dashboard: int = 15_000
    backtest: int = 8_000


class ScopeScalingConfig(BaseModel):
    """Configuration for loading or overriding scope scaling parameters."""

    parameters_path: str | None = None
    fallback_factors: Dict[str, float] = Field(default_factory=dict)
    overrides: Dict[str, float] = Field(default_factory=dict)
    seasonal_overrides: Dict[int, Dict[str, float]] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    """Configuration namespace for statistical models."""

    scope_scaling: ScopeScalingConfig = Field(default_factory=ScopeScalingConfig)


class AnalyticsConfig(BaseModel):
    """Controls for downstream analytics heuristics and defaults."""

    backend: str = "auto"
    value_threshold: float = 0.02
    correlation_penalty: float = 0.0
    correlation_limits: Dict[str, float] = Field(default_factory=dict)
    kelly_fraction: float = 0.5
    bankroll: float = 1_000.0
    portfolio_fraction: float = 0.5
    risk_trials: int = 0
    risk_seed: int | None = None
    history_limit: int = 256
    movement_threshold: int = 30
    iterations: IterationConfig = Field(default_factory=IterationConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


class FuzzyMatchingConfig(BaseModel):
    """Feature flag and score thresholds for fuzzy identifier resolution."""

    enabled: bool = False
    score_thresholds: Dict[str, float] = Field(default_factory=dict)
    ambiguity_margin: float = 5.0


class NormalizationConfig(BaseModel):
    """Controls for canonical identifier loading and fuzzy resolution."""

    canonical_identifiers_path: str | None = "config/identifiers/betting_entities.json"
    fuzzy: FuzzyMatchingConfig = Field(default_factory=FuzzyMatchingConfig)


class BettingConfig(BaseModel):
    """Aggregate configuration for the betting stack."""

    environment: str = "default"
    scrapers: list[ScraperConfig] = Field(default_factory=list)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)


class ConfigurationError(ValueError):
    """Raised when betting configuration validation fails."""


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration at {path} must be a mapping")
    return dict(data)


def _merge_layers(base: Dict[str, Any], layer: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in layer.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _merge_layers(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_env_tokens(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)
    if isinstance(value, Mapping):
        return {k: _resolve_env_tokens(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_resolve_env_tokens(item) for item in value]
    return value


def _coerce_env_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return raw


def _set_nested(mapping: MutableMapping[str, Any], path: Iterable[str], value: Any) -> None:
    segments = list(path)
    if not segments:
        return
    head, *tail = segments
    key = head.lower().replace("-", "_")
    if not tail:
        mapping[key] = value
        return
    child = mapping.get(key)
    if not isinstance(child, MutableMapping):
        child = {}
    else:
        child = dict(child)
    mapping[key] = child
    _set_nested(child, tail, value)


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(data)
    for key, raw_value in os.environ.items():
        if not key.startswith(ENV_OVERRIDE_PREFIX):
            continue
        suffix = key[len(ENV_OVERRIDE_PREFIX) :]
        if not suffix:
            continue
        path = [segment for segment in suffix.split("__") if segment]
        if not path:
            continue
        _set_nested(updated, path, _coerce_env_value(raw_value))
    return updated


def load_betting_config(
    *,
    base_path: str | os.PathLike[str] | None = None,
    environment: str | None = None,
    extra_paths: Sequence[str | os.PathLike[str]] | None = None,
) -> BettingConfig:
    """Load layered configuration for the betting stack.

    The loader merges ``config/betting.yaml`` with optional environment-specific
    overrides (``config/betting.<env>.yaml``), additional override files, and
    environment variable overrides that use ``NFLREADPY_BETTING__`` prefixes.
    """

    config_path = Path(base_path or "config/betting.yaml")
    data = _load_yaml(config_path)

    env_name = environment or os.getenv(ENVIRONMENT_VARIABLE) or data.get("environment")
    if isinstance(env_name, str):
        env_path = config_path.with_name(f"{config_path.stem}.{env_name}{config_path.suffix}")
        if env_path.exists():
            data = _merge_layers(data, _load_yaml(env_path))
        data["environment"] = env_name

    merged = dict(data)
    override_sources: list[Path] = []
    if extra_paths:
        override_sources.extend(Path(path) for path in extra_paths)
    env_overrides = os.getenv(EXTRA_CONFIG_VARIABLE)
    if env_overrides:
        override_sources.extend(Path(token) for token in env_overrides.split(os.pathsep) if token)

    for override in override_sources:
        if override.exists():
            merged = _merge_layers(merged, _load_yaml(override))

    merged = _apply_env_overrides(merged)
    merged = _resolve_env_tokens(merged)

    return BettingConfig.model_validate(merged)


def validate_betting_config(config: BettingConfig) -> list[str]:
    """Validate a :class:`BettingConfig` instance.

    Args:
        config: Parsed configuration object to validate.

    Returns:
        A list of warning messages. The function raises
        :class:`ConfigurationError` if any fatal issues are detected.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not config.scrapers:
        errors.append("at least one scraper must be defined")
    else:
        enabled = [scraper for scraper in config.scrapers if scraper.enabled]
        if not enabled:
            errors.append("all scrapers are disabled; enable at least one")
        for index, scraper in enumerate(config.scrapers):
            if not scraper.type.strip():
                errors.append(f"scraper #{index + 1} is missing a type")
            runtime = scraper.runtime
            for field_name in (
                "poll_interval_seconds",
                "retry_attempts",
                "retry_backoff",
                "timeout_seconds",
            ):
                value = getattr(runtime, field_name)
                if value is not None and value < 0:
                    errors.append(
                        f"scraper '{scraper.type}' runtime field '{field_name}' must be non-negative"
                    )

    ingestion = config.ingestion
    if not str(ingestion.storage_path).strip():
        errors.append("ingestion.storage_path cannot be empty")
    if ingestion.stale_after_seconds <= 0:
        errors.append("ingestion.stale_after_seconds must be greater than zero")
    scheduler = ingestion.scheduler
    if scheduler.interval_seconds < 0:
        errors.append("ingestion.scheduler.interval_seconds must be non-negative")
    if scheduler.jitter_seconds < 0:
        errors.append("ingestion.scheduler.jitter_seconds must be non-negative")
    if scheduler.retries < 0:
        errors.append("ingestion.scheduler.retries must be non-negative")
    if scheduler.retry_backoff < 0:
        errors.append("ingestion.scheduler.retry_backoff must be non-negative")
    if scheduler.interval_seconds and scheduler.interval_seconds < 10:
        warnings.append(
            "ingestion scheduler interval is below 10 seconds; be mindful of provider rate limits"
        )

    analytics = config.analytics
    if analytics.bankroll <= 0:
        errors.append("analytics.bankroll must be greater than zero")
    if not 0 < analytics.kelly_fraction <= 1:
        errors.append("analytics.kelly_fraction must be within (0, 1]")
    if not 0 < analytics.portfolio_fraction <= 1:
        errors.append("analytics.portfolio_fraction must be within (0, 1]")
    if analytics.value_threshold <= 0:
        errors.append("analytics.value_threshold must be greater than zero")
    elif analytics.value_threshold < 0.01:
        warnings.append(
            "analytics value_threshold is very low; expect a large number of candidate opportunities"
        )
    if analytics.risk_trials < 0:
        errors.append("analytics.risk_trials must be non-negative")
    if analytics.history_limit <= 0:
        errors.append("analytics.history_limit must be greater than zero")
    if analytics.movement_threshold < 0:
        errors.append("analytics.movement_threshold must be non-negative")

    optimizer_cfg = analytics.optimizer
    if optimizer_cfg.risk_aversion < 0:
        errors.append("analytics.optimizer.risk_aversion must be non-negative")
    if optimizer_cfg.shots <= 0:
        errors.append("analytics.optimizer.shots must be greater than zero")
    if optimizer_cfg.temperature <= 0:
        errors.append("analytics.optimizer.temperature must be greater than zero")
    if optimizer_cfg.annealing_steps <= 0:
        errors.append("analytics.optimizer.annealing_steps must be greater than zero")
    if optimizer_cfg.annealing_initial_temp <= 0:
        errors.append(
            "analytics.optimizer.annealing_initial_temp must be greater than zero"
        )
    if optimizer_cfg.annealing_cooling_rate <= 0:
        errors.append(
            "analytics.optimizer.annealing_cooling_rate must be greater than zero"
        )
    if optimizer_cfg.qaoa_layers <= 0:
        errors.append("analytics.optimizer.qaoa_layers must be greater than zero")
    if optimizer_cfg.qaoa_gamma < 0:
        errors.append("analytics.optimizer.qaoa_gamma must be non-negative")
    if optimizer_cfg.qaoa_beta < 0:
        errors.append("analytics.optimizer.qaoa_beta must be non-negative")

    for name, value in analytics.iterations.model_dump().items():
        if value <= 0:
            errors.append(f"analytics.iterations.{name} must be greater than zero")

    normalization = config.normalization
    if normalization.canonical_identifiers_path is not None and not str(
        normalization.canonical_identifiers_path
    ).strip():
        errors.append("normalization.canonical_identifiers_path cannot be empty")
    fuzzy = normalization.fuzzy
    if fuzzy.ambiguity_margin < 0:
        errors.append("normalization.fuzzy.ambiguity_margin must be non-negative")
    for domain, threshold in fuzzy.score_thresholds.items():
        if threshold < 0 or threshold > 100:
            errors.append(
                "normalization.fuzzy.score_thresholds." +
                f"{domain} must be between 0 and 100"
            )
    if fuzzy.enabled and not fuzzy.score_thresholds:
        warnings.append(
            "normalization.fuzzy.enabled is true but no score thresholds are defined; "
            "default thresholds will be used"
        )

    if errors:
        bullet_list = "\n".join(f"- {message}" for message in errors)
        raise ConfigurationError(f"Configuration validation failed:\n{bullet_list}")

    return warnings


def create_scrapers_from_config(config: BettingConfig) -> list["SportsbookScraper"]:
    """Instantiate sportsbook scrapers defined in the configuration."""

    from .ingestion import SCRAPER_REGISTRY

    scrapers: list["SportsbookScraper"] = []
    for scraper_cfg in config.scrapers:
        if not scraper_cfg.enabled:
            continue
        scraper_type = scraper_cfg.type.lower()
        scraper_cls = SCRAPER_REGISTRY.get(scraper_type)
        if not scraper_cls:
            raise ValueError(f"Unknown scraper type: {scraper_cfg.type}")
        scraper = scraper_cls(**scraper_cfg.parameters)
        runtime = scraper_cfg.runtime
        if runtime.poll_interval_seconds is not None:
            scraper.poll_interval_seconds = runtime.poll_interval_seconds
        if runtime.retry_attempts is not None:
            scraper.retry_attempts = runtime.retry_attempts
        if runtime.retry_backoff is not None:
            scraper.retry_backoff = runtime.retry_backoff
        if runtime.timeout_seconds is not None:
            scraper.timeout_seconds = runtime.timeout_seconds
        scrapers.append(scraper)
    return scrapers


def create_ingestion_service(
    config: BettingConfig,
    *,
    alert_sink: "AlertSink" | None = None,
    storage_path: str | os.PathLike[str] | None = None,
    audit_logger: logging.Logger | None = None,
) -> "OddsIngestionService":
    """Build an :class:`OddsIngestionService` from configuration."""

    from .ingestion import OddsIngestionService

    stale_after = dt.timedelta(seconds=config.ingestion.stale_after_seconds)
    storage = str(storage_path or config.ingestion.storage_path)
    scrapers = create_scrapers_from_config(config)
    return OddsIngestionService(
        scrapers,
        storage_path=storage,
        stale_after=stale_after,
        alert_sink=alert_sink,
        audit_logger=audit_logger,
    )


def create_edge_detector(
    config: BettingConfig,
    *,
    alert_manager: "AlertManager" | None = None,
) -> "EdgeDetector":
    """Construct an :class:`EdgeDetector` with configuration defaults."""

    from .analytics import EdgeDetector

    analytics = config.analytics
    return EdgeDetector(
        value_threshold=analytics.value_threshold,
        alert_manager=alert_manager,
        backend=analytics.backend,
        correlation_penalty=analytics.correlation_penalty,
    )


def load_scope_scaling_model(
    config: BettingConfig,
    *,
    base_path: str | os.PathLike[str] | None = None,
) -> "ScopeScalingModel":
    """Load the scope scaling model referenced by configuration."""

    from .scope_scaling import ScopeScalingModel

    scope_cfg = config.models.scope_scaling
    baseline = dict(ScopeScalingModel.DEFAULT_FACTORS)
    for key, value in scope_cfg.fallback_factors.items():
        baseline[ScopeScalingModel.canonical_scope(key)] = float(value)

    parameters_path = scope_cfg.parameters_path
    resolved_path: Path | None = None
    if parameters_path:
        candidate = Path(parameters_path)
        if not candidate.is_absolute() and base_path is not None:
            base_root = Path(base_path)
            if base_root.is_file():
                base_root = base_root.parent
            candidate = base_root / candidate
        resolved_path = candidate

    model: ScopeScalingModel
    if resolved_path and resolved_path.exists():
        model = ScopeScalingModel.load(resolved_path, default_factors=baseline)
    else:
        model = ScopeScalingModel(base_factors=baseline, default_factors=baseline)

    if scope_cfg.overrides:
        model = model.with_overrides(scope_cfg.overrides)

    if scope_cfg.seasonal_overrides:
        for season, overrides in scope_cfg.seasonal_overrides.items():
            model = model.with_overrides(overrides, season=int(season))

    return model


__all__ = [
    "AnalyticsConfig",
    "BettingConfig",
    "ConfigurationError",
    "FuzzyMatchingConfig",
    "IngestionConfig",
    "IterationConfig",
    "NormalizationConfig",
    "SchedulerConfig",
    "ScopeScalingConfig",
    "ScraperConfig",
    "ScraperRuntimeConfig",
    "load_scope_scaling_model",
    "validate_betting_config",
    "create_edge_detector",
    "create_ingestion_service",
    "create_portfolio_optimizer",
    "create_scrapers_from_config",
    "load_betting_config",
]

