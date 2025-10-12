from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import polars as pl

from nflreadpy.betting.configuration import (
    BettingConfig,
    ConfigurationError,
    create_edge_detector,
    create_ingestion_service,
    create_scrapers_from_config,
    load_betting_config,
    load_scope_scaling_model,
    validate_betting_config,
)


def test_default_configuration_loads(tmp_path: Path) -> None:
    config = load_betting_config()
    assert isinstance(config, BettingConfig)
    assert config.scrapers, "expected at least one scraper configuration"

    scrapers = create_scrapers_from_config(config)
    assert scrapers, "scraper factory should build instances"

    service = create_ingestion_service(
        config,
        storage_path=tmp_path / "odds.sqlite3",
    )
    assert service.storage_path.exists()
    detector = create_edge_detector(config)
    assert detector.value_threshold == pytest.approx(config.analytics.value_threshold)


def test_configuration_layers_and_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "betting.yaml"
    base.write_text(
        """
scrapers:
  - type: mock
    enabled: true
    parameters: {}
ingestion:
  storage_path: base.sqlite3
analytics:
  value_threshold: 0.1
  bankroll: 500.0
"""
    )
    env_override = tmp_path / "betting.production.yaml"
    env_override.write_text(
        """
ingestion:
  storage_path: prod.sqlite3
analytics:
  value_threshold: 0.2
"""
    )
    extra_override = tmp_path / "override.yaml"
    extra_override.write_text(
        """
analytics:
  value_threshold: 0.3
"""
    )

    monkeypatch.setenv("NFLREADPY_BETTING_ENV", "production")
    monkeypatch.setenv("NFLREADPY_BETTING_CONFIG", str(extra_override))
    monkeypatch.setenv("NFLREADPY_BETTING__ingestion__storage_path", "env.sqlite3")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    config = load_betting_config(base_path=base)

    assert config.ingestion.storage_path == "env.sqlite3"
    assert config.analytics.value_threshold == pytest.approx(0.3)
    # Ensure other values still merge correctly
    assert config.analytics.bankroll == pytest.approx(500.0)


def test_environment_token_substitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "betting.yaml"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "data"))
    base.write_text(
        """
scrapers:
  - type: mock
    enabled: true
    parameters: {}
ingestion:
  storage_path: "${STORAGE_ROOT}/odds.sqlite3"
analytics:
  value_threshold: 0.15
"""
    )

    config = load_betting_config(base_path=base)
    expected = Path(os.environ["STORAGE_ROOT"]) / "odds.sqlite3"
    assert Path(config.ingestion.storage_path) == expected


def test_validate_default_configuration() -> None:
    config = load_betting_config()
    warnings = validate_betting_config(config)
    assert isinstance(warnings, list)


def test_validate_configuration_detects_disabled_scrapers() -> None:
    config = load_betting_config()
    disabled = [scraper.model_copy(update={"enabled": False}) for scraper in config.scrapers]
    broken = config.model_copy(update={"scrapers": disabled})
    with pytest.raises(ConfigurationError):
        validate_betting_config(broken)


def test_validate_configuration_emits_warnings() -> None:
    config = load_betting_config()
    tweaked_analytics = config.analytics.model_copy(update={"value_threshold": 0.001})
    tweaked = config.model_copy(update={"analytics": tweaked_analytics})
    warnings = validate_betting_config(tweaked)
    assert warnings, "expected warnings for very low value threshold"


def test_validate_config_cli(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "betting.yaml"
    env = {**os.environ, "DATA_DIR": str(tmp_path)}
    existing_path = env.get("PYTHONPATH")
    path_entries = [str(repo_root / "src")]
    if existing_path:
        path_entries.append(existing_path)
    env["PYTHONPATH"] = os.pathsep.join(path_entries)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nflreadpy.betting.cli",
            "validate-config",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Configuration" in result.stdout


def test_load_scope_scaling_model(tmp_path: Path) -> None:
    config = load_betting_config()

    frame = pl.DataFrame(
        {
            "scope": ["game", "1h", "2h"],
            "factor": [1.0, 0.55, 0.45],
            "samples": [8, 8, 8],
            "season": [None, None, None],
        }
    )
    destination = tmp_path / "scaling.parquet"
    frame.write_parquet(destination)

    scope_cfg = config.models.scope_scaling.model_copy(
        update={
            "parameters_path": destination.name,
            "overrides": {"4q": 0.22},
            "fallback_factors": {"3q": 0.25},
        }
    )
    models_cfg = config.models.model_copy(update={"scope_scaling": scope_cfg})
    tweaked = config.model_copy(update={"models": models_cfg})

    model = load_scope_scaling_model(tweaked, base_path=tmp_path)

    assert model("1h") == pytest.approx(0.55, rel=1e-6)
    assert model("2h") == pytest.approx(0.45, rel=1e-6)
    assert model("3q") == pytest.approx(0.25, rel=1e-6)
    assert model("4q") == pytest.approx(0.22, rel=1e-6)

