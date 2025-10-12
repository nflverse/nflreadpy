from __future__ import annotations
import os
from pathlib import Path

import pytest

from nflreadpy.betting.configuration import (
    BettingConfig,
    create_edge_detector,
    create_ingestion_service,
    create_scrapers_from_config,
    load_betting_config,
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

