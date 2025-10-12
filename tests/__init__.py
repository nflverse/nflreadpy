"""Test suite for nflreadpy."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from pydantic import ConfigDict  # noqa: F401  # re-exported check only
except ImportError as exc:  # pragma: no cover - import-time guard
    msg = (
        "pydantic>=2 is required for the test suite; "
        "install a compatible version that exposes pydantic.ConfigDict."
    )
    raise RuntimeError(msg) from exc

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
