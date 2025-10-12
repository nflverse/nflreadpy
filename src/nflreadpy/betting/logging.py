"""Logging helpers for the betting toolkit."""

from __future__ import annotations

import logging
from typing import Iterable


def configure_logging(level: int = logging.INFO, handlers: Iterable[logging.Handler] | None = None) -> None:
    """Configure root logging for interactive sessions.

    The dashboard is expected to run continuously; structured logging makes
    it easier to diagnose scraper failures or simulation anomalies.  The
    helper can be called by applications embedding the toolkit to quickly
    establish a consistent format.
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=list(handlers) if handlers else None,
    )

