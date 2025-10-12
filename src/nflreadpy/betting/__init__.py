"""Sports betting analytics toolkit built on top of :mod:`nflreadpy` data access.

This module exposes high level orchestrators for the Bloomberg-style NFL
betting platform requested in ``AGENTS.md``.  It wires together the
asynchronous sportsbook scrapers, data ingestion layer, Monte Carlo
simulation engine, analytical edge detection utilities, and a lightweight
terminal dashboard for monitoring opportunities in near real-time.

The subpackages are intentionally granular so that researchers can extend
or swap components – for example, by registering new scrapers or
specialised simulation models – without rewriting the rest of the stack.
"""

from .analytics import (
    EdgeDetector,
    KellyCriterion,
    LineMovementAnalyzer,
    Opportunity,
    PortfolioManager,
    consolidate_best_prices,
)
from .compliance import (
    ComplianceConfig,
    ComplianceEngine,
    ResponsibleGamingControls,
)
from .dashboard import Dashboard
from .ingestion import OddsIngestionService
from .models import (
    GameSimulationConfig,
    MonteCarloEngine,
    PlayerPropForecaster,
    PlayerProjection,
    ProbabilityTriple,
    SimulationResult,
)
from .normalization import NameNormalizer
from .quantum import QuantumPortfolioOptimizer
from .scrapers.base import (
    MultiScraperCoordinator,
    OddsQuote,
    SportsbookScraper,
    best_prices_by_selection,
)
from .scrapers.mock import MockSportsbookScraper

__all__ = [
    "Dashboard",
    "ComplianceConfig",
    "ComplianceEngine",
    "EdgeDetector",
    "GameSimulationConfig",
    "KellyCriterion",
    "LineMovementAnalyzer",
    "MockSportsbookScraper",
    "MonteCarloEngine",
    "MultiScraperCoordinator",
    "NameNormalizer",
    "OddsIngestionService",
    "OddsQuote",
    "Opportunity",
    "PortfolioManager",
    "PlayerPropForecaster",
    "PlayerProjection",
    "QuantumPortfolioOptimizer",
    "ProbabilityTriple",
    "ResponsibleGamingControls",
    "SimulationResult",
    "SportsbookScraper",
    "best_prices_by_selection",
    "consolidate_best_prices",
]
