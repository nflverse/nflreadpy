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

from .alerts import (
    AlertManager,
    EmailAlertSink,
    SMSAlertSink,
    SlackAlertSink,
    get_alert_manager,
)
from .analytics import (
    BankrollSimulationResult,
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
from .dashboard import (
    Dashboard,
    DashboardHotkey,
    DashboardSnapshot,
    RiskSummary,
    TerminalDashboardSession,
)
from .dashboard_core import DashboardFilters, DashboardSearchState
from .dashboard_tui import DashboardKeyboardController, run_curses_dashboard
from .ingestion import OddsIngestionService
from .models import (
    GameSimulationConfig,
    GLMPlayerModel,
    HistoricalGameRecord,
    MonteCarloEngine,
    NGBoostPlayerModel,
    PlayerPropForecaster,
    PlayerProjection,
    ProbabilityTriple,
    SimulationBenchmark,
    SimulationResult,
    XGBoostPlayerModel,
)
from .normalization import NameNormalizer
from .quantum import QuantumPortfolioOptimizer
from .scheduler import Scheduler
from .scrapers.base import (
    MultiScraperCoordinator,
    OddsQuote,
    SportsbookScraper,
    american_to_decimal,
    american_to_fractional,
    american_to_profit_multiplier,
    best_prices_by_selection,
    decimal_to_american,
    decimal_to_fractional,
    fractional_to_american,
    fractional_to_decimal,
    implied_probability_from_american,
    implied_probability_from_decimal,
    implied_probability_from_fractional,
    implied_probability_to_american,
    implied_probability_to_decimal,
    implied_probability_to_fraction,
)
from .scrapers.mock import MockSportsbookScraper

__all__ = [
    "AlertManager",
    "Dashboard",
    "DashboardKeyboardController",
    "DashboardHotkey",
    "DashboardFilters",
    "DashboardSnapshot",
    "DashboardSearchState",
    "TerminalDashboardSession",
    "ComplianceConfig",
    "ComplianceEngine",
    "EdgeDetector",
    "GameSimulationConfig",
    "GLMPlayerModel",
    "EmailAlertSink",
    "HistoricalGameRecord",
    "KellyCriterion",
    "LineMovementAnalyzer",
    "MockSportsbookScraper",
    "MonteCarloEngine",
    "NGBoostPlayerModel",
    "MultiScraperCoordinator",
    "NameNormalizer",
    "RiskSummary",
    "run_curses_dashboard",
    "OddsIngestionService",
    "OddsQuote",
    "Opportunity",
    "PortfolioManager",
    "BankrollSimulationResult",
    "PlayerPropForecaster",
    "PlayerProjection",
    "QuantumPortfolioOptimizer",
    "ProbabilityTriple",
    "SMSAlertSink",
    "SimulationBenchmark",
    "Scheduler",
    "SimulationResult",
    "SportsbookScraper",
    "SlackAlertSink",
    "XGBoostPlayerModel",
    "best_prices_by_selection",
    "decimal_to_american",
    "decimal_to_fractional",
    "fractional_to_american",
    "fractional_to_decimal",
    "get_alert_manager",
    "implied_probability_from_american",
    "implied_probability_from_decimal",
    "implied_probability_from_fractional",
    "implied_probability_to_american",
    "implied_probability_to_decimal",
    "implied_probability_to_fraction",
    "consolidate_best_prices",
]
