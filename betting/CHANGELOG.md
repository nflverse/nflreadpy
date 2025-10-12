# nflreadpy Betting Changelog

This changelog tracks updates that are specific to the optional betting
workflows that ship with **nflreadpy**.  Entries here are intended to
highlight sportsbook ingestion, analytics, and dashboard changes without
cluttering the core library notes.

## Unreleased

- _No unreleased changes._

## v0.1.4 (2025-09-28)

### Highlights

- Stabilised the Bloomberg-style betting workstation by tightening the
  coordination between asynchronous sportsbook scrapers, the
  `OddsIngestionService`, and the `Scheduler` so that live odds move
  through the pipeline in under a second.
- Expanded the analytics layer with robust bankroll and portfolio
  simulations, Kelly staking utilities, and price consolidation helpers
  that flow directly into the terminal and web dashboards.
- Hardened operational tooling including alert fan-out (email, SMS,
  Slack), compliance guard-rails, and the curses-based monitoring
  experience exposed via `TerminalDashboardSession`.

## v0.1.0 (2025-08-15)

### Initial release

- Introduced the modular sportsbook scrapers, ingestion layer, Monte Carlo
  simulation engine, and player projection models that power betting edge
  detection.
- Added the Streamlit/terminal dashboards, quantum portfolio optimiser,
  and supporting utilities for odds format conversions and name
  normalisation.
- Delivered compliance and responsible gaming controls plus alert routing
  sinks to round out the production-ready toolkit.
