# Betting Dashboards

nflreadpy ships with two complementary dashboards for monitoring real-time
markets, simulation output, and portfolio risk. Before launching either
experience make sure you have followed the [setup](setup.md) and
[configuration](configuration.md) guides so that ingestion workers and models
are providing fresh data.

The terminal experience is optimised for analysts who prefer a keyboard-driven
workflow, while the web interface focuses on rich visualisations for market
surveillance.

Both front ends share a common state layer implemented in
`nflreadpy.betting.dashboard_core`.  The module exposes
`DashboardFilters`, `DashboardSearchState`, and the ladder matrix utilities used
to render line ladders.  Custom automation can import these primitives directly
to apply consistent filter logic when running bespoke reports or building
alternative user interfaces.

## Terminal dashboard

The `Dashboard` class renders ASCII panels summarising active simulations,
recent sportsbook quotes, detected opportunities, ladder matrices, and risk
exposure.  Two interactive front ends ship with the library:

1. **Command interpreter** – `TerminalDashboardSession` exposes `show`,
   `filter`, `toggle`, `search`, and `order` commands for analysts who prefer
   typewritten workflows.
2. **Curses layout** – `run_curses_dashboard` and
   `DashboardKeyboardController` provide a Bloomberg-style pane layout with
   keyboard navigation, inline filter prompts, scope toggles, and real-time
   search.

### Launching the command interpreter

```python
from nflreadpy.betting import Dashboard, TerminalDashboardSession
from nflreadpy.betting.ingestion import IngestedOdds

session = TerminalDashboardSession(Dashboard())
print(session.handle("help", odds, simulations, opportunities))
```

The interpreter understands the following commands:

- `show` — render the dashboard using the current filters.
- `filter key=value` — constrain by sportsbook, market group, market, scope,
  or event (comma separated values are supported).
- `toggle quarters|halves` — include or exclude partial game scopes.
- `toggle panel <key>` — collapse or expand individual panels.
- `search <query> [--case] [--in quotes,opportunities,simulations]` — apply a
  search term across one or more data sources.
- `clear search` — remove the active search query.
- `order controls,quotes,...` — customise the panel ordering.
- `reset` — clear filters and the search state.

### Launching the curses interface

```python
from nflreadpy.betting import DashboardKeyboardController, run_curses_dashboard

# ``feed`` must implement the DashboardFeed protocol (see ``dashboard_tui``).
run_curses_dashboard(feed, refresh_seconds=3.0)
```

The curses UI supports the following keys:

- `Tab` / `Shift+Tab` — cycle between panels.
- `Space` or `Enter` — collapse/expand the focused panel.
- `f` — open an inline prompt for filter expressions
  (e.g. `sportsbooks=FanDuel markets=spread`).
- `/` — open the search prompt (the query is shared with the command
  interpreter and the Streamlit dashboard).
- `Q` / `H` — toggle quarter and half scopes respectively.
- `c` — reset filters to their defaults.
- `n` — clear the search query.
- `r` — force a data refresh immediately.
- `?` — show the key bindings in the status bar.
- `q` or `Esc` — exit the dashboard.

The rendered output includes a dedicated **Search Results** panel that
summarises matches across the selected datasets, making it straightforward to
locate specific players, events, or markets without leaving the terminal.

## Streamlit dashboard

The Streamlit implementation lives in `nflreadpy.betting.web.app` and expects a
`DashboardDataProvider` that supplies live market quotes, model opportunities,
line history points, calibration diagnostics, and portfolio positions.  The
page renders:

- a sortable table of live markets filtered using the same controls as the
  terminal dashboard;
- a model opportunity table when edges are detected;
- Vega-Lite charts for line movement and probability calibration; and
- portfolio analytics summarising stake, expected value, Kelly sizing, and
  realised PnL.

An auto-refresh slider keeps the Streamlit view streaming new quotes,
opportunities, and analytics in near real time.  If Streamlit provides the
`st.autorefresh` helper (v1.27+), the page automatically re-runs at the chosen
interval.  The sidebar also exposes a **Refresh now** button that triggers
`st.experimental_rerun()` for manual updates.

Run the dashboard with:

```bash
uv run streamlit run -m nflreadpy.betting.web.app
```

Optional dependencies (`streamlit`, `fastapi`, and `uvicorn`) can be installed
via:

```bash
uv add "nflreadpy[betting]"
```

## FastAPI service

For programmatic integrations, `nflreadpy.betting.web.create_api_app` builds a
FastAPI application that mirrors the Streamlit data access patterns.  The API
exposes `/markets`, `/opportunities`, `/line-history`, `/calibration`,
`/portfolio`, and `/filters` endpoints, returning JSON payloads derived from the
provider.  A `/markets/stream` endpoint emits newline-delimited JSON snapshots
at a caller-defined cadence, allowing downstream services to stream updates into
message queues or WebSocket hubs without polling.

Launch a development server with:

```bash
uvicorn nflreadpy.betting.web.api:create_api_app --factory
```

The factory expects a `DashboardDataProvider` instance supplied via the
`--factory` hook; see the module docstrings for an example provider.

## Operational tips

- Use the [CLI utilities](cli.md) to inspect opportunities or export data when
  the dashboards are not available.
- Review the [operations guide](operations.md) for recommended metrics and
  alerting rules that keep the dashboards healthy.
