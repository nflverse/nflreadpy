# Betting Dashboards

nflreadpy ships with two complementary dashboards for monitoring real-time
markets, simulation output, and portfolio risk.  The terminal experience is
optimised for analysts who prefer a keyboard-driven workflow, while the web
interface focuses on rich visualisations for market surveillance.

## Terminal dashboard

The `Dashboard` class renders ASCII panels summarising active simulations,
recent sportsbook quotes, detected opportunities, and ladder matrices across
available scopes.  A `TerminalDashboardSession` wraps the class with a
command-driven interface that supports filters, panel toggles, and
full-text search across odds, simulations, and opportunities.

### Launching an interactive session

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
provider.

Launch a development server with:

```bash
uvicorn nflreadpy.betting.web.api:create_api_app --factory
```

The factory expects a `DashboardDataProvider` instance supplied via the
`--factory` hook; see the module docstrings for an example provider.
