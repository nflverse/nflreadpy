# Betting dashboards

The betting module ships with two complementary dashboards:

- `nflreadpy.betting.dashboard.Dashboard` — an interactive terminal view with
  Bloomberg-style panels and ASCII ladder matrices.
- `nflreadpy.betting.web.run_dashboard` — a Streamlit application that exposes
  live markets, line histories, calibration diagnostics, and portfolio stats via
  the browser.

## Terminal dashboard

The terminal dashboard accepts raw odds quotes, simulation results, and detected
opportunities.  Panels can be toggled on/off, reordered, and filtered.

```python
from nflreadpy.betting.dashboard import Dashboard

# odds, simulations, opportunities collected elsewhere
terminal = Dashboard()
terminal.set_filters(sportsbooks={"BookA", "BookB"})
terminal.toggle_quarters()  # hide first/second/third/fourth quarter markets
print(terminal.render(odds, simulations, opportunities))
```

Key features:

- **Filter aware panels** – the control panel at the top shows the active
  filters and toggles for quarter/half markets.
- **Interactive panel state** – call `toggle_panel("ladders")` to collapse the
  line ladder matrix or `reorder_panels([...])` to rearrange sections.
- **Ladder matrices** – the ladder panel aggregates lines by scope and
  selection, making it easy to scan alt prices across books.

## Streamlit dashboard

The web dashboard uses the same filtering logic so the browser view behaves the
same way as the terminal view.  Provide a data provider implementing the
`DashboardDataProvider` protocol:

```python
from nflreadpy.betting.web import DashboardDataProvider, run_dashboard

class LiveProvider(DashboardDataProvider):
    ...  # implement live_markets, line_history, calibration, portfolio

run_dashboard(LiveProvider())
```

Sidebar controls allow users to select sportsbooks, market groups, markets,
scopes, events, and toggle quarter/half markets.  The UI renders:

- **Live markets** — a table of the most recent quotes that respect the active
  filters.
- **Model opportunities** — optional, surfaced when the provider implements the
  `opportunities()` method.
- **Line movement charts** — Vega-Lite powered visuals showing how prices shift
  over time.
- **Calibration scatter plots** — compare observed hit rates against model
  expectations for each bucket.
- **Portfolio view** — tabular position tracking with total stake and realized
  PnL summaries.

Streamlit is an optional dependency.  Install it with `uv add streamlit` (or
`pip install streamlit`) before calling `run_dashboard`.

## Testing and automation

Integration tests covering both dashboards live in
`tests/betting/test_dashboard_interfaces.py`.  They exercise rendering,
filtering, ladder generation, Streamlit wiring, and data preparation helpers.
Use `uv run pytest tests/betting/test_dashboard_interfaces.py` to focus on the
new suites.

