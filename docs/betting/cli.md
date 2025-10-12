# Betting Toolkit CLI

The `nflreadpy-betting` command-line interface orchestrates sportsbook scraping,
Monte Carlo analytics, and reporting flows. Install the betting extra before
invoking the CLI:

```bash
uv add "nflreadpy[betting]"
```

## Command overview

| Command | Purpose |
| --- | --- |
| `nflreadpy-betting validate-config` | Lint layered configuration files and surface warnings before deployment. |
| `nflreadpy-betting ingest` | Run the ingestion scheduler or perform a one-shot scrape for configured sportsbooks. |
| `nflreadpy-betting simulate` | Fetch the latest odds, simulate matchups, and rank opportunities. |
| `nflreadpy-betting scan` | Analyse stored history for edges without triggering fresh scrapes. |
| `nflreadpy-betting dashboard` | Render the terminal dashboard summarising exposure and bankroll simulations. |
| `nflreadpy-betting backtest` | Replay historical odds snapshots to estimate cumulative expected value. |

Every command inherits the shared options `--config`, `--environment`,
`--storage`, and `--alerts-config`, making it easy to point the CLI at specific
configuration stacks or alerting backends.

## Configuration validation

Before rolling configuration changes into CI/CD pipelines, run the validator:

```bash
uv run nflreadpy-betting validate-config --config config/betting.yaml \
  --environment production --warnings-as-errors
```

Validation raises on missing scrapers, negative retry settings, invalid bankroll
parameters, and non-positive Monte Carlo iteration counts. Suspicious but
technically valid settings (for example, extremely low value thresholds) emit
warnings; pass `--warnings-as-errors` to treat them as failures.

## Ingestion

Launch the ingestion scheduler with environment defaults:

```bash
uv run nflreadpy-betting ingest --interval 60 --jitter 5
```

Setting `--interval 0` (or omitting the flag when the configuration already
specifies zero) performs a one-shot scrape and exits. The command emits summary
metrics and surfaces alerts via the configured alert manager.

## Simulation workflows

Use the `simulate` command to run fresh Monte Carlo simulations against the
latest odds snapshot:

```bash
uv run nflreadpy-betting simulate --iterations 25000 --value-threshold 0.03
```

When `--refresh` is omitted the command reuses stored odds; otherwise it fetches
new quotes before running the models. The resulting opportunities include
expected value, Kelly stakes, and correlation-aware portfolio recommendations.

For offline analysis, `scan` skips the live scrape and analyses the configured
history window:

```bash
uv run nflreadpy-betting scan --history-limit 500 --movement-threshold 20
```

## Dashboards and backtesting

Render the terminal dashboard (a textual Bloomberg-style view) with:

```bash
uv run nflreadpy-betting dashboard --refresh --iterations 20000
```

The dashboard aggregates current odds, model outcomes, bankroll risk summaries,
and exposure reports. For historical analysis, the backtest command replays a
fixed number of stored snapshots and reports aggregate expected value:

```bash
uv run nflreadpy-betting backtest --limit 250 --iterations 15000
```

Each subcommand respects the layered configuration defaults, meaning flags are
optional unless you want to override per-run behaviour.
