# CLI Command Reference

The `nflreadpy-betting` console script wraps ingestion, modelling, and reporting workflows. All
commands accept the shared options `--config`, `--environment`, `--storage`, and
`--alerts-config` to control configuration sources and alert routing. When omitted, the CLI reads
`config/betting.yaml` and honours the layered merge order described in the [configuration guide](configuration.md).

## `ingest`

Continuously poll configured scrapers and persist odds to the storage backend.

```bash
uv run nflreadpy-betting ingest --interval 5 --retries 3 --retry-backoff 1.5
```

| Flag | Description |
| --- | --- |
| `--interval` | Polling cadence in seconds. |
| `--jitter` | Random jitter applied per poll to avoid thundering herd issues. |
| `--retries` | Maximum retry attempts before surfacing a failure. |
| `--retry-backoff` | Backoff multiplier (in seconds) applied between retries. |

## `simulate`

Run Monte Carlo simulations over the most recent odds snapshot and surface edge opportunities.

```bash
uv run nflreadpy-betting simulate --iterations 20000 --value-threshold 0.02
```

| Flag | Description |
| --- | --- |
| `--iterations` | Number of Monte Carlo runs per event. |
| `--bankroll` | Bankroll amount used when sizing positions. |
| `--refresh` | Force a fresh ingestion cycle before simulating. |
| `--value-threshold` | Minimum expected value required to report an opportunity. |
| `--history-limit` | Number of historical quotes to load for movement context. |
| `--movement-threshold` | Minimum basis point change to flag as movement. |
| `--kelly-fraction` | Overrides the default Kelly fraction from config. |
| `--portfolio-fraction` | Caps capital allocated in a single run. |
| `--correlation-limit` | Define exposure limits per correlation group (`group=value`). |
| `--risk-trials` | Run bankroll simulations to understand drawdown risk. |
| `--risk-seed` | Seed applied to bankroll simulations. |

## `scan`

Inspect previously stored odds to identify drift, stale markets, and open alerts.

```bash
uv run nflreadpy-betting scan --history-limit 250 --movement-threshold 25
```

| Flag | Description |
| --- | --- |
| `--iterations` | Optional Monte Carlo iterations for fresh simulations. |
| `--history-limit` | Maximum historical quotes loaded per market. |
| `--value-threshold` | Filters edges below the expected value threshold. |
| `--movement-threshold` | Minimum price change (basis points) to treat as movement. |
| `--kelly-fraction` | Override Kelly sizing for reported edges. |
| `--correlation-limit` | Exposure limits per correlation group. |

## `dashboard`

Render the ASCII terminal dashboard with live odds, model output, and risk summaries.

```bash
uv run nflreadpy-betting dashboard --refresh
```

| Flag | Description |
| --- | --- |
| `--iterations` | Override default simulation iterations. |
| `--refresh` | Trigger ingestion before the dashboard launches. |
| `--value-threshold` | Minimum expected value shown in the opportunities panel. |
| `--bankroll` | Bankroll figure used in risk summaries. |
| `--portfolio-fraction` | Cap per-run capital deployment. |
| `--kelly-fraction` | Override Kelly fraction. |
| `--correlation-limit` | Exposure limits per correlation group. |
| `--risk-trials` | Number of Monte Carlo bankroll trials for risk stats. |
| `--risk-seed` | Seed for bankroll trials. |

## `backtest`

Replay historical ingestion data to compute realised value and bankroll swings.

```bash
uv run nflreadpy-betting backtest --limit 200 --iterations 5000
```

| Flag | Description |
| --- | --- |
| `--limit` | Number of stored snapshots to evaluate. |
| `--iterations` | Monte Carlo iterations used to rebuild simulations. |
| `--value-threshold` | Minimum expected value to include in the report. |

## `validate-config`

Merge and validate layered configuration files without starting services. The command prints the
effective configuration, warnings, and any fatal validation errors.

```bash
uv run nflreadpy-betting validate-config \
  --config config/betting.yaml \
  --environment staging
```

Use it in pre-commit hooks or CI to guard against malformed YAML and incompatible overrides.

Combine the commands in scripts or schedulers to automate ingestion, modelling, alerting, and
reporting loops.
