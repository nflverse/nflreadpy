# Configuration Reference

nflreadpy centralises configuration through the `BettingConfig` dataclass and the global
`update_config` helpers exposed in `nflreadpy.config`. This guide lists the most common knobs
for end-to-end deployments.

## Core library settings

Use `nflreadpy.update_config` to control caching, timeouts, and logging:

```python
from nflreadpy import update_config

update_config(
    cache_mode="filesystem",
    cache_dir=".cache/nflreadpy",
    cache_duration=300,
    timeout=20,
    verbose=False,
    user_agent="my-company-betting-bot/1.0",
)
```

| Option | Description |
| --- | --- |
| `cache_mode` | `memory`, `filesystem`, or `off` depending on persistence needs. |
| `cache_dir` | Directory used for filesystem cache entries. |
| `cache_duration` | Freshness window (seconds) before a cached asset is re-fetched. |
| `timeout` | HTTP timeout applied to downloader requests. |
| `verbose` | Enables progress bars and debug logging for downloads. |
| `user_agent` | Overrides the default HTTP user agent. |

## Betting configuration file

The betting stack reads structured configuration from `config/betting.toml` by default. Load
it with `load_betting_config()` and feed it to helper constructors.

```python
from nflreadpy.betting.configuration import load_betting_config, create_ingestion_service

config = load_betting_config("config/betting.toml")
ingestion = create_ingestion_service(config)
```

Key sections inside the TOML file include:

- `[ingestion]` – Enables or disables specific scrapers, sets poll intervals, and configures
  retry backoff.
- `[analytics]` – Controls Monte Carlo iterations, Kelly fraction defaults, and bankroll guard
  rails.
- `[alerts]` – Declares alert routing (Slack, email, PagerDuty) and severity thresholds.
- `[dashboards]` – Defines Streamlit/terminal layout preferences and update cadence.

## Overriding at runtime

The CLI accepts overrides via flags or environment variables. Example:

```bash
uv run nflreadpy-betting run \
  --config config/prod.toml \
  --refresh-interval 30 \
  --kelly-fraction 0.6 \
  --enable-alerts
```

Use overrides for incident response or rapid experimentation. Persist long-term changes in
source-controlled configuration files to keep environments reproducible.
