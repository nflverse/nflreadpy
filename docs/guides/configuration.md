# Configuration Reference

nflreadpy centralises configuration through the :mod:`nflreadpy.config` helpers and the
layered betting settings parsed into the `BettingConfig` Pydantic model. This guide summarises
the key switches and how to combine configuration sources across environments.

## Core library settings

Use :func:`nflreadpy.config.update_config` to control caching, timeouts, and logging for the
data loading APIs:

```python
from nflreadpy.config import update_config

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

## Layered betting configuration

The betting stack reads YAML configuration via :func:`nflreadpy.betting.configuration.load_betting_config`.
Start from `config/betting.yaml`, optionally merge in environment-specific files, and finish with
environment-variable overrides.

```python
from nflreadpy.betting.configuration import load_betting_config, create_ingestion_service

config = load_betting_config(
    base_path="config/betting.yaml",
    environment="production",  # or set NFLREADPY_BETTING_ENV
    extra_paths=["/etc/nflreadpy/betting.d/overrides.yaml"],
)
service = create_ingestion_service(config)
```

The loader applies layers in the following order:

1. Load the base file (defaults to `config/betting.yaml`).
2. If the resolved environment (argument, `NFLREADPY_BETTING_ENV`, or the file's own
   ``environment`` field) is set, merge `config/betting.<env>.yaml` when present.
3. Merge any additional override files supplied via ``extra_paths`` or the
   `NFLREADPY_BETTING_CONFIG` environment variable (colon-separated list).
4. Apply overrides from environment variables prefixed with `NFLREADPY_BETTING__` where double
   underscores delimit nesting (for example ``NFLREADPY_BETTING__ingestion__storage_path=/data/odds.sqlite3``).
5. Resolve `${TOKEN}` placeholders inside YAML values using the process environment.

Environment overrides accept native types. Set ``NFLREADPY_BETTING__analytics__value_threshold=0.03``
to tighten the edge detector or ``NFLREADPY_BETTING__scrapers__0__runtime__poll_interval_seconds=30``
to slow a single scraper without touching files.

### Validating configuration

Before deploying a new layer, validate the merged output with the CLI:

```bash
uv run nflreadpy-betting validate-config \
  --config config/betting.yaml \
  --environment production
```

The command prints merged settings, warnings, and validation errors. Incorporate it into CI/CD
to guard against malformed overrides.
