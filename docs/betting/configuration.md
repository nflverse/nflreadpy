# Layered betting configuration

The betting toolkit reads configuration from a stack of YAML documents and
environment variables. This layered approach lets you version baseline defaults
in the repository while injecting environment-specific overrides and secrets at
deploy time.

## Configuration sources

Configuration is loaded in the following order:

1. **Base file** – `config/betting.yaml` ships with sane defaults for scrapers,
   ingestion, and analytics. Pass a different file with the `--config` flag or
   `base_path` argument when calling `load_betting_config`.
2. **Environment overlay** – Set `NFLREADPY_BETTING_ENV=production` (or any
   other name) to merge `config/betting.<env>.yaml` on top of the base file.
3. **Additional documents** – Provide extra override files via the
   `NFLREADPY_BETTING_CONFIG` environment variable or the
   `extra_paths` parameter. Multiple paths are accepted using the OS path
   separator.
4. **Environment variables** – Any environment variable beginning with
   `NFLREADPY_BETTING__` updates nested values. Tokens are split on double
   underscores and parsed as JSON when possible. For example:

   ```bash
   export NFLREADPY_BETTING__INGESTION__STORAGE_PATH="/data/odds.sqlite3"
   export NFLREADPY_BETTING__ANALYTICS__KELLY_FRACTION=0.4
   export NFLREADPY_BETTING__SCRAPERS__0__RUNTIME__POLL_INTERVAL_SECONDS=30
   ```

5. **Environment tokens** – Strings in the YAML files can reference other
   environment variables using `${VAR_NAME}`. Secrets stay outside source
   control while still flowing into the runtime configuration.

The merged payload is validated against `BettingConfig` and used to build
sportsbook scrapers, the ingestion service, and analytics components. All
initialisation helpers (`create_scrapers_from_config`,
`create_ingestion_service`, `create_edge_detector`) read from the same object,
ensuring consistent defaults across the CLI, background workers, and tests.

## Validating configuration

Use the CLI to verify that a configuration stack is internally consistent
before deploying:

```bash
uv run nflreadpy-betting validate-config --config config/betting.yaml \
  --environment production
```

The validator checks for missing or disabled scrapers, negative polling
intervals, out-of-range bankroll and Kelly fractions, and ensures Monte Carlo
iteration counts are positive. Warnings are emitted for suspicious thresholds;
pass `--warnings-as-errors` to fail when warnings are encountered.

Programmatic validation is also available:

```python
from nflreadpy.betting.configuration import load_betting_config, validate_betting_config

config = load_betting_config(environment="production")
warnings = validate_betting_config(config)
if warnings:
    for message in warnings:
        print(f"⚠️ {message}")
```

## Managing secrets

Secrets are best supplied through environment variables and substituted into the
YAML files via `${VAR}` placeholders. Combine this with
`NFLREADPY_BETTING_CONFIG` overlays to keep production-only keys outside the
repository. The GitHub Actions workflow, Kubernetes secrets, or any process
manager that supports environment injection can populate these values at
runtime.

## Related documentation

* [Deployment guide](deployment.md) – Covers environment bootstrapping, secrets
  management, and container workflows in more detail.
* [CLI reference](cli.md) – Documents the `nflreadpy-betting` commands that use
  the layered configuration system.
