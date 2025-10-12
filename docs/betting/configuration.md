# Betting Toolkit Configuration

Configuration is centralised through `nflreadpy.get_config()` and the betting
module's own settings registry. Use the following layers to control runtime
behaviour and deployment-specific secrets.

## Global nflreadpy config

The base library exposes a mutable configuration object that affects caching,
HTTP behaviour, and logging. Review the [API reference](../api/configuration.md)
and focus on the options most relevant to betting workloads:

| Setting | Description | Typical value |
| --- | --- | --- |
| `cache_mode` | Enables in-memory, filesystem, or disabled caching. | `filesystem` for shared deployments |
| `cache_path` | Filesystem location for cached market pulls and simulations. | `.cache/nflreadpy` |
| `cache_duration` | Seconds before cached responses expire. | `180`–`600` |
| `request_timeout` | Timeout passed to `requests` when hitting sportsbook APIs. | `10` |
| `user_agent` | Identifies the client when calling third-party APIs. | Custom string per organisation |

Update config values at runtime:

```python
import nflreadpy as nfl

nfl.update_config(cache_mode="filesystem", cache_path=".cache/nflreadpy", cache_duration=300)
```

Persist organisation defaults by wrapping the call inside your bootstrap code or
using an environment-aware module that executes during worker start-up.

## Betting settings file

The betting module reads extended configuration from a declarative YAML or JSON
file. Point the `NFLREADPY_BETTING_CONFIG` environment variable at the document
when launching services:

```bash
export NFLREADPY_BETTING_CONFIG=config/betting.yaml
```

A minimal `betting.yaml` looks like:

```yaml
ingestion:
  sportsbooks:
    - name: pinnacle
      markets: [moneyline, spread, total]
      regions: [us]
      polling_interval: 30
    - name: draftkings
      markets: [moneyline, spread, player_prop]
      regions: [us]
      polling_interval: 20
  storage:
    url: postgresql://nflreadpy:secret@localhost:5432/nflreadpy
    schema: betting
analytics:
  models:
    - module: nflreadpy.betting.models.elo
      kwargs:
        k_factor: 22
    - module: nflreadpy.betting.models.simulator
      kwargs:
        iterations: 10000
  portfolio:
    bankroll: 10000
    max_fraction: 0.03
notifications:
  slack:
    webhook_url: https://hooks.slack.com/services/.../...
  email:
    sender: bettingbot@example.com
    recipients:
      - ops@example.com
```

### Overrides

Override any key using environment variables prefixed with `NFLREADPY_BETTING__`
and upper-cased path segments. For example, to change the bankroll in
containerised deployments:

```bash
export NFLREADPY_BETTING__ANALYTICS__PORTFOLIO__BANKROLL=25000
```

Dotted keys map to nested dictionaries, and lists accept JSON payloads:

```bash
export NFLREADPY_BETTING__INGESTION__SPORTSBOOKS='[{"name":"pinnacle","markets":["moneyline"]}]'
```

## Secrets management

Avoid storing API keys or database credentials inside source control. Recommended
patterns include:

- Injecting secrets through your orchestrator (systemd, Kubernetes, Airflow).
- Using a secrets manager such as AWS Secrets Manager or HashiCorp Vault and
  populating environment variables at runtime.
- Encrypting configuration files with `sops` and decrypting during deployment.

The ingestion and analytics services respect the `NFLREADPY_BETTING_SECRET_*`
namespace for ad-hoc secrets you wish to expose programmatically.

## Configuration validation

Run the built-in validator before deploying configuration changes:

```bash
uv run nflreadpy-betting validate-config --path config/betting.yaml
```

The command checks schema compatibility, verifies connection strings, and warns
when polling intervals or bankroll parameters exceed recommended thresholds.
