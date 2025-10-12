# Configuration

nflreadpy centralises configuration through the `NflreadpyConfig` dataclass and
helper functions that adjust caching, HTTP behaviour, and verbosity at runtime.

## Core settings

The configuration exposes the following keys:

| Field | Type | Description |
| --- | --- | --- |
| `cache_mode` | choices: `"memory"`, `"filesystem"`, `"off"` | Controls whether responses are cached in memory, written to disk, or disabled. |
| `cache_path` | `Path` or `None` | Filesystem location used when `cache_mode="filesystem"`. |
| `cache_duration` | `int` | Time-to-live in seconds before cached entries are refreshed. |
| `request_timeout` | `int` | Timeout passed to `requests` when downloading datasets. |
| `user_agent` | `str` | Custom user-agent string for outbound HTTP requests. |
| `verbose` | `bool` | Enables progress bars and debug logging during downloads. |

Retrieve and mutate the configuration with the convenience helpers:

```python
import nflreadpy as nfl

config = nfl.get_config()
print(config.cache_mode)

nfl.update_config(cache_mode="filesystem", cache_path=".cache/nflreadpy", cache_duration=600)
```

Call `nfl.reset_config()` to restore defaults or set environment variables
(`NFLREADPY_CACHE`, `NFLREADPY_CACHE_DIR`, `NFLREADPY_CACHE_DURATION`, etc.)
before importing the package to override values globally.

## Betting compliance extensions

The betting module adds compliance and responsible gaming settings that are
loaded from configuration files or the environment. These settings govern which
sportsbooks may be queried, how pushes are handled, and what credentials must be
present before placing wagers.

Example YAML snippet:

```yaml
betting:
  compliance:
    allowed_push_handling: ["push", "refund"]
    jurisdiction_allowlist: ["nj", "ny"]
    credential_requirements:
      fanduel: ["session_token", "account_id"]
      draftkings: ["api_key"]
    credentials_available:
      fanduel:
        - session_token
  responsible_gaming:
    session_loss_limit: 250.0
    session_stake_limit: 500.0
    cooldown_seconds: 900
```

Environment overrides follow the prefix `NFLREADPY_COMPLIANCE_` (for compliance
fields) and `NFLREADPY_RESPONSIBLE_` (for responsible gaming). For example:

```bash
export NFLREADPY_COMPLIANCE_JURISDICTION_ALLOWLIST=nj,ny
export NFLREADPY_RESPONSIBLE_SESSION_LOSS_LIMIT=250
```

These settings integrate with the betting portfolio manager to enforce exposure
limits and pause betting activity when cooling-off rules are triggered.
