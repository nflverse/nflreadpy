# Configuration

::: nflreadpy.config.NflreadpyConfig
::: nflreadpy.config.update_config
::: nflreadpy.config.get_config
::: nflreadpy.config.reset_config

## Betting Compliance

The betting toolkit exposes configuration objects that can be populated from
YAML dictionaries or the process environment.  This makes it possible to apply
jurisdiction-specific rules without changing code when deploying the system in
regulated environments.

::: nflreadpy.betting.compliance.ComplianceConfig

::: nflreadpy.betting.compliance.ResponsibleGamingControls

::: nflreadpy.betting.compliance.ComplianceEngine

### YAML example

```yaml
betting:
  compliance:
    allowed_push_handling: ["push", "refund"]
    jurisdiction_allowlist:
      - nj
      - ny
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

### Environment variables

The ``ComplianceConfig.from_env`` factory understands the following variables
when the default ``NFLREADPY_COMPLIANCE_`` prefix is used:

* ``NFLREADPY_COMPLIANCE_ALLOWED_PUSH_HANDLING`` – comma separated list of
  accepted push handling treatments.
* ``NFLREADPY_COMPLIANCE_REQUIRE_OVERTIME_INCLUDED`` – toggle requiring markets
  to explicitly declare overtime handling (``1``/``0`` or ``true``/``false``).
* ``NFLREADPY_COMPLIANCE_JURISDICTION_ALLOWLIST`` – comma separated list of
  jurisdictions the bettor is authorised for.
* ``NFLREADPY_COMPLIANCE_BANNED_SPORTSBOOKS`` – sportsbook names to block.
* ``NFLREADPY_COMPLIANCE_REQUIRED_CREDENTIALS`` – JSON or semicolon-delimited
  map (``fanduel:session_token,account_id;pinnacle:api_key``) describing the
  credential fields that must be available per sportsbook.
* ``NFLREADPY_COMPLIANCE_CREDENTIALS_AVAILABLE`` – JSON or semicolon-delimited
  map indicating which credential fields are currently populated.

The responsible gaming factory listens for the ``NFLREADPY_RESPONSIBLE_``
prefix with keys ``SESSION_LOSS_LIMIT``, ``SESSION_STAKE_LIMIT`` and
``COOLDOWN_SECONDS`` to cap session losses and automatically trigger cooling
off periods in the :class:`~nflreadpy.betting.analytics.PortfolioManager`.
