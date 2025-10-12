# Compliance & Responsible Gaming

`nflreadpy.betting` ships with a lightweight compliance engine that can be
layered into any staking workflow.  The engine protects downstream portfolio
allocation by enforcing sportsbook market rules, jurisdiction availability, and
credential policies before a bet is sized.

## Configuration Sources

Compliance rules are driven by `ComplianceConfig` objects, which can be created
programmatically, from YAML/JSON configuration files, or through environment
variables.  YAML payloads map directly onto the dataclass fields:

```yaml
# compliance.yaml
allowed_push_handling:
  - push
  - refund
require_overtime_included: true
jurisdiction_allowlist:
  - nj
  - ny
banned_sportsbooks:
  - offshorebook
credential_requirements:
  fanduel:
    - session_token
credentials_available:
  fanduel:
    - session_token
```

The same configuration can be injected via environment variables using the
`NFLREADPY_COMPLIANCE_` prefix:

- `NFLREADPY_COMPLIANCE_ALLOWED_PUSH_HANDLING` – comma separated list
  (e.g. `push,refund`).
- `NFLREADPY_COMPLIANCE_REQUIRE_OVERTIME_INCLUDED` – set to `1`/`0` or
  `true`/`false`.
- `NFLREADPY_COMPLIANCE_JURISDICTION_ALLOWLIST` – comma separated list of
  permitted jurisdictions.
- `NFLREADPY_COMPLIANCE_BANNED_SPORTSBOOKS` – comma separated list of books to
  reject outright.
- `NFLREADPY_COMPLIANCE_REQUIRED_CREDENTIALS` – JSON object or
  semicolon-delimited string of `<sportsbook>:<comma separated credentials>`.
- `NFLREADPY_COMPLIANCE_CREDENTIALS_AVAILABLE` – same format as above, listing
  credentials that are already on file.
- `NFLREADPY_COMPLIANCE_REQUIRED_METADATA_FIELDS` – comma separated list of
  metadata keys (e.g. `jurisdictions,market_rules`) that must be populated on
  every opportunity before it can be staked.

Responsible gaming safeguards live in the companion
`ResponsibleGamingControls` dataclass and read from the
`NFLREADPY_RESPONSIBLE_` prefix:

- `NFLREADPY_RESPONSIBLE_SESSION_LOSS_LIMIT` – maximum loss allowed in a session
  before cooling down.
- `NFLREADPY_RESPONSIBLE_SESSION_STAKE_LIMIT` – cumulative stake cap for a
  session.
- `NFLREADPY_RESPONSIBLE_COOLDOWN_SECONDS` – cooling-off period applied when a
  limit is triggered.

## Audit Logging Hooks

Both the `ComplianceEngine` and higher-level services (such as the
`PortfolioManager` and `OddsIngestionService`) emit structured audit logs using
an `nflreadpy.betting.audit` logger by default.  Applications can supply their
own logger instances to capture compliance violations, rejected allocations, or
validation discards and forward them to SIEM tooling.

`PortfolioManager` emits `portfolio.rejected` warnings with a `reason` and, for
compliance failures, a structured `reasons` list describing each policy breach.
Likewise, `OddsIngestionService` populates `compliance_reasons` on
`ingestion.discarded` records whenever quotes are filtered out by the compliance
engine, making it easy to attribute rejections to specific jurisdiction or
credential gaps.

## Putting It Together

```python
from nflreadpy.betting import (
    ComplianceConfig,
    ComplianceEngine,
    PortfolioManager,
    ResponsibleGamingControls,
)

config = ComplianceConfig.from_mapping(load_yaml("compliance.yaml"))
controls = ResponsibleGamingControls.from_env()
manager = PortfolioManager(
    bankroll=10_000,
    compliance_engine=ComplianceEngine(config),
    responsible_gaming=controls,
)
```

With this wiring, any opportunity that fails the configured policies will be
rejected, logged, and excluded from bankroll deployment before a stake is
calculated.

## Ingestion-Time Enforcement

`OddsIngestionService` can also enforce the same policies while normalising
odds quotes.  Pass either a ready-made `ComplianceEngine` or a
`ComplianceConfig` instance and the ingestion pipeline will discard quotes that
violate jurisdiction rules, credential requirements, required metadata fields,
or market metadata before they reach persistence.  Rejections are surfaced
through the audit logger and the `last_validation_summary` metrics block for
downstream monitoring.
