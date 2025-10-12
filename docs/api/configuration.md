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
