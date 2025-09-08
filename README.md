# nflreadpy

A Python package for downloading NFL data from nflverse repositories. This is a Python port of the popular R package [nflreadr](https://github.com/nflverse/nflreadr), designed to provide easy access to NFL data with caching, progress tracking, and modern Python conventions.

## Features

- **Fast data loading** with Polars DataFrames
- **Intelligent caching** (memory or filesystem)
- **Progress tracking** for large downloads
- **Modern Python** (3.10+) with type hints
- **Compatible API** with nflreadr R package

## Installation

```bash
# Using uv (recommended)
uv add nflreadpy

# Using pip
pip install nflreadpy
```

## Quick Start

```python
import nflreadpy as nfl

# Load current season play-by-play data
pbp = nfl.load_pbp()

# Load multiple seasons
pbp_multi = nfl.load_pbp([2022, 2023])

# Load all available data
pbp_all = nfl.load_pbp(seasons=True)

# Load current season player stats
player_stats = nfl.load_player_stats()

# Load team stats
team_stats = nfl.load_team_stats()

# Load rosters
rosters = nfl.load_rosters()

# Load schedules
schedules = nfl.load_schedules()
```

## Available Functions

### Core Loading Functions

- `load_pbp()` - Play-by-play data
- `load_player_stats()` - Player statistics
- `load_team_stats()` - Team statistics  
- `load_rosters()` - Team rosters
- `load_schedules()` - Game schedules

### Utility Functions

- `get_current_season()` - Get current NFL season
- `get_current_week()` - Get current NFL week
- `clear_cache()` - Clear cached data

## Configuration

Configure nflreadpy using environment variables:

```bash
# Cache settings
export NFLREADPY_CACHE=filesystem  # "memory", "filesystem", or "off"
export NFLREADPY_CACHE_DIR=/path/to/cache

# Data preferences  
export NFLREADPY_PREFER=parquet    # "parquet" or "csv"

# Behavior
export NFLREADPY_VERBOSE=true     # Show progress messages
export NFLREADPY_TIMEOUT=30       # Request timeout in seconds
```

Or configure programmatically:

```python
from nflreadpy.config import update_config

update_config(
    cache_mode="memory",
    verbose=False,
    prefer_format="csv"
)
```

## Data Sources

nflreadpy downloads data from the following nflverse repositories:

- [nflverse-data](https://github.com/nflverse/nflverse-data) - Play-by-play, rosters, stats
- [nfldata](https://github.com/nflverse/nfldata) - Schedules and game data
- [espnscrapeR-data](https://github.com/nflverse/espnscrapeR-data) - ESPN QBR data
- [dynastyprocess](https://github.com/dynastyprocess/data) - Draft and contract data
- [ffopportunity](https://github.com/ffverse/ffopportunity) - Fantasy football data

## Why Polars?

nflreadpy uses [Polars](https://pola.rs/) instead of pandas for better performance:

- **Faster processing** of large NFL datasets
- **Lower memory usage** 
- **Better type system** with lazy evaluation
- **Modern API** designed for performance

Convert to pandas if needed:
```python
df_pandas = df.to_pandas()
```

## Development

This project uses modern Python tooling:

- **uv** for dependency management
- **Ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run ruff format

# Type check
uv run mypy src
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- The [nflverse community](https://github.com/nflverse) for providing the data infrastructure
- The original [nflreadr](https://github.com/nflverse/nflreadr) R package authors