# CLAUDE.md - Development Guidelines

This file contains guidelines and patterns for maintaining and extending the nflreadpy package.

## Project Overview

nflreadpy is a Python port of the R package nflreadr, providing access to NFL data from nflverse repositories. The package uses modern Python conventions with Polars DataFrames, intelligent caching, and comprehensive type hints.

## Key Architecture Decisions

### Core Technologies
- **Python 3.10+** (not 3.9 - it's nearly end of life)
- **Polars** as the default DataFrame library (not pandas - much faster for large datasets)
- **uv** for package management and build system (not pip/setuptools)
- **Ruff** for both linting and formatting (not Black + separate linter)

### Package Structure
- Modular design with separate `load_*.py` files matching nflreadr's structure
- Separated utility modules (`utils_date.py`, not monolithic `utils.py`)
- Source layout: `src/nflreadpy/` (not flat package structure)

## Development Patterns

### Load Functions
When adding new load functions, follow this pattern:

1. **File Structure**: Create `src/nflreadpy/load_[function_name].py`
2. **Import Pattern**: 
   ```python
   from .utils_date import get_current_season  # NOT from .utils
   ```
3. **Season Logic**: 
   - Use `get_current_season()` for game data
   - Use `get_current_season(roster=True)` for roster/depth chart data
4. **URL Structure**: Check the actual nflreadr R files for correct paths
5. **Format Preference**: Default to Parquet, explicitly use CSV only where needed

### Data Format Preferences
- **Default**: Parquet (fastest, most efficient)
- **Fallback**: CSV (when Parquet unavailable)
- **Never**: RDS (R-specific format, not readable by Python)

### Season Validation
Always validate season ranges with appropriate minimum years:
```python
# Validate seasons
current_season = get_current_season()
for season in seasons:
    if not isinstance(season, int) or season < MIN_YEAR or season > current_season:
        raise ValueError(f"Season must be between {MIN_YEAR} and {current_season}")
```

### Testing
- Update `tests/test_integration.py` when adding new functions
- Add import tests for all new load functions
- Update the `expected_exports` list in `test_all_exports()`

## Common Mistakes to Avoid

1. **Import Paths**: Use `from .utils_date import get_current_season`, not `from .utils import`  
2. **Data Formats**: Don't default to CSV globally - only use it where Parquet isn't available
3. **R Dependencies**: Don't try to read RDS files - they're R-specific format

## URL Pattern Examples

Common nflverse data URL patterns:
- **Seasonal data**: `{repo}/releases/download/{category}/{name}_{season}.{format}`
- **Static data**: `{repo}/releases/download/{category}/{name}.{format}`
- **CSV files**: Use `format_preference=DataFormat.CSV` for known CSV-only sources

## Repository Structure

```
nflreadpy/
├── src/nflreadpy/
│   ├── __init__.py           # Main exports
│   ├── config.py             # Configuration management
│   ├── cache.py              # Caching system
│   ├── downloader.py         # HTTP client and data fetching
│   ├── utils_date.py         # Date utilities (separated)
│   └── load_*.py             # Individual load functions
├── tests/                    # Test suite
├── pyproject.toml           # uv + modern Python packaging
└── README.md                # User documentation
```

## Development Commands

```bash
# Install dependencies
uv sync --dev

# Format code
uv run ruff format

# Lint code  
uv run ruff check --fix

# Type check
uv run mypy src

# Run tests
uv run pytest

# Build package
uv build
```

## When Adding New Load Functions

1. **Research**: Check the corresponding R file in nflreadr for URL patterns
2. **Validate**: Ensure minimum season years are correct
3. **Test**: Add to integration tests
4. **Export**: Add to `__init__.py` imports and `__all__`
5. **Document**: Update README.md if it's a major function

## Performance Considerations

- Polars is much faster than pandas for large NFL datasets
- Use filesystem caching by default (configurable via environment)
- Implement progress bars for large downloads
- Batch multiple seasons efficiently with `pl.concat()`

This package aims to provide a modern, fast, and maintainable Python interface to NFL data while preserving API compatibility with the original nflreadr R package.