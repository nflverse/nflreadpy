# Utility Helpers

Two convenience functions expose season and week metadata derived from the
current calendar date.

## `get_current_season()`

Returns the integer season that should be considered "current". The helper rolls
over to the new season at the start of August to align with the nflverse data
release schedule.

```python
import nflreadpy as nfl

season = nfl.get_current_season()
print(f"Working with season {season}")
```

## `get_current_week()`

Returns a tuple of `(season, week)` representing the active NFL week. During the
off-season the function returns `(season, None)` until pre-season data becomes
available.

```python
import nflreadpy as nfl

season, week = nfl.get_current_week()
print(season, week)
```

Both helpers are timezone aware and rely on the nflverse calendar definitions.
