"""Convert game-level data to team-level data."""

import re

import polars as pl


def clean_homeaway(
    dataframe: pl.DataFrame, invert: list[str] | None = None
) -> pl.DataFrame:
    """
    Convert home/away game rows into team/opponent rows.

    Args:
        dataframe:
            A Polars DataFrame with matching home/away columns. Prefixes
            `home_` and `away_` become `team_` and `opponent_`, with
            `home_team` and `away_team` becoming `team` and `opponent`.
            Suffixes `_home` and `_away` become no suffix and `_opponent`.
            The names are reversed for the away team's row.
        invert:
            Optional list of signed numeric output columns to negate for the
            away team, such as `result` or `spread_line`. All-null columns are
            left unchanged.

    Returns:
        A new DataFrame with two rows per input row, home team first, in input
        order. Other columns are repeated unchanged. `location` is set to
        `home` or `away`, except an input value of `Neutral` becomes `neutral`
        for both rows. Null values are preserved, and the input is not modified.

    Raises:
        TypeError: If dataframe or invert has an invalid type.
        ValueError: If home/away columns do not have matching counterparts.

    See Also:
        <https://nflreadr.nflverse.com/reference/clean_homeaway.html>

    Examples:
        >>> games = pl.DataFrame({"home_team": ["KC"], "away_team": ["BUF"]})
        >>> clean_homeaway(games).to_dict(as_series=False)
        {'team': ['KC', 'BUF'], 'opponent': ['BUF', 'KC'], 'location': ['home', 'away']}
    """
    if not isinstance(dataframe, pl.DataFrame):
        raise TypeError("argument `dataframe` must be a Polars DataFrame")
    if invert is not None and (
        not isinstance(invert, list)
        or not all(isinstance(column, str) for column in invert)
    ):
        raise TypeError("argument `invert` must be a list of column names or None")

    home = _perspective(dataframe, "home")
    away = _perspective(dataframe, "away")
    if set(home.columns) != set(away.columns):
        raise ValueError("dataframe must have matching home/away columns")
    if invert:
        away = away.with_columns(
            -pl.col(column) for column in invert if away.schema.get(column) != pl.Null
        )

    combined = pl.concat([home, away.select(home.columns)], how="vertical_relaxed")
    # Interleave home and away rows without adding a temporary column to the data.
    row = pl.int_range(0, 2 * dataframe.height)
    return combined.select(pl.all().gather(row // 2 + (row % 2) * dataframe.height))


def _perspective(dataframe: pl.DataFrame, side: str) -> pl.DataFrame:
    opponent = "away" if side == "home" else "home"
    names = {}
    for column in dataframe.columns:
        name = re.sub(f"^{side}_", "team_", column)
        name = re.sub(f"^{opponent}_", "opponent_", name)
        name = re.sub(f"_{side}$", "", name)
        name = re.sub(f"_{opponent}$", "_opponent", name)
        names[column] = name.replace("team_team", "team").replace(
            "opponent_team", "opponent"
        )

    frame = dataframe.rename(names)
    location = (
        pl.when(pl.col("location") == "Neutral")
        .then(pl.lit("neutral"))
        .otherwise(pl.lit(side))
        if "location" in frame.columns
        else pl.repeat(side, dataframe.height)
    )
    return frame.with_columns(location.alias("location"))
