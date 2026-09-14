"""Tests for converting game rows to team rows."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import nflreadpy as nfl


def test_documented_example():
    games = pl.DataFrame(
        {
            "game_id": ["2020_20_TB_GB", "2020_20_BUF_KC", "2020_21_KC_TB"],
            "game_type": ["CON", "CON", "SB"],
            "away_team": ["TB", "BUF", "KC"],
            "away_score": [31, 24, 9],
            "home_team": ["GB", "KC", "TB"],
            "home_score": [26, 38, 31],
            "location": ["Home", "Home", "Neutral"],
            "result": [-5, 14, 22],
            "spread_line": [3, 3, -3],
        }
    )
    original = games.clone()
    expected = pl.DataFrame(
        {
            "game_id": [
                "2020_20_TB_GB",
                "2020_20_TB_GB",
                "2020_20_BUF_KC",
                "2020_20_BUF_KC",
                "2020_21_KC_TB",
                "2020_21_KC_TB",
            ],
            "game_type": ["CON", "CON", "CON", "CON", "SB", "SB"],
            "opponent": ["TB", "GB", "BUF", "KC", "KC", "TB"],
            "opponent_score": [31, 26, 24, 38, 9, 31],
            "team": ["GB", "TB", "KC", "BUF", "TB", "KC"],
            "team_score": [26, 31, 38, 24, 31, 9],
            "location": ["home", "away", "home", "away", "neutral", "neutral"],
            "result": [-5, 5, 14, -14, 22, -22],
            "spread_line": [3, -3, 3, -3, -3, 3],
        }
    )

    assert_frame_equal(nfl.clean_homeaway(games, ["result", "spread_line"]), expected)
    assert_frame_equal(games, original)


def test_suffix_columns_and_input_order():
    games = pl.DataFrame(
        {
            "game_id": ["second", "first", "first"],
            "team_home": ["GB", "KC", "TB"],
            "team_away": ["TB", "BUF", "KC"],
            "score_home": [26, 38, 31],
            "score_away": [31, 24, 9],
            "homecoming": [1, 2, 3],
            "__row_order": [9, 8, 7],
        }
    )
    expected = pl.DataFrame(
        {
            "game_id": ["second", "second", "first", "first", "first", "first"],
            "team": ["GB", "TB", "KC", "BUF", "TB", "KC"],
            "team_opponent": ["TB", "GB", "BUF", "KC", "KC", "TB"],
            "score": [26, 31, 38, 24, 31, 9],
            "score_opponent": [31, 26, 24, 38, 9, 31],
            "homecoming": [1, 1, 2, 2, 3, 3],
            "__row_order": [9, 9, 8, 8, 7, 7],
            "location": ["home", "away", "home", "away", "home", "away"],
        }
    )

    assert_frame_equal(nfl.clean_homeaway(games), expected)


@pytest.mark.parametrize("invert", [None, [], ["result"]])
def test_nullable_values_and_location(invert):
    games = pl.DataFrame(
        {
            "home_team": ["KC", "TB"],
            "away_team": ["BUF", "GB"],
            "home_score": [None, 31],
            "away_score": [24, None],
            "location": [None, "Neutral"],
            "result": [None, 5.5],
        }
    )
    result = nfl.clean_homeaway(games, invert)

    assert result["team_score"].to_list() == [None, 24, 31, None]
    assert result["opponent_score"].to_list() == [24, None, None, 31]
    assert result["location"].to_list() == ["home", "away", "neutral", "neutral"]
    assert result["result"].to_list() == [None, None, 5.5, -5.5 if invert else 5.5]


def test_empty_input_keeps_schema():
    games = pl.DataFrame(schema={"home_team": pl.Utf8, "away_team": pl.Utf8})
    expected = pl.DataFrame(
        schema={"team": pl.Utf8, "opponent": pl.Utf8, "location": pl.Utf8}
    )

    assert_frame_equal(nfl.clean_homeaway(games), expected)


def test_empty_input_without_columns():
    expected = pl.DataFrame(schema={"location": pl.Utf8})

    assert_frame_equal(nfl.clean_homeaway(pl.DataFrame()), expected)


def test_all_null_result_can_be_inverted():
    games = pl.DataFrame({"home_team": ["KC"], "away_team": ["BUF"], "result": [None]})
    expected = pl.DataFrame(
        {
            "team": ["KC", "BUF"],
            "opponent": ["BUF", "KC"],
            "result": [None, None],
            "location": ["home", "away"],
        }
    )

    assert_frame_equal(nfl.clean_homeaway(games, ["result"]), expected)


def test_numeric_types_are_reconciled():
    games = pl.DataFrame(
        {"home_score": pl.Series([1], dtype=pl.Int32), "away_score": [2.5]}
    )
    expected = pl.DataFrame(
        {
            "team_score": [1.0, 2.5],
            "opponent_score": [2.5, 1.0],
            "location": ["home", "away"],
        }
    )

    assert_frame_equal(nfl.clean_homeaway(games), expected)


def test_invert_uses_output_column_names():
    games = pl.DataFrame({"home_score": [10], "away_score": [20]})
    result = nfl.clean_homeaway(games, ["team_score"])

    assert result["team_score"].to_list() == [10, -20]
    assert result["opponent_score"].to_list() == [20, 10]


@pytest.mark.parametrize("dataframe", [None, [], {}, "games"])
def test_invalid_dataframe(dataframe):
    with pytest.raises(TypeError, match="dataframe.*Polars DataFrame"):
        nfl.clean_homeaway(dataframe)


@pytest.mark.parametrize("invert", ["result", 1, [1], ["result", None]])
def test_invalid_invert(invert):
    with pytest.raises(TypeError, match="invert.*list of column names"):
        nfl.clean_homeaway(pl.DataFrame({"result": [1]}), invert)


def test_unpaired_columns_are_rejected():
    with pytest.raises(ValueError, match="matching home/away columns"):
        nfl.clean_homeaway(pl.DataFrame({"home_score": [10]}))


def test_missing_invert_column_is_rejected():
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        nfl.clean_homeaway(pl.DataFrame({"result": [1]}), ["missing"])
