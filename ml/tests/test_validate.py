import pandas as pd
import pytest

from goldforecast.data.validate import (
    DataValidationError,
    missing_value_check,
    range_check,
    schema_check,
)


def _good_df():
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3),
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200],
        }
    )


def test_schema_check_passes_on_valid_df():
    schema_check(_good_df())


def test_schema_check_raises_on_missing_column():
    df = _good_df().drop(columns=["Volume"])
    with pytest.raises(DataValidationError):
        schema_check(df)


def test_missing_value_check_passes_without_nan():
    missing_value_check(_good_df())


def test_missing_value_check_raises_on_nan():
    df = _good_df()
    df.loc[1, "Close"] = None
    with pytest.raises(DataValidationError):
        missing_value_check(df)


def test_range_check_passes_within_bounds():
    range_check(_good_df(), "Close", min_value=0)


def test_range_check_raises_on_negative_price():
    df = _good_df()
    df.loc[0, "Low"] = -5.0
    with pytest.raises(DataValidationError):
        range_check(df, "Low", min_value=0)
