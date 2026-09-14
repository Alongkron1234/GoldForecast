import pandas as pd

from goldforecast.features.lags import lag_features


def test_lag_features_shifts_values_by_correct_offset():
    df = pd.DataFrame({"Close": [10.0, 20.0, 30.0, 40.0, 50.0]})

    result = lag_features(df, columns=["Close"], lags=[1, 2])

    assert result["Close_lag_1"].isna().sum() == 1
    assert list(result["Close_lag_1"].iloc[1:]) == [10.0, 20.0, 30.0, 40.0]

    assert result["Close_lag_2"].isna().sum() == 2
    assert list(result["Close_lag_2"].iloc[2:]) == [10.0, 20.0, 30.0]


def test_lag_features_preserves_original_columns():
    df = pd.DataFrame({"Close": [10.0, 20.0], "Volume": [100, 200]})

    result = lag_features(df, columns=["Close"], lags=[1])

    assert "Close" in result.columns
    assert "Volume" in result.columns
    assert "Close_lag_1" in result.columns


def test_lag_features_handles_multiple_columns():
    df = pd.DataFrame({"Close": [10.0, 20.0, 30.0], "dxy_close": [90.0, 91.0, 92.0]})

    result = lag_features(df, columns=["Close", "dxy_close"], lags=[1])

    assert "Close_lag_1" in result.columns
    assert "dxy_close_lag_1" in result.columns
