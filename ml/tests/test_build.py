import pandas as pd

from goldforecast.features.build import build_feature_matrix, stationary_feature_columns


def test_build_feature_matrix_adds_expected_columns():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=40),
            "Close": [100.0 + i for i in range(40)],
            "dxy_close": [90.0 + i * 0.1 for i in range(40)],
        }
    )

    result = build_feature_matrix(df)

    for expected in ["ma_5", "ma_20", "ema_12", "ema_26", "rsi_14", "macd", "macd_signal"]:
        assert expected in result.columns

    assert "Close_lag_1" in result.columns
    assert "dxy_close_lag_1" in result.columns


def test_build_feature_matrix_has_values_once_warmup_period_passes():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=40),
            "Close": [100.0 + i for i in range(40)],
        }
    )

    result = build_feature_matrix(df)

    # ma_20 needs 20 rows of warmup; row 39 (the 40th) should have a real value
    assert pd.notna(result["ma_20"].iloc[-1])


def test_build_feature_matrix_adds_stationary_features():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=40),
            "Close": [100.0 + i for i in range(40)],
            "dxy_close": [90.0 + i * 0.1 for i in range(40)],
        }
    )

    result = build_feature_matrix(df)

    for expected in [
        "ma_ratio_5",
        "ma_ratio_20",
        "ema_ratio_12",
        "ema_ratio_26",
        "macd_pct",
        "macd_signal_pct",
        "daily_return_pct",
        "dxy_return",
        "daily_return_pct_lag_1",
        "dxy_return_lag_1",
    ]:
        assert expected in result.columns, expected


def test_stationary_feature_columns_excludes_raw_price_levels():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=40),
            "Close": [100.0 + i for i in range(40)],
            "dxy_close": [90.0 + i * 0.1 for i in range(40)],
        }
    )

    built = build_feature_matrix(df)
    stationary = stationary_feature_columns(built)

    # must never include raw price-level columns — that's the whole point
    for leaky in ["Close", "dxy_close", "ma_5", "ma_20", "ema_12", "macd", "Close_lag_1", "Date"]:
        assert leaky not in stationary

    # must include the scale-free replacements
    for safe in ["ma_ratio_5", "rsi_14", "macd_pct", "daily_return_pct", "dxy_return"]:
        assert safe in stationary
