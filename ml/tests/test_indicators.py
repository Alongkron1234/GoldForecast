import pandas as pd
import pytest

from goldforecast.features.indicators import ema, macd, moving_average, rsi


def test_moving_average_matches_hand_computed_values():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = moving_average(series, window=3)

    assert result.isna().sum() == 2
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_matches_hand_computed_values():
    # span=2 -> alpha = 2/(2+1) = 2/3
    series = pd.Series([1.0, 2.0, 3.0])

    result = ema(series, span=2)

    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx(5 / 3)
    assert result.iloc[2] == pytest.approx(23 / 9)


def test_rsi_is_100_for_strictly_increasing_series():
    series = pd.Series([float(i) for i in range(1, 10)])

    result = rsi(series, window=3)

    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_for_strictly_decreasing_series():
    series = pd.Series([float(i) for i in range(10, 1, -1)])

    result = rsi(series, window=3)

    assert result.iloc[-1] == pytest.approx(0.0)


def test_macd_line_equals_ema_fast_minus_ema_slow():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    macd_line, signal_line = macd(series, fast=2, slow=3, signal=2)

    expected_macd_line = ema(series, span=2) - ema(series, span=3)
    pd.testing.assert_series_equal(macd_line, expected_macd_line)

    expected_signal_line = ema(macd_line, span=2)
    pd.testing.assert_series_equal(signal_line, expected_signal_line)
