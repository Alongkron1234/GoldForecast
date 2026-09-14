import pandas as pd
import pytest

from goldforecast.labels import make_targets


def test_make_targets_horizon_1_computes_return_and_direction():
    df = pd.DataFrame({"Close": [100.0, 110.0, 99.0, 99.0]})

    result = make_targets(df, horizon=1)

    # row 0: 100 -> 110, +10%
    assert result["return_pct_t1"].iloc[0] == pytest.approx(10.0)
    assert result["direction_t1"].iloc[0] == "up"

    # row 1: 110 -> 99, -10%
    assert result["return_pct_t1"].iloc[1] == pytest.approx(-10.0)
    assert result["direction_t1"].iloc[1] == "down"

    # row 2: 99 -> 99, 0% (treated as "down", not "up")
    assert result["return_pct_t1"].iloc[2] == pytest.approx(0.0)
    assert result["direction_t1"].iloc[2] == "down"


def test_make_targets_last_horizon_rows_are_unknown():
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})

    result = make_targets(df, horizon=2)

    # only row 0 has a known price 2 days ahead (row 2); rows 1-2 do not
    assert result["return_pct_t2"].isna().sum() == 2
    assert pd.isna(result["direction_t2"].iloc[-1])
    assert pd.isna(result["direction_t2"].iloc[-2])


def test_make_targets_uses_horizon_specific_column_names():
    df = pd.DataFrame({"Close": [100.0, 105.0, 110.0]})

    result_h1 = make_targets(df, horizon=1)
    result_h1_h2 = make_targets(result_h1, horizon=2)

    assert "return_pct_t1" in result_h1_h2.columns
    assert "return_pct_t2" in result_h1_h2.columns
    assert "direction_t1" in result_h1_h2.columns
    assert "direction_t2" in result_h1_h2.columns
