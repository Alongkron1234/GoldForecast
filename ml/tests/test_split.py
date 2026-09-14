import pandas as pd
import pytest

from goldforecast.split import chronological_split


def _df(n_days=100):
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=n_days, freq="D"),
            "Close": range(n_days),
        }
    )


def test_chronological_split_respects_proportions():
    train, val, test = chronological_split(_df(100), train=0.6, val=0.2, test=0.2)

    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20


def test_chronological_split_has_no_leakage():
    train, val, test = chronological_split(_df(100))

    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max() < test["Date"].min()


def test_chronological_split_sorts_unsorted_input():
    df = _df(10).sample(frac=1.0, random_state=42)  # shuffle rows

    train, val, test = chronological_split(df, train=0.6, val=0.2, test=0.2)

    assert train["Date"].is_monotonic_increasing
    assert val["Date"].is_monotonic_increasing
    assert test["Date"].is_monotonic_increasing
    assert train["Date"].max() < val["Date"].min()


def test_chronological_split_raises_when_proportions_dont_sum_to_one():
    with pytest.raises(ValueError):
        chronological_split(_df(10), train=0.5, val=0.3, test=0.3)
