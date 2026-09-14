import pandas as pd

from goldforecast.data.merge import merge_sources


def _ohlcv(dates, closes):
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [100] * len(dates),
        }
    )


def test_merge_sources_joins_on_date_with_prefixed_columns():
    gold = _ohlcv(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 101.0, 102.0])
    dxy = _ohlcv(["2024-01-01", "2024-01-02", "2024-01-03"], [90.0, 90.5, 91.0])

    merged = merge_sources({"gold": gold, "dxy": dxy})

    assert "dxy_close" in merged.columns
    assert list(merged["dxy_close"]) == [90.0, 90.5, 91.0]
    assert len(merged) == 3


def test_merge_sources_forward_fills_missing_source_days():
    gold = _ohlcv(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 101.0, 102.0])
    # dxy is missing 2024-01-02 (e.g. a holiday in that market)
    dxy = _ohlcv(["2024-01-01", "2024-01-03"], [90.0, 91.0])

    merged = merge_sources({"gold": gold, "dxy": dxy})

    assert list(merged["dxy_close"]) == [90.0, 90.0, 91.0]


def test_merge_sources_requires_gold_key():
    dxy = _ohlcv(["2024-01-01"], [90.0])
    try:
        merge_sources({"dxy": dxy})
        assert False, "expected ValueError"
    except ValueError:
        pass
