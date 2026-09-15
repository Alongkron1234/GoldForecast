from unittest.mock import patch

import pandas as pd
import pytest

from goldforecast.data.fetch import fetch_source


def _fake_yf_frame():
    index = pd.date_range("2024-01-01", periods=3, freq="D", name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Adj Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200],
        },
        index=index,
    )


@patch("goldforecast.data.fetch.yf.download")
def test_fetch_source_returns_normalized_columns(mock_download):
    mock_download.return_value = _fake_yf_frame()

    df = fetch_source("GC=F", start="2024-01-01", end="2024-01-04")

    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 3
    assert df["Date"].is_monotonic_increasing


@patch("goldforecast.data.fetch.yf.download")
def test_fetch_source_raises_on_empty_result(mock_download):
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError):
        fetch_source("GC=F", start="2024-01-01", end="2024-01-04")
