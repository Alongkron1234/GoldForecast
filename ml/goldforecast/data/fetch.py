"""Fetch raw OHLCV data for a single ticker from Yahoo Finance."""

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def fetch_source(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data for `ticker` between `start` and `end` (YYYY-MM-DD).

    Returns a DataFrame with columns Date, Open, High, Low, Close, Volume,
    sorted by Date ascending.
    """
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"no data returned for ticker={ticker!r} between {start} and {end}")

    df = raw.reset_index()
    # yfinance returns MultiIndex columns (ticker, field) for some versions/inputs;
    # flatten to plain field names since we only ever fetch one ticker at a time.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df.sort_values("Date").reset_index(drop=True)
