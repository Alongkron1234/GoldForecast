"""Merge per-source OHLCV DataFrames into a single wide table keyed on Date."""

import pandas as pd


def merge_sources(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join multiple sources (name -> DataFrame with Date/Open/High/Low/Close/Volume)
    into one wide DataFrame on Date, prefixing each source's columns with its name
    (except the shared "gold" primary series, which keeps its plain column names).

    Missing values from non-trading days in one market but not another (e.g. gold
    trades when a specific exchange is closed elsewhere) are forward-filled, since a
    macro indicator's last known value is a better estimate than a gap.
    """
    if "gold" not in sources:
        raise ValueError("merge_sources requires a 'gold' entry as the primary series")

    merged = sources["gold"].copy()

    for name, df in sources.items():
        if name == "gold":
            continue
        renamed = df[["Date", "Close"]].rename(columns={"Close": f"{name}_close"})
        merged = merged.merge(renamed, on="Date", how="left")

    merged = merged.sort_values("Date").reset_index(drop=True)
    merged = merged.ffill()
    return merged
