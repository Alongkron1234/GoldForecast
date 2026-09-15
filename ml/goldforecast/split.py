"""Chronological train/val/test split — no shuffling, since shuffling a time
series before splitting would leak future information into training."""

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train: float = 0.6,
    val: float = 0.2,
    test: float = 0.2,
    date_col: str = "Date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError(f"train + val + test must sum to 1.0, got {train + val + test}")

    ordered = df.sort_values(date_col).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train)
    val_end = train_end + int(n * val)

    train_df = ordered.iloc[:train_end]
    val_df = ordered.iloc[train_end:val_end]
    test_df = ordered.iloc[val_end:]
    return train_df, val_df, test_df
