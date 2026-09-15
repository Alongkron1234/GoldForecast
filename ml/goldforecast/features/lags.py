"""Lag features: past values of a column, shifted forward as new columns."""

import pandas as pd

DEFAULT_LAGS = [1, 2, 3, 5, 10]


def lag_features(
    df: pd.DataFrame, columns: list[str], lags: list[int] = DEFAULT_LAGS
) -> pd.DataFrame:
    """Return a copy of `df` with `{column}_lag_{n}` added for each column/lag pair.

    Row i's `{column}_lag_{n}` holds the value of `column` from n rows before row i
    (i.e. n trading days earlier, assuming df is already sorted by date). The first
    n rows of each lag column are NaN since there is no earlier value to reference.
    """
    result = df.copy()
    for column in columns:
        for lag in lags:
            result[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return result
