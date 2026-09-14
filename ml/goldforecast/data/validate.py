"""Validate ingested market data. Every check raises loudly on bad data
instead of letting it flow silently into the pipeline."""

import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


class DataValidationError(Exception):
    """Raised when ingested market data fails a validation check."""


def schema_check(df: pd.DataFrame, required_columns: list[str] = REQUIRED_COLUMNS) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")


def missing_value_check(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    columns = columns or list(df.columns)
    null_counts = df[columns].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise DataValidationError(f"missing values found: {bad.to_dict()}")


def range_check(
    df: pd.DataFrame,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    if min_value is not None and (df[column] < min_value).any():
        raise DataValidationError(f"{column} has values below {min_value}")
    if max_value is not None and (df[column] > max_value).any():
        raise DataValidationError(f"{column} has values above {max_value}")
