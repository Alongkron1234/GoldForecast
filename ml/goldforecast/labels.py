"""Forecast targets: % return and direction, `horizon` trading days ahead."""

import pandas as pd


def make_targets(df: pd.DataFrame, horizon: int, price_col: str = "Close") -> pd.DataFrame:
    """Return a copy of `df` with two target columns added for this `horizon`:

    - `return_pct_t{horizon}`: % change from `price_col` today to `price_col` `horizon`
      trading days ahead.
    - `direction_t{horizon}`: "up" if that future return is positive, "down" otherwise.

    The last `horizon` rows have no known future price yet, so both target columns are
    NaN/NA there — callers drop those rows before training, they are never valid samples.
    """
    result = df.copy()
    future_price = df[price_col].shift(-horizon)  # อนาคต

    return_col = f"return_pct_t{horizon}" # ปัจจุบัน
    direction_col = f"direction_t{horizon}"

    result[return_col] = (future_price - df[price_col]) / df[price_col] * 100  # หา % return

    direction = pd.Series(pd.NA, index=df.index, dtype="object")
    direction[result[return_col] > 0] = "up"
    direction[result[return_col] <= 0] = "down"
    result[direction_col] = direction

    return result
