"""Assemble the full feature matrix: technical indicators + lag features,
using the parameters defined in config.py so every horizon/model sees the
same feature set built the same way."""

import pandas as pd

from goldforecast import config
from goldforecast.features.indicators import ema, macd, moving_average, rsi
from goldforecast.features.lags import lag_features


def build_feature_matrix(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Build both the raw-level indicators (kept for charting/EDA — see
    notebooks/eda.ipynb) and a second set of *stationary* (scale-free)
    features actually meant for model training. See `STATIONARY_FEATURE_COLUMNS`.

    Raw price-level features (Close, ma_5, ma_20, ...) are unsafe as direct
    model inputs on a trending series: in a decade-long bull market the price
    LEVEL alone nearly identifies the date, so a tree model can "cheat" by
    memorizing price level -> time period instead of learning a transferable
    pattern (confirmed empirically: a classifier trained on these hit 100%
    train accuracy but underperformed a majority-class baseline on the test
    set). The stationary features below (% returns, ratios to a moving
    average, bounded oscillators) don't carry that leakage.
    """
    result = df.copy()

    for window in config.MA_WINDOWS:
        ma = moving_average(df[price_col], window)
        result[f"ma_{window}"] = ma
        result[f"ma_ratio_{window}"] = df[price_col] / ma - 1

    for span in config.EMA_SPANS:
        e = ema(df[price_col], span)
        result[f"ema_{span}"] = e
        result[f"ema_ratio_{span}"] = df[price_col] / e - 1

    result[f"rsi_{config.RSI_WINDOW}"] = rsi(df[price_col], config.RSI_WINDOW)

    macd_line, signal_line = macd(df[price_col], **config.MACD_PARAMS)
    result["macd"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_pct"] = macd_line / df[price_col] * 100
    result["macd_signal_pct"] = signal_line / df[price_col] * 100

    lag_columns = [price_col] + [c for c in df.columns if c.endswith("_close")]
    result = lag_features(result, columns=lag_columns, lags=config.LAG_DAYS)

    # Stationary counterpart of every price-level series: its own % change,
    # plus lagged versions — comparable across time periods, unlike a raw
    # price/lag value.
    return_columns = []
    for col in lag_columns:
        if col == price_col:
            return_col = "daily_return_pct"
        else:
            return_col = f"{col.removesuffix('_close')}_return"
        result[return_col] = df[col].pct_change() * 100
        return_columns.append(return_col)

    result = lag_features(result, columns=return_columns, lags=config.LAG_DAYS)

    return result


def stationary_feature_columns(built_df: pd.DataFrame) -> list[str]:
    """The subset of `build_feature_matrix()`'s output that's safe to train
    models on directly — bounded oscillators and %/ratio features only, no
    raw price levels. Pass it the DataFrame `build_feature_matrix()` returned.
    See the module docstring above for why this matters."""
    prefixes = (
        "ma_ratio_",
        "ema_ratio_",
        "rsi_",
        "macd_pct",
        "macd_signal_pct",
        "daily_return_pct",
    )
    return [c for c in built_df.columns if c.startswith(prefixes) or "_return" in c]
