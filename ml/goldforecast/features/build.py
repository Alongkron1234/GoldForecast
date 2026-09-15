"""Assemble the full feature matrix: technical indicators + lag features,
using the parameters defined in config.py so every horizon/model sees the
same feature set built the same way."""

import pandas as pd

from goldforecast import config
from goldforecast.features.indicators import ema, macd, moving_average, rsi
from goldforecast.features.lags import lag_features


def build_feature_matrix(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    result = df.copy()

    for window in config.MA_WINDOWS:
        result[f"ma_{window}"] = moving_average(df[price_col], window)

    for span in config.EMA_SPANS:
        result[f"ema_{span}"] = ema(df[price_col], span)

    result[f"rsi_{config.RSI_WINDOW}"] = rsi(df[price_col], config.RSI_WINDOW)

    macd_line, signal_line = macd(df[price_col], **config.MACD_PARAMS)
    result["macd"] = macd_line
    result["macd_signal"] = signal_line

    lag_columns = [price_col] + [c for c in df.columns if c.endswith("_close")]
    result = lag_features(result, columns=lag_columns, lags=config.LAG_DAYS)

    return result
