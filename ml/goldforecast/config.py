"""Project-wide constants: data sources and forecast horizons."""

import os

# Falls back to a local SQLite file (no Docker needed) when the MLflow server
# from docker-compose isn't running — same code path works in both cases.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

TICKERS = {
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "crude_oil": "CL=F",
    "vix": "^VIX",
    "treasury_10y": "^TNX",
    "sp500": "^GSPC",
}

HORIZONS = [1, 5, 10]

# Fixed (not "today - 10 years") so re-running the data pipeline on a different day
# doesn't shift the train/val/test date boundaries — backtest results need to stay
# comparable across runs.
DATA_START = "2016-09-16"
DATA_END = "2026-09-14"

# Technical indicator parameters — standard values used across the industry,
# not project-specific tuning (see CONTEXT.md / docs/PROJECT.md for rationale).
MA_WINDOWS = [5, 20]
EMA_SPANS = [12, 26]
RSI_WINDOW = 14
MACD_PARAMS = {"fast": 12, "slow": 26, "signal": 9}

LAG_DAYS = [1, 2, 3, 5, 10]
