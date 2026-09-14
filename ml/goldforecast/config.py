"""Project-wide constants: data sources and forecast horizons."""

TICKERS = {
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "crude_oil": "CL=F",
    "vix": "^VIX",
    "treasury_10y": "^TNX",
    "sp500": "^GSPC",
}

HORIZONS = [1, 5, 10]

# Technical indicator parameters — standard values used across the industry,
# not project-specific tuning (see CONTEXT.md / docs/PROJECT.md for rationale).
MA_WINDOWS = [5, 20]
EMA_SPANS = [12, 26]
RSI_WINDOW = 14
MACD_PARAMS = {"fast": 12, "slow": 26, "signal": 9}

LAG_DAYS = [1, 2, 3, 5, 10]
