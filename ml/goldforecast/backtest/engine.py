"""Simulate the Signal strategy over a period, with transaction costs, and
compare it against buy-and-hold. The cost-aware counterpart to
`signals.rules._strategy_returns`, which is a fast, cost-free proxy used only
for threshold search — this is the number that actually gets reported."""

from dataclasses import dataclass

import pandas as pd

from goldforecast.signals.rules import generate_signal, sharpe_ratio


@dataclass(frozen=True)
class BacktestResult:
    cumulative_return_pct: float
    sharpe_ratio: float
    win_rate: float
    n_trades: int
    buy_and_hold_return_pct: float

# รับผลตอบแทนรา่ยวันแล้วแปลง แล้วคำนวนว่าทั้งหมดได้กำไร/ขาดทุนกี่ % จากต้นจนจบ
def _cumulative_return_pct(daily_returns_pct: pd.Series) -> float:
    """Compound daily % returns into one total % return over the period
    (not a simple sum — a +10% day followed by a -10% day is a net loss)."""
    if daily_returns_pct.empty:
        return 0.0
    return ((1 + daily_returns_pct / 100).prod() - 1) * 100


def run_backtest(
    predicted_return_pct: pd.Series,
    predicted_direction: pd.Series,
    confidence: pd.Series,
    actual_return_pct: pd.Series,
    x: float,
    y: float,
    transaction_cost_pct: float = 0.05,
) -> BacktestResult:
    """Simulate trading the Signal strategy (fixed (x, y), fixed position size)
    day by day. `actual_return_pct` is the realized % return for each day —
    normally `return_pct_t{horizon}` from the test split, so the model never
    saw these outcomes during training or threshold tuning."""

    # ได้เป็น list ของการกระทำ (action) สำหรับแต่ละวัน: "BUY", "SELL", หรือ "HOLD"
    # โดยใช้ฟังก์ชัน generate_signal
    actions = [
        generate_signal(r, d, c, x, y).action
        for r, d, c in zip(predicted_return_pct, predicted_direction, confidence, strict=True)
    ]
    actions = pd.Series(actions, index=actual_return_pct.index)

    # แปลงการกระทำเป็นผลตอบแทนรายวัน (daily return) โดยใช้ผลตอบแทนที่เกิดขึ้นจริง
    # (actual return) และทิศทางของการซื้อขาย (BUY/SELL/HOLD)
    daily_return = pd.Series(0.0, index=actual_return_pct.index)
    daily_return[actions == "BUY"] = actual_return_pct[actions == "BUY"]
    daily_return[actions == "SELL"] = -actual_return_pct[actions == "SELL"]

    # หักค่าธรรมเนียมการซื้อขาย (transaction cost
    is_trade = actions != "HOLD"
    daily_return[is_trade] -= transaction_cost_pct

    n_trades = int(is_trade.sum())
    win_rate = float((daily_return[is_trade] > 0).mean()) if n_trades > 0 else 0.0

    return BacktestResult(
        cumulative_return_pct=_cumulative_return_pct(daily_return),
        sharpe_ratio=sharpe_ratio(daily_return),
        win_rate=win_rate,
        n_trades=n_trades,
        buy_and_hold_return_pct=_cumulative_return_pct(actual_return_pct),
    )


def run_backtest_weighted(
    predicted_direction: pd.Series,
    confidence: pd.Series,
    actual_return_pct: pd.Series,
    transaction_cost_pct: float = 0.05,
) -> BacktestResult:
    """Secondary experiment (not the default — see docs/PROJECT.md): instead
    of a hard BUY/SELL/HOLD threshold that sits fully out of the market most
    days, size the position continuously by how confident the classifier is,
    every single day. A day just under the old confidence cutoff still gets
    partial exposure instead of zero — meant to test whether `run_backtest`'s
    finding (good win rate, but badly underperforms buy-and-hold because it
    barely participates) improves once the strategy stops sitting out so much
    of a strong trend.

    `confidence` is expected in [0.5, 1.0] (0.5 = coin flip, 1.0 = fully
    confident) and is rescaled here to a [0, 1] position weight."""
    weight = (confidence - 0.5) * 2
    direction_sign = predicted_direction.map({"up": 1, "down": -1})
    position = weight * direction_sign

    daily_return = position * actual_return_pct - transaction_cost_pct * position.abs()

    is_trade = position.abs() > 0
    n_trades = int(is_trade.sum())
    win_rate = float((daily_return[is_trade] > 0).mean()) if n_trades > 0 else 0.0

    return BacktestResult(
        cumulative_return_pct=_cumulative_return_pct(daily_return),
        sharpe_ratio=sharpe_ratio(daily_return),
        win_rate=win_rate,
        n_trades=n_trades,
        buy_and_hold_return_pct=_cumulative_return_pct(actual_return_pct),
    )
