"""Turn model predictions into BUY/SELL/HOLD, and find good thresholds for
doing so. See CONTEXT.md for the Signal / Confidence Score definitions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Signal:
    action: str  # "BUY" | "SELL" | "HOLD"
    confidence: float


def generate_signal(
    predicted_return_pct: float,
    predicted_direction: str,
    confidence: float,
    x: float,
    y: float,
) -> Signal:
    """BUY if direction="up" and return>x and confidence>y; SELL mirrored; else HOLD.

    Requiring both models to agree AND be confident is deliberate: when they
    disagree or aren't confident, HOLD is the safe default rather than a
    coin-flip trade — see docs/PROJECT.md for the original design discussion.
    """
    if predicted_direction == "up" and predicted_return_pct > x and confidence > y:
        return Signal(action="BUY", confidence=confidence)
    if predicted_direction == "down" and predicted_return_pct < -x and confidence > y:
        return Signal(action="SELL", confidence=confidence)
    return Signal(action="HOLD", confidence=confidence)


def _strategy_returns(
    predicted_return_pct: pd.Series,
    predicted_direction: pd.Series,
    confidence: pd.Series,
    actual_return_pct: pd.Series,
    x: float,
    y: float,
) -> pd.Series:
    """Daily return the strategy would have earned: +actual on BUY days,
    -actual on SELL days (short), 0 on HOLD days. No transaction costs — this
    is a fast proxy for threshold search; the full, cost-aware simulation is
    `backtest/engine.py`."""
    actions = [
        generate_signal(r, d, c, x, y).action
        for r, d, c in zip(predicted_return_pct, predicted_direction, confidence, strict=True)
    ]
    actions = pd.Series(actions, index=actual_return_pct.index)
    strategy_return = pd.Series(0.0, index=actual_return_pct.index)
    strategy_return[actions == "BUY"] = actual_return_pct[actions == "BUY"]
    strategy_return[actions == "SELL"] = -actual_return_pct[actions == "SELL"]
    return strategy_return


def sharpe_ratio(daily_returns_pct: pd.Series) -> float:
    """Annualized Sharpe ratio of a daily % return series. Shared with
    `backtest/engine.py` so threshold search and the final backtest score
    strategies the same way."""
    if daily_returns_pct.empty or daily_returns_pct.std() == 0:
        return 0.0
    return (daily_returns_pct.mean() / daily_returns_pct.std()) * np.sqrt(252)


def find_best_threshold(
    predicted_return_pct: pd.Series,
    predicted_direction: pd.Series,
    confidence: pd.Series,
    actual_return_pct: pd.Series,
    x_candidates: list[float],
    y_candidates: list[float],
) -> dict:
    """Sweep (x, y) candidates, score each by the Sharpe ratio of the simple
    (cost-free) strategy return, and return the best combination.

    Call this on the VALIDATION set only — never the test set — so the final
    backtest stays an honest out-of-sample evaluation."""
    best = {"x": None, "y": None, "sharpe": -np.inf}
    for x in x_candidates:
        for y in y_candidates:
            returns = _strategy_returns(
                predicted_return_pct, predicted_direction, confidence, actual_return_pct, x, y
            )
            sharpe = sharpe_ratio(returns)
            if sharpe > best["sharpe"]:
                best = {"x": x, "y": y, "sharpe": sharpe}
    return best
