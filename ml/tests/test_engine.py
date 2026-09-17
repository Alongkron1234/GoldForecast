import pandas as pd
import pytest

from goldforecast.backtest.engine import run_backtest, run_backtest_weighted


def _always_buy_inputs(n):
    return (
        pd.Series([1.0] * n),  # predicted_return_pct (> x=0.0, so BUY qualifies)
        pd.Series(["up"] * n),  # predicted_direction
        pd.Series([0.9] * n),  # confidence (> y=0.0)
    )


def test_run_backtest_compounds_returns_correctly_with_no_cost():
    predicted_return, predicted_direction, confidence = _always_buy_inputs(2)
    actual_return_pct = pd.Series([10.0, -10.0])

    result = run_backtest(
        predicted_return,
        predicted_direction,
        confidence,
        actual_return_pct,
        x=0.0,
        y=0.0,
        transaction_cost_pct=0.0,
    )

    # (1.10 * 0.90 - 1) * 100 = -1.0%, NOT 10 + (-10) = 0% (compounding, not summing)
    assert result.cumulative_return_pct == pytest.approx(-1.0)
    assert result.n_trades == 2


def test_run_backtest_transaction_cost_reduces_returns():
    predicted_return, predicted_direction, confidence = _always_buy_inputs(2)
    actual_return_pct = pd.Series([10.0, -10.0])

    no_cost = run_backtest(
        predicted_return, predicted_direction, confidence, actual_return_pct,
        x=0.0, y=0.0, transaction_cost_pct=0.0,
    )
    with_cost = run_backtest(
        predicted_return, predicted_direction, confidence, actual_return_pct,
        x=0.0, y=0.0, transaction_cost_pct=1.0,
    )

    assert with_cost.cumulative_return_pct < no_cost.cumulative_return_pct


def test_run_backtest_win_rate_only_counts_trade_days():
    predicted_return, predicted_direction, confidence = _always_buy_inputs(2)
    actual_return_pct = pd.Series([10.0, -10.0])

    result = run_backtest(
        predicted_return, predicted_direction, confidence, actual_return_pct,
        x=0.0, y=0.0, transaction_cost_pct=1.0,
    )

    # day 1: 10 - 1 = +9 (win), day 2: -10 - 1 = -11 (loss) -> 1 of 2 trades won
    assert result.win_rate == pytest.approx(0.5)


def test_run_backtest_hold_days_have_no_cost_and_no_return():
    n = 3
    # confidence=0.0 never exceeds y=0.5, so every day is HOLD
    predicted_return = pd.Series([5.0] * n)
    predicted_direction = pd.Series(["up"] * n)
    confidence = pd.Series([0.0] * n)
    actual_return_pct = pd.Series([10.0, -5.0, 3.0])

    result = run_backtest(
        predicted_return, predicted_direction, confidence, actual_return_pct,
        x=0.0, y=0.5, transaction_cost_pct=1.0,
    )

    assert result.n_trades == 0
    assert result.cumulative_return_pct == pytest.approx(0.0)
    assert result.win_rate == 0.0


def test_run_backtest_buy_and_hold_ignores_the_signal():
    n = 3
    predicted_return = pd.Series([5.0] * n)
    predicted_direction = pd.Series(["up"] * n)
    confidence = pd.Series([0.0] * n)  # forces all HOLD
    actual_return_pct = pd.Series([10.0, -5.0, 3.0])

    result = run_backtest(
        predicted_return, predicted_direction, confidence, actual_return_pct,
        x=0.0, y=0.5, transaction_cost_pct=1.0,
    )

    expected_buy_and_hold = ((1.10 * 0.95 * 1.03) - 1) * 100
    assert result.buy_and_hold_return_pct == pytest.approx(expected_buy_and_hold)


def test_run_backtest_weighted_scales_position_by_confidence():
    predicted_direction = pd.Series(["up", "up", "up"])
    confidence = pd.Series([0.5, 0.75, 1.0])  # weights: 0.0, 0.5, 1.0
    actual_return_pct = pd.Series([10.0, 10.0, 10.0])

    result = run_backtest_weighted(
        predicted_direction, confidence, actual_return_pct, transaction_cost_pct=0.0
    )

    # day1: weight 0 -> 0% return; day2: weight 0.5 -> 5%; day3: weight 1.0 -> 10%
    expected = ((1.00 * 1.05 * 1.10) - 1) * 100
    assert result.cumulative_return_pct == pytest.approx(expected)


def test_run_backtest_weighted_shorts_on_down_direction():
    predicted_direction = pd.Series(["down"])
    confidence = pd.Series([1.0])  # full weight, short
    actual_return_pct = pd.Series([-10.0])  # price fell -> short profits

    result = run_backtest_weighted(
        predicted_direction, confidence, actual_return_pct, transaction_cost_pct=0.0
    )

    assert result.cumulative_return_pct == pytest.approx(10.0)


def test_run_backtest_weighted_zero_confidence_means_zero_exposure():
    predicted_direction = pd.Series(["up", "down"])
    confidence = pd.Series([0.5, 0.5])  # coin flip -> no position either way
    actual_return_pct = pd.Series([10.0, -10.0])

    result = run_backtest_weighted(
        predicted_direction, confidence, actual_return_pct, transaction_cost_pct=1.0
    )

    assert result.n_trades == 0
    assert result.cumulative_return_pct == pytest.approx(0.0)
