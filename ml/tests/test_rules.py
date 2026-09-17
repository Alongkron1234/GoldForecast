import numpy as np
import pandas as pd

from goldforecast.signals.rules import find_best_threshold, generate_signal


def test_generate_signal_buy_when_up_confident_and_above_threshold():
    signal = generate_signal(
        predicted_return_pct=0.5, predicted_direction="up", confidence=0.7, x=0.1, y=0.6
    )
    assert signal.action == "BUY"
    assert signal.confidence == 0.7


def test_generate_signal_sell_when_down_confident_and_below_threshold():
    signal = generate_signal(
        predicted_return_pct=-0.5, predicted_direction="down", confidence=0.7, x=0.1, y=0.6
    )
    assert signal.action == "SELL"


def test_generate_signal_hold_when_models_disagree():
    # regression says positive, but classifier says "down" — no agreement, no trade
    signal = generate_signal(
        predicted_return_pct=0.3, predicted_direction="down", confidence=0.7, x=0.1, y=0.6
    )
    assert signal.action == "HOLD"


def test_generate_signal_hold_when_confidence_too_low():
    signal = generate_signal(
        predicted_return_pct=0.5, predicted_direction="up", confidence=0.52, x=0.1, y=0.6
    )
    assert signal.action == "HOLD"


def test_generate_signal_hold_when_magnitude_too_small():
    signal = generate_signal(
        predicted_return_pct=0.02, predicted_direction="up", confidence=0.8, x=0.1, y=0.6
    )
    assert signal.action == "HOLD"


def test_generate_signal_exactly_at_threshold_is_hold():
    # strictly-greater-than: sitting exactly on the threshold does not qualify
    signal = generate_signal(
        predicted_return_pct=0.1, predicted_direction="up", confidence=0.6, x=0.1, y=0.6
    )
    assert signal.action == "HOLD"


def test_find_best_threshold_picks_up_signal_on_obviously_predictive_data():
    n = 100
    predicted_return_pct = pd.Series([1.0] * n)
    predicted_direction = pd.Series(["up"] * n)
    confidence = pd.Series([0.9] * n)
    # actual return always positive (with a little noise so std != 0) and
    # sizeable when predicted "up" — a trivially profitable BUY-everything
    # scenario
    rng = np.random.default_rng(0)
    actual_return_pct = pd.Series(2.0 + rng.normal(0, 0.1, n))

    best = find_best_threshold(
        predicted_return_pct,
        predicted_direction,
        confidence,
        actual_return_pct,
        x_candidates=[0.0, 0.5],
        y_candidates=[0.5, 0.95],
    )

    # with confidence fixed at 0.9, only y=0.5 lets any signal through;
    # y=0.95 would produce zero trades (Sharpe 0), so the optimizer should
    # prefer the threshold that actually trades and captures the gain
    assert best["y"] == 0.5
    assert best["sharpe"] > 0
