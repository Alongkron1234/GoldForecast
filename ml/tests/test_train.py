import numpy as np
import pandas as pd

from goldforecast import config
from goldforecast.features.build import build_feature_matrix
from goldforecast.labels import make_targets
from goldforecast.models.train import train_horizon


def _synthetic_df(n=150):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    dxy_close = 90 + np.cumsum(rng.normal(0, 0.2, n))
    raw = pd.DataFrame({"Date": dates, "Close": close, "dxy_close": dxy_close})

    featured = build_feature_matrix(raw)
    return make_targets(featured, horizon=1)


def test_train_horizon_returns_models_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    df = _synthetic_df()

    result = train_horizon(df, horizon=1, n_estimators=10)

    assert result["horizon"] == 1
    expected_metric_keys = {
        "regression_val_mae",
        "regression_test_mae",
        "classification_val_accuracy",
        "classification_test_accuracy",
        "baseline_val_accuracy",
        "baseline_test_accuracy",
    }
    assert set(result["metrics"]) == expected_metric_keys
    assert 0.0 <= result["metrics"]["classification_val_accuracy"] <= 1.0
    assert 0.0 <= result["metrics"]["baseline_test_accuracy"] <= 1.0


def test_train_horizon_uses_only_stationary_features(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    df = _synthetic_df()

    result = train_horizon(df, horizon=1, n_estimators=10)

    # raw price-level columns must never be used as model inputs (see
    # features/build.py docstring for why — this is the overfitting bug
    # that motivated the stationary-feature-only design)
    assert "Close" not in result["feature_columns"]
    assert "dxy_close" not in result["feature_columns"]
    assert "ma_5" not in result["feature_columns"]
    assert "Date" not in result["feature_columns"]
    assert "return_pct_t1" not in result["feature_columns"]
    assert "direction_t1" not in result["feature_columns"]

    # stationary features should be present
    assert "rsi_14" in result["feature_columns"]
    assert "daily_return_pct" in result["feature_columns"]
    assert "dxy_return" in result["feature_columns"]
