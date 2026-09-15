import numpy as np
import pandas as pd
import pytest

from goldforecast import config
from goldforecast.features.build import build_feature_matrix
from goldforecast.labels import make_targets
from goldforecast.models.registry import load_promoted, promote, registered_model_name
from goldforecast.models.train import train_horizon


def _synthetic_df(n=150):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    dxy_close = 90 + np.cumsum(rng.normal(0, 0.2, n))
    raw = pd.DataFrame({"Date": dates, "Close": close, "dxy_close": dxy_close})

    featured = build_feature_matrix(raw)
    return make_targets(featured, horizon=1)


def test_registered_model_name_format():
    assert registered_model_name(1, "classifier") == "gold-1d-classifier"
    assert registered_model_name(10, "regressor") == "gold-10d-regressor"


def test_registered_model_name_rejects_unknown_task():
    with pytest.raises(ValueError):
        registered_model_name(1, "not-a-real-task")


def test_promote_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    df = _synthetic_df()
    result = train_horizon(df, horizon=1, n_estimators=10)

    model_uri = result['model_uris']['classifier']
    version = promote(model_uri, horizon=1, task="classifier")
    assert version is not None

    loaded = load_promoted(horizon=1, task="classifier")

    sample = df.dropna(subset=result["feature_columns"])[result["feature_columns"]].head(5)
    preds = loaded.predict(sample)
    assert len(preds) == 5


def test_promote_moves_production_alias_to_latest_version(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    df = _synthetic_df()

    first = train_horizon(df, horizon=1, n_estimators=10)
    v1 = promote(first['model_uris']['classifier'], horizon=1, task="classifier")

    second = train_horizon(df, horizon=1, n_estimators=10)
    v2 = promote(second['model_uris']['classifier'], horizon=1, task="classifier")

    assert v2 != v1

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    name = registered_model_name(1, "classifier")
    aliased = client.get_model_version_by_alias(name, "production")
    assert aliased.version == v2
