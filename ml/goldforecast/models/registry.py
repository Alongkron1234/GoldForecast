"""Resolve which trained model is "promoted" (currently active) for a given
(horizon, task), using MLflow's built-in Model Registry as the source of
truth. A Postgres-backed pointer/cache table sits in front of this from
Issue #5 onward — this module is what that table will end up wrapping, so
Airflow and FastAPI never talk to MLflow directly."""

import mlflow
from mlflow.tracking import MlflowClient

from goldforecast import config

TASKS = ("regressor", "classifier", "baseline")


def registered_model_name(horizon: int, task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
    return f"gold-{horizon}d-{task}"


PRODUCTION_ALIAS = "production"


def promote(model_uri: str, horizon: int, task: str) -> str:
    """Register a trained model (from a finished MLflow run, e.g.
    `runs:/{run_id}/{artifact_path}`) under a stable name and point the
    "production" alias at it for (horizon, task) — an alias always points at
    exactly one version, so this atomically replaces whatever was promoted
    before. Returns the new version number.

    Uses MLflow's alias API rather than the older stage-based API (stages are
    deprecated as of MLflow 2.9 and slated for removal)."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    name = registered_model_name(horizon, task)
    result = mlflow.register_model(model_uri, name)

    client = MlflowClient()
    client.set_registered_model_alias(name=name, alias=PRODUCTION_ALIAS, version=result.version)
    return result.version


def load_promoted(horizon: int, task: str) -> mlflow.pyfunc.PyFuncModel:
    """Load the model currently behind the "production" alias for (horizon, task)."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    name = registered_model_name(horizon, task)
    return mlflow.pyfunc.load_model(f"models:/{name}@{PRODUCTION_ALIAS}")
