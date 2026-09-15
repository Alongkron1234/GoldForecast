"""Train regression + classification models (plus a Logistic Regression
baseline) for one forecast horizon, logging everything to MLflow."""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from goldforecast import config
from goldforecast.features.build import stationary_feature_columns
from goldforecast.split import chronological_split


def train_horizon(df: pd.DataFrame, horizon: int, n_estimators: int = 300) -> dict:
    """Train regressor + classifier + LR baseline for `horizon` and log the run
    to MLflow. `df` must already have `labels.make_targets(df, horizon)` applied.

    Rows with an unknown target (the last `horizon` rows) or any missing
    feature value (the rolling-window warmup at the start) are dropped before
    training/splitting — neither is a valid training sample.

    Returns the trained models and their validation/test metrics.
    """
    return_col = f"return_pct_t{horizon}"
    direction_col = f"direction_t{horizon}"
    feature_cols = stationary_feature_columns(df)

    clean = df.dropna(subset=[return_col, direction_col, *feature_cols]).reset_index(drop=True)
    train_df, val_df, test_df = chronological_split(clean)

    X_train, X_val, X_test = (train_df[feature_cols], val_df[feature_cols], test_df[feature_cols])
    y_train_reg, y_val_reg, y_test_reg = (
        train_df[return_col],
        val_df[return_col],
        test_df[return_col],
    )
    y_train_dir = (train_df[direction_col] == "up").astype(int)
    y_val_dir = (val_df[direction_col] == "up").astype(int)
    y_test_dir = (test_df[direction_col] == "up").astype(int)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("gold-forecast")

    with mlflow.start_run(run_name=f"gold-{horizon}d") as run:
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size", len(val_df))
        mlflow.log_param("test_size", len(test_df))

        # Regularized on purpose: an untuned XGBoost (n_estimators=300, max_depth=4,
        # no regularization) hit 100% train accuracy on this data and underperformed
        # a majority-class baseline out of sample — classic overfitting on noisy
        # daily financial data. These settings were chosen by comparing train/val
        # accuracy gap across a small grid (see PR discussion / Issue #3 notes);
        # they don't close the gap to zero, but they stop the model from memorizing.
        model_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.03,
            reg_lambda=5,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            random_state=42,
        )

        regressor = XGBRegressor(**model_kwargs)
        regressor.fit(X_train, y_train_reg)
        metrics = {
            "regression_val_mae": mean_absolute_error(y_val_reg, regressor.predict(X_val)),
            "regression_test_mae": mean_absolute_error(y_test_reg, regressor.predict(X_test)),
        }
        regressor_info = mlflow.xgboost.log_model(regressor, name=f"regressor_h{horizon}")

        classifier = XGBClassifier(**model_kwargs, eval_metric="logloss")
        classifier.fit(X_train, y_train_dir)
        metrics["classification_val_accuracy"] = accuracy_score(
            y_val_dir, classifier.predict(X_val)
        )
        metrics["classification_test_accuracy"] = accuracy_score(
            y_test_dir, classifier.predict(X_test)
        )
        classifier_info = mlflow.xgboost.log_model(classifier, name=f"classifier_h{horizon}")

        # Logistic Regression (unlike the tree models above) is sensitive to
        # feature scale — our features range from raw prices (~1000s) to RSI
        # (0-100) to returns (~+-10), so it needs standardizing to converge.
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        baseline.fit(X_train, y_train_dir)
        metrics["baseline_val_accuracy"] = accuracy_score(y_val_dir, baseline.predict(X_val))
        metrics["baseline_test_accuracy"] = accuracy_score(y_test_dir, baseline.predict(X_test))
        baseline_info = mlflow.sklearn.log_model(baseline, name=f"baseline_h{horizon}")

        mlflow.log_metrics(metrics)

    return {
        "horizon": horizon,
        "run_id": run.info.run_id,
        "regressor": regressor,
        "classifier": classifier,
        "baseline": baseline,
        "metrics": metrics,
        "feature_columns": feature_cols,
        "model_uris": {
            "regressor": regressor_info.model_uri,
            "classifier": classifier_info.model_uri,
            "baseline": baseline_info.model_uri,
        },
    }
