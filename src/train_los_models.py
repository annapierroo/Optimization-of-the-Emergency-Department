"""Train LOS regression models from engineered parquet features.

This module migrates the model-training section of Hospital_LOS_Prediction_ML.ipynb
into executable Python code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import PipelineConfig, default_config

logger = logging.getLogger(__name__)


@dataclass
class LosTrainingArtifacts:
    best_model_path: Path
    encoders_path: Path
    metrics_path: Path
    leaderboard_path: Path


def _load_features(config: PipelineConfig) -> pd.DataFrame:
    path = config.feature_store_dir / config.features_filename
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found at {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded LOS training source from %s with shape %s", path, df.shape)
    return df


def _prepare_los_dataset(df: pd.DataFrame):
    """Prepare notebook-compatible LOS training matrix.

    Returns:
        X (DataFrame), y (Series), encoders (dict[str, LabelEncoder])
    """

    required = {
        "duration_hours_capped",
        "description_encoded",
        "reason_encoded",
        "time_of_day_encoded",
        "CODE",
        "BASE_COST",
        "start_hour",
        "start_day_of_week",
        "start_month",
        "start_year",
        "season",
        "is_weekend",
        "total_prior_encounters",
        "avg_prior_duration",
        "avg_prior_cost",
        "days_since_last_encounter",
        "encounters_last_30_days",
        "encounters_last_90_days",
        "has_reason",
        "is_emergency",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing LOS training columns: {sorted(missing)}")

    data = df.copy()
    data = data.dropna(subset=["duration_hours_capped"])

    feature_columns = [
        "start_hour",
        "start_day_of_week",
        "start_month",
        "start_year",
        "season",
        "is_weekend",
        "time_of_day_encoded",
        "total_prior_encounters",
        "avg_prior_duration",
        "avg_prior_cost",
        "days_since_last_encounter",
        "encounters_last_30_days",
        "encounters_last_90_days",
        "description_encoded",
        "reason_encoded",
        "CODE",
        "BASE_COST",
        "has_reason",
        "is_emergency",
    ]
    x = data[feature_columns].fillna(0)
    y = data["duration_hours_capped"]

    # Persistable metadata for downstream parity with notebook artifacts.
    encoders = {
        "description_encoded": None,
        "reason_encoded": None,
        "time_of_day_encoded": None,
    }
    return x, y, encoders


def _calculate_regression_metrics(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> dict:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = 0.0
    return {
        "Model": model_name,
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


def _train_models(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series):
    results = []
    trained_models = {}

    # Linear regression (scaled)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    lr = LinearRegression()
    lr.fit(x_train_scaled, y_train)
    lr_pred = lr.predict(x_test_scaled)
    results.append(_calculate_regression_metrics(y_test, lr_pred, "LinearRegression"))
    trained_models["LinearRegression"] = {"model": lr, "scaler": scaler}

    # Random forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    results.append(_calculate_regression_metrics(y_test, rf_pred, "RandomForest"))
    trained_models["RandomForest"] = {"model": rf, "scaler": None}

    # XGBoost with fallback
    try:
        import xgboost as xgb  # local import keeps module importable without xgboost

        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "XGBoost"
    except Exception:
        xgb_model = GradientBoostingRegressor(random_state=42)
        model_name = "GradientBoosting"

    xgb_model.fit(x_train, y_train)
    xgb_pred = xgb_model.predict(x_test)
    results.append(_calculate_regression_metrics(y_test, xgb_pred, model_name))
    trained_models[model_name] = {"model": xgb_model, "scaler": None}

    leaderboard = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    best_name = leaderboard.iloc[0]["Model"]
    return trained_models, leaderboard, best_name


def _save_los_artifacts(
    config: PipelineConfig,
    trained_models: dict,
    best_name: str,
    leaderboard: pd.DataFrame,
    encoders: dict,
) -> LosTrainingArtifacts:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    best_bundle = trained_models[best_name]

    model_path = config.model_dir / "best_los_model.joblib"
    encoders_path = config.model_dir / "los_encoders.joblib"
    metrics_path = config.model_dir / "los_metrics.json"
    leaderboard_path = config.model_dir / "los_leaderboard.csv"

    joblib.dump(best_bundle, model_path)
    joblib.dump(encoders, encoders_path)
    leaderboard.to_csv(leaderboard_path, index=False)
    metrics_path.write_text(
        json.dumps({"best_model": best_name, "leaderboard": leaderboard.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )

    logger.info("Saved LOS model artifacts under %s", config.model_dir)
    return LosTrainingArtifacts(
        best_model_path=model_path,
        encoders_path=encoders_path,
        metrics_path=metrics_path,
        leaderboard_path=leaderboard_path,
    )


def train_los_models(config: PipelineConfig | None = None) -> LosTrainingArtifacts:
    if config is None:
        config = default_config(Path("."))

    df = _load_features(config)
    x, y, encoders = _prepare_los_dataset(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.test_size, random_state=config.random_state)
    trained_models, leaderboard, best_name = _train_models(x_train, y_train, x_test, y_test)
    logger.info("LOS model leaderboard:\n%s", leaderboard)
    return _save_los_artifacts(config, trained_models, best_name, leaderboard, encoders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    train_los_models()
