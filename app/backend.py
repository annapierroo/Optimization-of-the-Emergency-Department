"""Minimal backend for Streamlit wait-time inference."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
import xgboost as xgb

from src.config import default_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOS_FEATURE_COLUMNS = [
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


def get_config():
    return default_config(PROJECT_ROOT)


def _monitoring_path() -> Path:
    config = get_config()
    return config.project_root / "data" / "monitoring" / "prediction_log.parquet"


def load_prediction_log() -> pd.DataFrame:
    """Load prediction monitoring log (may be empty)."""
    path = _monitoring_path()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def log_prediction_event(model_name: str, inputs: dict, prediction, meta: dict | None = None) -> str:
    """Append one inference event into monitoring parquet and return prediction id."""
    prediction_id = str(uuid4())
    row = {
        "prediction_id": prediction_id,
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "model_name": model_name,
        "prediction": str(prediction),
    }
    for key, value in inputs.items():
        row[f"input__{key}"] = value
    if meta:
        for key, value in meta.items():
            row[f"meta__{key}"] = value

    path = _monitoring_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_parquet(path)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_parquet(path, index=False)
    return prediction_id


def attach_actual_outcome(prediction_id: str, actual_outcome) -> float | None:
    """Attach actual outcome to an existing prediction row and return absolute error if numeric."""
    path = _monitoring_path()
    if not path.exists():
        raise FileNotFoundError(f"Prediction log not found at {path}")

    table = pd.read_parquet(path)
    if "prediction_id" not in table.columns:
        table["prediction_id"] = None
    if "actual_outcome" not in table.columns:
        table["actual_outcome"] = None
    if "abs_error" not in table.columns:
        table["abs_error"] = None

    mask = table["prediction_id"] == prediction_id
    if not mask.any():
        raise ValueError(f"Prediction id not found: {prediction_id}")

    table.loc[mask, "actual_outcome"] = str(actual_outcome)

    abs_error = None
    try:
        predicted = float(table.loc[mask, "prediction"].iloc[-1])
        actual = float(actual_outcome)
        abs_error = abs(actual - predicted)
        table.loc[mask, "abs_error"] = abs_error
    except (TypeError, ValueError):
        table.loc[mask, "abs_error"] = None

    table.to_parquet(path, index=False)
    return abs_error


def load_baseline_features() -> pd.DataFrame:
    """Load baseline feature table used for model training."""
    config = get_config()
    feature_path = config.feature_store_dir / config.features_filename
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature baseline not found at {feature_path}")
    return pd.read_parquet(feature_path)


def load_latest_model(model_dir: str | None = None):
    """Load wait-time model artifact from configured model directory."""
    if model_dir is None:
        config = default_config(PROJECT_ROOT)
        preferred = config.model_dir / config.model_filename
    else:
        preferred = Path(model_dir) / "xgb_model.json"

    if preferred.exists():
        if preferred.suffix == ".joblib":
            return joblib.load(preferred)
        model = xgb.XGBRegressor()
        model.load_model(str(preferred))
        return model

    fallback = preferred.with_suffix(".joblib")
    if fallback.exists():
        return joblib.load(fallback)

    raise FileNotFoundError(f"Model not found at {preferred} or {fallback}")


def predict_duration(model, day_index: int, arrival_hour: int) -> float:
    """Predict waiting time in minutes from day index and arrival hour."""
    frame = pd.DataFrame({"Day_Index": [int(day_index)], "Arrival_Hour": [int(arrival_hour)]})
    return float(model.predict(frame)[0])


def load_next_activity_assets():
    """Load next-activity model and encoders from configured artifact directory."""
    config = get_config()
    model_path = config.model_dir / config.next_activity_model_filename
    model_joblib_path = model_path.with_suffix(".joblib")
    input_encoder_path = config.model_dir / config.next_activity_input_encoder_filename
    output_encoder_path = config.model_dir / config.next_activity_output_encoder_filename

    if not input_encoder_path.exists() or not output_encoder_path.exists():
        raise FileNotFoundError(
            f"Missing encoder artifacts: {input_encoder_path} and/or {output_encoder_path}"
        )

    if model_path.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        resolved_model_path = model_path
    elif model_joblib_path.exists():
        model = joblib.load(model_joblib_path)
        resolved_model_path = model_joblib_path
    else:
        raise FileNotFoundError(f"Model not found at {model_path} or {model_joblib_path}")

    input_encoder = joblib.load(input_encoder_path)
    output_encoder = joblib.load(output_encoder_path)
    return model, input_encoder, output_encoder, resolved_model_path


def predict_next_activity(model, input_encoder, output_encoder, current_activity: str, hour: int, day_index: int) -> str:
    """Predict next activity label from current activity + time context."""
    encoded_current = input_encoder.transform([current_activity])[0]
    frame = pd.DataFrame(
        {
            "Current_Activity_Encoded": [int(encoded_current)],
            "Hour": [int(hour)],
            "Day_of_Week": [int(day_index)],
        }
    )
    predicted = int(model.predict(frame)[0])

    if hasattr(output_encoder, "inverse_transform"):
        return str(output_encoder.inverse_transform([predicted])[0])
    return str(output_encoder.classes_[predicted])


def load_los_assets():
    """Load LOS model bundle from artifact directory."""
    config = get_config()
    bundle_path = config.model_dir / config.los_best_model_filename
    if not bundle_path.exists():
        raise FileNotFoundError(f"LOS model bundle not found at {bundle_path}")

    bundle = joblib.load(bundle_path)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        scaler = bundle.get("scaler")
    else:
        model = bundle
        scaler = None
    return model, scaler, bundle_path


def predict_los(model, scaler, values: dict[str, float]) -> float:
    """Predict LOS in hours from engineered LOS feature vector."""
    row = {column: float(values.get(column, 0.0)) for column in LOS_FEATURE_COLUMNS}
    frame = pd.DataFrame([row])
    x_input = scaler.transform(frame) if scaler is not None else frame
    return float(model.predict(x_input)[0])


def load_los_procedure_options() -> pd.DataFrame:
    """Return available procedures with CODE and description encoding for LOS UI."""
    config = get_config()
    feature_path = config.feature_store_dir / config.features_filename
    if feature_path.exists():
        table = pd.read_parquet(feature_path, columns=["DESCRIPTION", "CODE", "description_encoded"])
        table = table.dropna(subset=["DESCRIPTION", "CODE", "description_encoded"]).copy()
        table["CODE"] = table["CODE"].astype(int)
        table["description_encoded"] = table["description_encoded"].astype(int)
        options = (
            table.groupby("DESCRIPTION", as_index=False)
            .agg({"CODE": "first", "description_encoded": "first"})
            .sort_values("DESCRIPTION")
            .reset_index(drop=True)
        )
        if not options.empty:
            return options

    raw_path = config.raw_data_dir / "EventLog.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path, sep=";", usecols=["DESCRIPTION"])
        raw = raw.dropna(subset=["DESCRIPTION"]).copy()
        raw["DESCRIPTION"] = raw["DESCRIPTION"].astype(str)
        unique = sorted(raw["DESCRIPTION"].unique().tolist())
        return pd.DataFrame(
            {
                "DESCRIPTION": unique,
                "CODE": list(range(len(unique))),
                "description_encoded": list(range(len(unique))),
            }
        )

    raise FileNotFoundError(f"No feature parquet or raw EventLog.csv found under {config.project_root}")
