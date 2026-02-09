"""Model evaluation components."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import joblib
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class EvaluatorPort(Protocol):
    """Computes metrics and diagnostics."""

    def run_evaluation(self) -> None:
        """Store metric reports."""


@dataclass
class DefaultEvaluator(EvaluatorPort):
    """Placeholder evaluator."""

    config: PipelineConfig

    def run_evaluation(self) -> None:
        features = _load_feature_table(self.config)
        results: dict[str, dict] = {}
        skipped: dict[str, str] = {}

        evaluators = {
            "wait_time": _evaluate_wait_time_model,
            "los": _evaluate_los_model,
            "next_activity": _evaluate_next_activity_model,
        }

        for model_name, evaluator in evaluators.items():
            try:
                results[model_name] = evaluator(self.config, features)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                logger.warning("Skipping %s evaluation: %s", model_name, exc)
                skipped[model_name] = str(exc)

        report = {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_path": str(self.config.feature_store_dir / self.config.features_filename),
            "results": results,
            "skipped": skipped,
        }
        report_path = _write_report(self.config, report)
        logger.info("Wrote evaluation report to %s", report_path)

        if not results:
            raise RuntimeError(f"No models were evaluated. Details: {skipped}")


def _load_feature_table(config: PipelineConfig) -> pd.DataFrame:
    feature_path = config.feature_store_dir / config.features_filename
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found at {feature_path}")
    return pd.read_parquet(feature_path)


def _evaluate_wait_time_model(config: PipelineConfig, table: pd.DataFrame) -> dict:
    required = {"Day_Index", "Arrival_Hour", "Waiting_Time_Mins"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing wait-time evaluation columns: {missing}")

    model_path = config.model_dir / config.model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Wait-time model not found at {model_path}")

    model = _load_xgb_model(model_path, model_kind="regressor")
    x = table[["Day_Index", "Arrival_Hour"]]
    y = table["Waiting_Time_Mins"]
    pred = _to_float_list(model.predict(x))

    return {
        "model_path": str(model_path),
        "rows": len(x),
        "mae": _mae(y, pred),
        "rmse": _rmse(y, pred),
        "r2": _r2(y, pred),
    }


def _evaluate_los_model(config: PipelineConfig, table: pd.DataFrame) -> dict:
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
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing LOS evaluation columns: {missing}")

    model_path = config.model_dir / config.los_best_model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"LOS model bundle not found at {model_path}")

    bundle = joblib.load(model_path)
    model = bundle.get("model") if isinstance(bundle, dict) else bundle
    scaler = bundle.get("scaler") if isinstance(bundle, dict) else None
    if model is None:
        raise ValueError(f"Invalid LOS model bundle at {model_path}: missing 'model'")

    data = table.dropna(subset=["duration_hours_capped"]).copy()
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
    x_input = scaler.transform(x) if scaler is not None else x
    pred = _to_float_list(model.predict(x_input))

    return {
        "model_path": str(model_path),
        "rows": len(x),
        "mae": _mae(y, pred),
        "rmse": _rmse(y, pred),
        "r2": _r2(y, pred),
        "mape": _mape(y, pred),
    }


def _evaluate_next_activity_model(config: PipelineConfig, table: pd.DataFrame) -> dict:
    required = {"case:concept:name", "concept:name", "start:timestamp"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing next-activity evaluation columns: {missing}")

    model_path = config.model_dir / config.next_activity_model_filename
    input_encoder_path = config.model_dir / config.next_activity_input_encoder_filename
    output_encoder_path = config.model_dir / config.next_activity_output_encoder_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Next-activity model not found at {model_path}")
    if not input_encoder_path.exists() or not output_encoder_path.exists():
        raise FileNotFoundError(
            f"Next-activity encoders not found at {input_encoder_path} and {output_encoder_path}"
        )

    input_encoder = joblib.load(input_encoder_path)
    output_encoder = joblib.load(output_encoder_path)
    model = _load_xgb_model(model_path, model_kind="classifier")

    events = table[["case:concept:name", "concept:name", "start:timestamp"]].copy()
    events["start:timestamp"] = pd.to_datetime(events["start:timestamp"], utc=True, errors="coerce")
    events = events.dropna(subset=["case:concept:name", "concept:name", "start:timestamp"])
    events = events.sort_values(by=["case:concept:name", "start:timestamp"])
    events["Next_Activity"] = events.groupby("case:concept:name")["concept:name"].shift(-1)
    events = events.dropna(subset=["Next_Activity"])

    events = events[events["concept:name"].isin(getattr(input_encoder, "classes_", []))]
    events = events[events["Next_Activity"].isin(getattr(output_encoder, "classes_", []))]
    if events.empty:
        raise ValueError("No valid rows available for next-activity evaluation after encoder filtering")

    events["Hour"] = events["start:timestamp"].dt.hour
    events["Day_of_Week"] = events["start:timestamp"].dt.dayofweek
    events["Current_Activity_Encoded"] = input_encoder.transform(events["concept:name"])
    y_true = output_encoder.transform(events["Next_Activity"])

    x = events[["Current_Activity_Encoded", "Hour", "Day_of_Week"]]
    y_pred = model.predict(x)
    accuracy = _accuracy(y_true, y_pred)

    return {
        "model_path": str(model_path),
        "rows": len(x),
        "accuracy": accuracy,
    }


def _load_xgb_model(model_path: Path, model_kind: str):
    try:
        import xgboost as xgb
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("xgboost is required to evaluate xgb models") from exc

    if model_kind == "classifier":
        model = xgb.XGBClassifier()
    elif model_kind == "regressor":
        model = xgb.XGBRegressor()
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    model.load_model(str(model_path))
    return model


def _write_report(config: PipelineConfig, report: dict) -> Path:
    report_path = config.reports_dir / "evaluation_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _to_float_list(values) -> list[float]:
    return [float(v) for v in values]


def _mae(y_true, y_pred) -> float:
    true_values = _to_float_list(y_true)
    pred_values = _to_float_list(y_pred)
    if not true_values:
        return 0.0
    return sum(abs(a - b) for a, b in zip(true_values, pred_values)) / len(true_values)


def _rmse(y_true, y_pred) -> float:
    true_values = _to_float_list(y_true)
    pred_values = _to_float_list(y_pred)
    if not true_values:
        return 0.0
    mse = sum((a - b) ** 2 for a, b in zip(true_values, pred_values)) / len(true_values)
    return mse ** 0.5


def _r2(y_true, y_pred) -> float:
    true_values = _to_float_list(y_true)
    pred_values = _to_float_list(y_pred)
    if not true_values:
        return 0.0
    mean_true = sum(true_values) / len(true_values)
    ss_tot = sum((value - mean_true) ** 2 for value in true_values)
    if ss_tot == 0:
        return 0.0
    ss_res = sum((a - b) ** 2 for a, b in zip(true_values, pred_values))
    return 1 - (ss_res / ss_tot)


def _mape(y_true, y_pred) -> float:
    true_values = _to_float_list(y_true)
    pred_values = _to_float_list(y_pred)
    values = [abs((a - b) / a) for a, b in zip(true_values, pred_values) if a != 0]
    if not values:
        return 0.0
    return (sum(values) / len(values)) * 100


def _accuracy(y_true, y_pred) -> float:
    true_values = list(y_true)
    pred_values = list(y_pred)
    if not true_values:
        return 0.0
    correct = sum(1 for a, b in zip(true_values, pred_values) if a == b)
    return correct / len(true_values)
