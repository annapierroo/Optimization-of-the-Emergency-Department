"""Configuration objects and utilities for the emergency-department project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Root directories and resource identifiers."""

    project_root: Path
    data_path: Path
    raw_data_dir: Path
    processed_data_dir: Path
    processed_filename: str
    feature_store_dir: Path
    features_filename: str
    model_dir: Path
    model_filename: str
    metrics_filename: str
    test_size: float
    random_state: int
    reports_dir: Path
    los_best_model_filename: str = "best_los_model.joblib"
    los_encoders_filename: str = "los_encoders.joblib"
    los_metrics_filename: str = "los_metrics.json"
    los_leaderboard_filename: str = "los_leaderboard.csv"
    next_activity_model_filename: str = "next_activity_xgb.json"
    next_activity_input_encoder_filename: str = "input_encoder.pkl"
    next_activity_output_encoder_filename: str = "output_encoder.pkl"


def default_config(project_root: Path) -> PipelineConfig:
    """Return default directory layout."""

    return PipelineConfig(
        project_root=project_root,
        data_path=project_root / "data" / "raw" / "EventLog.csv",
        raw_data_dir=project_root / "data" / "raw",
        processed_data_dir=project_root / "data" / "processed",
        processed_filename="patient_journey_log.csv",
        feature_store_dir=project_root / "data" / "features",
        features_filename="encounter_features.parquet",
        model_dir=project_root / "artifacts" / "models",
        model_filename="xgb_model.json",
        metrics_filename="metrics.json",
        test_size=0.2,
        random_state=42,
        reports_dir=project_root / "reports",
        los_best_model_filename="best_los_model.joblib",
        los_encoders_filename="los_encoders.joblib",
        los_metrics_filename="los_metrics.json",
        los_leaderboard_filename="los_leaderboard.csv",
        next_activity_model_filename="next_activity_xgb.json",
        next_activity_input_encoder_filename="input_encoder.pkl",
        next_activity_output_encoder_filename="output_encoder.pkl",
    )
