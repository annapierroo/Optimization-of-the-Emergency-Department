"""Feature engineering pipeline for waiting-time prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PipelineConfig, default_config


class FeaturePipelinePort:
    """Creates model-ready feature sets."""

    def build_features(self):
        raise NotImplementedError


def _load_events(config):
    """Load processed log emitted by ingest_data."""

    processed_path = config.processed_data_dir / config.processed_filename
    if not processed_path.exists():
        # Helpful error message for debugging pipeline order
        raise FileNotFoundError(f"Processed event log not found at: {processed_path}. Did you run 'src/ingest_data.py'?")

    df = pd.read_csv(processed_path)
    df["start:timestamp"] = pd.to_datetime(df["start:timestamp"], utc=True, errors="coerce")
    df["end:timestamp"] = pd.to_datetime(df["end:timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["case:concept:name", "concept:name", "start:timestamp", "end:timestamp"])
    return df


def _build_waiting_time_features(events):
    """Build a superset feature table"""

    features = events.copy()
    features["Waiting_Time_Mins"] = (
        (features["end:timestamp"] - features["start:timestamp"]).dt.total_seconds().div(60)
    )
    features = features[features["Waiting_Time_Mins"] >= 0]
    features["Day_Index"] = features["start:timestamp"].dt.dayofweek
    features["Arrival_Hour"] = features["start:timestamp"].dt.hour

    # Notebook-compatible naming and temporal columns
    features["ENCOUNTER"] = features["case:concept:name"]
    # Processed dataset does not currently provide patient id; use encounter id as fallback key.
    features["PATIENT"] = features["case:concept:name"]
    features["START"] = features["start:timestamp"]
    features["STOP"] = features["end:timestamp"]
    features["DESCRIPTION"] = features["concept:name"]
    features["CODE"] = features["DESCRIPTION"].astype("category").cat.codes

    # Reason columns are not emitted by current ingestion; keep placeholders for notebook compatibility.
    features["REASONCODE"] = 0
    features["REASONDESCRIPTION"] = "No Reason Specified"
    features["has_reason"] = 0

    features["duration_hours"] = features["Waiting_Time_Mins"] / 60.0
    upper = features["duration_hours"].quantile(0.95)
    features["duration_hours_capped"] = features["duration_hours"].clip(upper=upper)

    features["start_hour"] = features["start:timestamp"].dt.hour
    features["start_day_of_week"] = features["start:timestamp"].dt.dayofweek
    features["start_month"] = features["start:timestamp"].dt.month
    features["start_year"] = features["start:timestamp"].dt.year
    features["start_day"] = features["start:timestamp"].dt.day
    features["is_weekend"] = (features["start_day_of_week"] >= 5).astype(int)

    features["season"] = features["start_month"].map(
        {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
    )

    def _time_of_day(hour):
        if 6 <= hour < 12:
            return "Morning"
        if 12 <= hour < 18:
            return "Afternoon"
        if 18 <= hour < 22:
            return "Evening"
        return "Night"

    features["time_of_day"] = features["start_hour"].apply(_time_of_day)
    features["time_of_day_encoded"] = features["time_of_day"].astype("category").cat.codes
    features["reason_encoded"] = features["REASONDESCRIPTION"].astype("category").cat.codes
    features["description_encoded"] = features["DESCRIPTION"].astype("category").cat.codes
    features["is_emergency"] = (
        features["DESCRIPTION"].astype(str).str.lower().str.contains("emergency|urgent|acute|trauma|critical")
    ).astype(int)

    # Placeholder notebook columns not available from current ingestion
    features["BASE_COST"] = 0.0
    features["total_prior_encounters"] = 0
    features["avg_prior_duration"] = 0.0
    features["avg_prior_cost"] = 0.0
    features["days_since_last_encounter"] = 0.0
    features["encounters_last_30_days"] = 0
    features["encounters_last_90_days"] = 0
    features["is_outlier"] = 0

    return features[
        [
            "case:concept:name",
            "concept:name",
            "start:timestamp",
            "end:timestamp",
            "Day_Index",
            "Arrival_Hour",
            "Waiting_Time_Mins",
            "ENCOUNTER",
            "PATIENT",
            "START",
            "STOP",
            "DESCRIPTION",
            "CODE",
            "REASONCODE",
            "REASONDESCRIPTION",
            "has_reason",
            "duration_hours",
            "duration_hours_capped",
            "start_hour",
            "start_day_of_week",
            "start_month",
            "start_year",
            "start_day",
            "season",
            "is_weekend",
            "time_of_day",
            "time_of_day_encoded",
            "reason_encoded",
            "description_encoded",
            "is_emergency",
            "BASE_COST",
            "total_prior_encounters",
            "avg_prior_duration",
            "avg_prior_cost",
            "days_since_last_encounter",
            "encounters_last_30_days",
            "encounters_last_90_days",
            "is_outlier",
        ]
    ]


def _save_features(config, features):
    """Persist feature table to the configured feature store directory."""

    output_path = config.feature_store_dir / config.features_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)
    return output_path


@dataclass
class DefaultFeaturePipeline(FeaturePipelinePort):
    """Feature builder aggregating encounter durations and procedures."""

    config: PipelineConfig

    def build_features(self):
        events = _load_events(self.config)
        features = _build_waiting_time_features(events)
        output_path = _save_features(self.config, features)
        print(f"Waiting-time features successfully stored at {output_path}")

# --- EXECUTION ENTRY POINT ---
# This block ensures the pipeline runs only when executed directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    cfg = default_config(project_root)
    DefaultFeaturePipeline(cfg).build_features()
