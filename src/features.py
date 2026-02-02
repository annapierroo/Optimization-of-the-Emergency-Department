"""Feature engineering pipeline for waiting-time prediction."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --- SYSTEM PATH SETUP ---
# We add the project root to sys.path to allow imports from 'src' 
# even when running this file directly as a script (python src/features.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Changed relative import to absolute to prevent ImportError in standalone execution
from src.config import PipelineConfig, default_config


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
    """Build event-level features for waiting-time prediction."""

    features = events.copy()
    features["Waiting_Time_Mins"] = (
        (features["end:timestamp"] - features["start:timestamp"]).dt.total_seconds().div(60)
    )
    features = features[features["Waiting_Time_Mins"] >= 0]
    features["Day_Index"] = features["start:timestamp"].dt.dayofweek
    features["Arrival_Hour"] = features["start:timestamp"].dt.hour
    return features[
        [
            "case:concept:name",
            "concept:name",
            "start:timestamp",
            "end:timestamp",
            "Day_Index",
            "Arrival_Hour",
            "Waiting_Time_Mins",
        ]
    ]


def _save_features(config, features):
    """Persist feature table to the configured feature store directory."""

    output_path = config.feature_store_dir / config.features_filename
    os.makedirs(output_path.parent, exist_ok=True)
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
    print("🛠  Starting Feature Engineering Pipeline...")
    
    # Initialize configuration using the project root determined above
    cfg = default_config(PROJECT_ROOT)
    
    # Run the pipeline
    pipeline = DefaultFeaturePipeline(cfg)
    pipeline.build_features()
