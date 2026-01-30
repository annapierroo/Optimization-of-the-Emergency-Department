"""Feature engineering pipeline for encounter-duration prediction."""

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

PROCESSED_FILENAME = "patient_journey_log.csv"
FEATURES_FILENAME = "encounter_features.parquet"


class FeaturePipelinePort:
    """Creates model-ready feature sets."""

    def build_features(self):
        raise NotImplementedError


def _load_events(config):
    """Load processed log emitted by ingest_data."""

    processed_path = config.processed_data_dir / PROCESSED_FILENAME
    if not processed_path.exists():
        # Helpful error message for debugging pipeline order
        raise FileNotFoundError(f"Processed event log not found at: {processed_path}. Did you run 'src/ingest_data.py'?")

    df = pd.read_csv(processed_path)
    df["start:timestamp"] = pd.to_datetime(df["start:timestamp"], utc=True, errors="coerce")
    df["end:timestamp"] = pd.to_datetime(df["end:timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["case:concept:name", "concept:name", "start:timestamp", "end:timestamp"])
    return df


def _summarize_encounter_duration(events):
    """Compute encounter duration and base stats per encounter."""

    summary = (
        events.groupby("case:concept:name")
        .agg(
            encounter_start=("start:timestamp", "min"),
            encounter_end=("end:timestamp", "max"),
            event_count=("concept:name", "count"),
        )
    )
    summary["encounter_duration_minutes"] = (
        (summary["encounter_end"] - summary["encounter_start"]).dt.total_seconds().div(60)
    )
    summary = summary[summary["encounter_duration_minutes"] > 0]
    summary["total_hours"] = summary["encounter_duration_minutes"] / 60.0
    return summary


def _procedure_matrix(events, top_k=50):
    """Pivot event log so each column counts a procedure for each encounter."""

    events["concept:name"] = events["concept:name"].astype(str).str.strip()
    top_procedures = events["concept:name"].value_counts().nlargest(top_k).index
    filtered = events[events["concept:name"].isin(top_procedures)]
    matrix = filtered.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0)
    matrix.columns = [f"proc_count__{col}" for col in matrix.columns]
    return matrix


def _save_features(config, features):
    """Persist feature table to the configured feature store directory."""

    output_path = config.feature_store_dir / FEATURES_FILENAME
    os.makedirs(output_path.parent, exist_ok=True)
    features.to_parquet(output_path)
    return output_path


@dataclass
class DefaultFeaturePipeline(FeaturePipelinePort):
    """Feature builder aggregating encounter durations and procedures."""

    config: PipelineConfig

    def build_features(self):
        events = _load_events(self.config)
        durations = _summarize_encounter_duration(events)
        procedures = _procedure_matrix(events)
        features = durations.join(procedures, how="left")
        features.fillna(0, inplace=True)
        output_path = _save_features(self.config, features)
        print(f"Encounter features successfully stored at {output_path}")

# --- EXECUTION ENTRY POINT ---
# This block ensures the pipeline runs only when executed directly
if __name__ == "__main__":
    print("🛠  Starting Feature Engineering Pipeline...")
    
    # Initialize configuration using the project root determined above
    cfg = default_config(PROJECT_ROOT)
    
    # Run the pipeline
    pipeline = DefaultFeaturePipeline(cfg)
    pipeline.build_features()
