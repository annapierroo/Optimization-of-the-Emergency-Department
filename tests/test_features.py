import pandas as pd
import pytest
from pathlib import Path

from src.config import PipelineConfig
from src.features import (
    _build_waiting_time_features,
    _load_events,
    _save_features,
)


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        project_root=tmp_path,
        data_path=tmp_path / "data" / "raw" / "EventLog.csv",
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        processed_filename="patient_journey_log.csv",
        feature_store_dir=tmp_path / "data" / "features",
        features_filename="encounter_features.parquet",
        model_dir=tmp_path / "artifacts" / "models",
        model_filename="xgb_model.json",
        test_size=0.2,
        random_state=42,
        reports_dir=tmp_path / "reports",
    )


def _write_processed(tmp_path: Path, rows: list[dict]) -> Path:
    config = _make_config(tmp_path)
    processed_path = config.processed_data_dir / config.processed_filename
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed_path, index=False)
    return processed_path


def test__load_events_missing_file(tmp_path: Path):
    config = _make_config(tmp_path)
    with pytest.raises(FileNotFoundError):
        _load_events(config)


def test__load_events_parses_timestamps(tmp_path: Path):
    rows = [
        {
            "case:concept:name": "enc_1",
            "concept:name": "proc_a",
            "start:timestamp": "2020-01-01T00:00:00Z",
            "end:timestamp": "2020-01-01T01:00:00Z",
        },
        {
            "case:concept:name": "enc_1",
            "concept:name": "proc_b",
            "start:timestamp": "2020-01-01T01:15:00Z",
            "end:timestamp": "2020-01-01T01:45:00Z",
        },
    ]
    config = _make_config(tmp_path)
    processed_path = config.processed_data_dir / config.processed_filename
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed_path, index=False)

    events = _load_events(config)
    assert len(events) == 2
    assert events["start:timestamp"].dt.tz is not None
    assert events["end:timestamp"].dt.tz is not None


def test__build_waiting_time_features():
    df = pd.DataFrame(
        [
            {
                "case:concept:name": "enc_a",
                "concept:name": "proc_a",
                "start:timestamp": pd.Timestamp("2020-01-01T00:00:00Z"),
                "end:timestamp": pd.Timestamp("2020-01-01T01:00:00Z"),
            },
            {
                "case:concept:name": "enc_a",
                "concept:name": "proc_b",
                "start:timestamp": pd.Timestamp("2020-01-01T01:10:00Z"),
                "end:timestamp": pd.Timestamp("2020-01-01T02:00:00Z"),
            },
        ]
    )
    features = _build_waiting_time_features(df)
    assert len(features) == 2
    assert "Waiting_Time_Mins" in features.columns
    assert "Day_Index" in features.columns
    assert "Arrival_Hour" in features.columns
    assert features["Waiting_Time_Mins"].iloc[0] == 60


def test__save_features_writes_parquet(tmp_path: Path, monkeypatch):
    config = _make_config(tmp_path)
    features = pd.DataFrame(
        {"Waiting_Time_Mins": [60, 30], "Day_Index": [1, 2], "Arrival_Hour": [9, 10]},
        index=["enc_a", "enc_b"],
    )

    def fake_to_parquet(self, path, *_, **__):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet-placeholder")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)
    output_path = _save_features(config, features)
    assert output_path.exists()
    assert output_path.name == config.features_filename
