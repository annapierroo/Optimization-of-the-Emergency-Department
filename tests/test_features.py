import pandas as pd
import pytest
from pathlib import Path

from src.config import PipelineConfig
from src.features import (
    FEATURES_FILENAME,
    PROCESSED_FILENAME,
    _load_events,
    _procedure_matrix,
    _save_features,
    _summarize_encounter_duration,
)


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        project_root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        feature_store_dir=tmp_path / "data" / "features",
        model_dir=tmp_path / "artifacts" / "models",
        reports_dir=tmp_path / "reports",
    )


def _write_processed(tmp_path: Path, rows: list[dict]) -> Path:
    config = _make_config(tmp_path)
    processed_path = config.processed_data_dir / PROCESSED_FILENAME
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
    processed_path = config.processed_data_dir / PROCESSED_FILENAME
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed_path, index=False)

    events = _load_events(config)
    assert len(events) == 2
    assert events["start:timestamp"].dt.tz is not None
    assert events["end:timestamp"].dt.tz is not None


def test__summarize_encounter_duration_positive():
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
    summary = _summarize_encounter_duration(df)
    assert summary.loc["enc_a", "event_count"] == 2
    assert summary.loc["enc_a", "encounter_duration_minutes"] == 120


def test__procedure_matrix_counts_topk():
    df = pd.DataFrame(
        [
            {"case:concept:name": "enc_a", "concept:name": "proc_a"},
            {"case:concept:name": "enc_a", "concept:name": "proc_a"},
            {"case:concept:name": "enc_a", "concept:name": "proc_b"},
            {"case:concept:name": "enc_b", "concept:name": "proc_b"},
        ]
    )
    matrix = _procedure_matrix(df, top_k=2)
    assert "proc_count__proc_a" in matrix.columns
    assert matrix.loc["enc_a", "proc_count__proc_a"] == 2
    assert matrix.loc["enc_b", "proc_count__proc_a"] == 0


def test__save_features_writes_parquet(tmp_path: Path, monkeypatch):
    config = _make_config(tmp_path)
    features = pd.DataFrame(
        {"encounter_duration_minutes": [60, 30], "total_hours": [1, 0.5]},
        index=["enc_a", "enc_b"],
    )

    def fake_to_parquet(self, path, *_, **__):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet-placeholder")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)
    output_path = _save_features(config, features)
    assert output_path.exists()
    assert output_path.name == FEATURES_FILENAME
