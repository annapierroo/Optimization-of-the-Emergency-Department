import pandas as pd
from pathlib import Path

from src.config import PipelineConfig
from src.features import (
    FEATURES_FILENAME,
    PROCESSED_FILENAME,
    DefaultFeaturePipeline,
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


def test_default_pipeline_build_features_creates_output(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    rows = [
        {
            "case:concept:name": "enc_x",
            "concept:name": "proc_a",
            "start:timestamp": "2020-02-01T00:00:00Z",
            "end:timestamp": "2020-02-01T01:00:00Z",
        },
        {
            "case:concept:name": "enc_x",
            "concept:name": "proc_b",
            "start:timestamp": "2020-02-01T01:15:00Z",
            "end:timestamp": "2020-02-01T02:15:00Z",
        },
        {
            "case:concept:name": "enc_y",
            "concept:name": "proc_b",
            "start:timestamp": "2020-02-02T03:00:00Z",
            "end:timestamp": "2020-02-02T04:00:00Z",
        },
    ]
    processed_path = config.processed_data_dir / PROCESSED_FILENAME
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed_path, index=False)

    def fake_to_parquet(self, path, *_, **__):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_buffer = self.to_csv(index=False)
        path.write_text(csv_buffer, encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)

    pipeline = DefaultFeaturePipeline(config)
    pipeline.build_features()

    output_path = config.feature_store_dir / FEATURES_FILENAME
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert {"encounter_duration_minutes", "total_hours", "event_count"}.issubset(df.columns)
    assert any(col.startswith("proc_count__") for col in df.columns)
