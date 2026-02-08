import pandas as pd
from pathlib import Path

from src.config import PipelineConfig
from src.features import DefaultFeaturePipeline


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
        metrics_filename="metrics.json",
        test_size=0.2,
        random_state=42,
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
    processed_path = config.processed_data_dir / config.processed_filename
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

    output_path = config.feature_store_dir / config.features_filename
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert {"Waiting_Time_Mins", "Day_Index", "Arrival_Hour"}.issubset(df.columns)
    assert {"duration_hours", "duration_hours_capped", "DESCRIPTION", "REASONDESCRIPTION"}.issubset(df.columns)
