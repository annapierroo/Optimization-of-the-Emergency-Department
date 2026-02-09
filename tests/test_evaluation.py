import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import PipelineConfig
from src.evaluation import DefaultEvaluator


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


def _write_feature_table(config: PipelineConfig) -> None:
    rows = []
    for idx in range(14):
        encounter = "c1" if idx < 7 else "c2"
        activity = "A" if idx % 2 == 0 else "B"
        ts = pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(hours=idx)
        rows.append(
            {
                "case:concept:name": encounter,
                "concept:name": activity,
                "start:timestamp": ts,
                "end:timestamp": ts + pd.Timedelta(minutes=30),
                "Day_Index": ts.dayofweek,
                "Arrival_Hour": ts.hour,
                "Waiting_Time_Mins": 30.0 + idx,
                "duration_hours_capped": 1.0 + (idx * 0.1),
                "description_encoded": idx % 3,
                "reason_encoded": 0,
                "time_of_day_encoded": idx % 4,
                "CODE": 100 + idx,
                "BASE_COST": 1000.0 + idx,
                "start_hour": ts.hour,
                "start_day_of_week": ts.dayofweek,
                "start_month": ts.month,
                "start_year": ts.year,
                "season": 1,
                "is_weekend": int(ts.dayofweek >= 5),
                "total_prior_encounters": idx // 2,
                "avg_prior_duration": 1.0,
                "avg_prior_cost": 1200.0,
                "days_since_last_encounter": 1.0,
                "encounters_last_30_days": 2,
                "encounters_last_90_days": 4,
                "has_reason": 0,
                "is_emergency": 0,
            }
        )

    feature_path = config.feature_store_dir / config.features_filename
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(feature_path)


class _IdentityScaler:
    def transform(self, x):
        return x


class _ConstantRegressor:
    def predict(self, x):
        return [1.0 for _ in range(len(x))]


class _SimpleEncoder:
    def __init__(self, classes):
        self.classes_ = list(classes)
        self._map = {label: idx for idx, label in enumerate(self.classes_)}

    def transform(self, values):
        return [self._map[value] for value in values]


def test_run_evaluation_writes_report_for_all_three_models(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    _write_feature_table(config)
    config.model_dir.mkdir(parents=True, exist_ok=True)

    # Wait-time and next-activity XGBoost paths are existence-gated in evaluator.
    (config.model_dir / config.model_filename).write_text("stub", encoding="utf-8")
    (config.model_dir / config.next_activity_model_filename).write_text("stub", encoding="utf-8")

    # LOS bundle
    joblib.dump(
        {"model": _ConstantRegressor(), "scaler": _IdentityScaler()},
        config.model_dir / config.los_best_model_filename,
    )
    # Next-activity encoders
    joblib.dump(_SimpleEncoder(["A", "B"]), config.model_dir / config.next_activity_input_encoder_filename)
    joblib.dump(_SimpleEncoder(["A", "B"]), config.model_dir / config.next_activity_output_encoder_filename)

    class _StubModel:
        def __init__(self, kind):
            self.kind = kind

        def predict(self, x):
            if self.kind == "classifier":
                return [0 for _ in range(len(x))]
            return [1.0 for _ in range(len(x))]

    monkeypatch.setattr(
        "src.evaluation._load_xgb_model",
        lambda _path, model_kind: _StubModel(model_kind),
    )

    DefaultEvaluator(config).run_evaluation()

    report_path = config.reports_dir / "evaluation_metrics.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report["results"].keys()) == {"wait_time", "los", "next_activity"}
    assert report["skipped"] == {}


def test_run_evaluation_raises_when_nothing_is_evaluated(tmp_path):
    config = _make_config(tmp_path)
    _write_feature_table(config)

    evaluator = DefaultEvaluator(config)
    try:
        evaluator.run_evaluation()
        assert False, "Expected run_evaluation to fail when no model artifacts are available"
    except RuntimeError as exc:
        assert "No models were evaluated" in str(exc)
