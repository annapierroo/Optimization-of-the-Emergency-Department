from pathlib import Path

import pandas as pd
import pytest

from src import training
from src.config import PipelineConfig


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
        test_size=0.25,
        random_state=42,
        reports_dir=tmp_path / "reports",
    )


def test_load_feature_table_reads_features(tmp_path: Path, monkeypatch):
    config = _make_config(tmp_path)
    feature_path = config.feature_store_dir / config.features_filename

    expected = pd.DataFrame(
        {
            "Waiting_Time_Mins": [10.0, 20.0],
            "Day_Index": [1, 2],
            "Arrival_Hour": [9, 10],
        }
    )

    monkeypatch.setattr(training.os.path, "exists", lambda p: str(p) == str(feature_path))
    monkeypatch.setattr(training.pd, "read_parquet", lambda _: expected.copy())

    result = training.load_feature_table(config)
    assert len(result) == 2
    assert {"Waiting_Time_Mins", "Day_Index", "Arrival_Hour"}.issubset(result.columns)


def test_load_feature_table_validates_required_columns(tmp_path: Path, monkeypatch):
    config = _make_config(tmp_path)
    monkeypatch.setattr(training.os.path, "exists", lambda _: True)
    monkeypatch.setattr(training.pd, "read_parquet", lambda _: pd.DataFrame({"Waiting_Time_Mins": [1.0]}))

    with pytest.raises(ValueError):
        training.load_feature_table(config)


def test_split_train_val_splits_data(tmp_path: Path):
    config = _make_config(tmp_path)
    df = pd.DataFrame(
        {
            "Day_Index": [0, 1, 2, 3],
            "Arrival_Hour": [8, 9, 10, 11],
            "Waiting_Time_Mins": [15.0, 20.0, 25.0, 30.0],
        }
    )

    x_train, x_test, y_train, y_test = training.split_train_val(df, config)
    assert len(x_train) + len(x_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert list(x_train.columns) == ["Day_Index", "Arrival_Hour"]


def test_train_baseline_model_returns_fitted_model(tmp_path: Path, monkeypatch):
    config = _make_config(tmp_path)
    x_train = pd.DataFrame({"Day_Index": [1, 2], "Arrival_Hour": [9, 10]})
    y_train = pd.Series([10.0, 20.0])

    class FakeRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_called = False

        def fit(self, x, y):
            self.fit_called = True
            self.x = x
            self.y = y
            return self

    monkeypatch.setattr(training.xgb, "XGBRegressor", FakeRegressor)
    model = training.train_baseline_model(x_train, y_train, config)

    assert model.fit_called is True
    assert model.kwargs["random_state"] == config.random_state


def test_evaluate_model_returns_mae():
    class DummyModel:
        def predict(self, x):
            return [10.0 for _ in range(len(x))]

    x_test = pd.DataFrame({"Day_Index": [1, 2], "Arrival_Hour": [9, 10]})
    y_test = pd.Series([10.0, 20.0])

    metrics = training.evaluate_model(DummyModel(), x_test, y_test)
    assert "mae" in metrics
    assert metrics["mae"] == pytest.approx(5.0)


def test_save_artifacts_writes_model_and_metrics(tmp_path: Path):
    config = _make_config(tmp_path)

    class DummyModel:
        def save_model(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("model", encoding="utf-8")

    artifacts = training.save_artifacts(config, DummyModel(), {"mae": 1.23})
    assert Path(artifacts["model_path"]).exists()
    assert Path(artifacts["metrics_path"]).exists()


def test_default_trainer_orchestrates_pipeline(tmp_path: Path, monkeypatch):
    config = _make_config(tmp_path)
    call_order = []

    monkeypatch.setattr(training, "load_feature_table", lambda *_: call_order.append("load") or pd.DataFrame())
    monkeypatch.setattr(
        training,
        "split_train_val",
        lambda *_: call_order.append("split")
        or (pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)),
    )
    monkeypatch.setattr(training, "train_baseline_model", lambda *_: call_order.append("train") or object())
    monkeypatch.setattr(training, "evaluate_model", lambda *_: call_order.append("evaluate") or {"mae": 0.0})
    monkeypatch.setattr(training, "save_artifacts", lambda *_: call_order.append("save") or {})

    trainer = training.DefaultModelTrainer(config=config)
    trainer.train_model()

    assert call_order == ["load", "split", "train", "evaluate", "save"]
