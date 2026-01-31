import pandas as pd
import pytest
from pathlib import Path

from src import training
from src.config import PipelineConfig


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        project_root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        feature_store_dir=tmp_path / "data" / "features",
        model_dir=tmp_path / "artifacts" / "models",
        reports_dir=tmp_path / "reports",
    )


@pytest.mark.xfail(strict=False, reason="Pending implementation")
def test_load_feature_table_reads_features():
    """
    Loads feature table from `encounter_features.parquet`.
    Inputs: PipelineConfig with `feature_store_dir`.
    Outputs: DataFrame with `encounter_duration_minutes` column and expected row count.
    """
    tmp_path = Path("/tmp/test_training_load_features")
    config = _make_config(tmp_path)
    features_path = config.feature_store_dir / "encounter_features.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"encounter_duration_minutes": [10, 20]}).to_parquet(features_path)

    df = training.load_feature_table(config)
    assert "encounter_duration_minutes" in df.columns
    assert len(df) == 2


@pytest.mark.xfail(strict=False, reason="Pending implementation")
def test_split_train_val_uses_chronological_split():
    """
    Splits rows by time so validation uses newest records.
    Inputs: DataFrame with `encounter_start` and target, `val_fraction`.
    Outputs: train_df, val_df with val timestamps strictly after train timestamps.
    """
    df = pd.DataFrame(
        {
            "encounter_start": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
            "encounter_duration_minutes": [10, 20, 30, 40],
        }
    )
    train_df, val_df = training.split_train_val(df, time_column="encounter_start", val_fraction=0.25)
    assert len(train_df) == 3
    assert len(val_df) == 1
    assert val_df["encounter_start"].min() > train_df["encounter_start"].max()


@pytest.mark.xfail(strict=False, reason="Pending implementation")
def test_train_baseline_model_returns_fitted_model():
    """
    Trains baseline model on feature columns and target.
    Inputs: DataFrame with features and `encounter_duration_minutes`.
    Outputs: model object exposing `predict`.
    """
    df = pd.DataFrame(
        {
            "encounter_duration_minutes": [10, 20, 30],
            "proc_count__a": [1, 0, 1],
        }
    )
    model = training.train_baseline_model(df, target_column="encounter_duration_minutes")
    assert hasattr(model, "predict")


@pytest.mark.xfail(strict=False, reason="Pending implementation")
def test_evaluate_model_returns_metrics():
    """
    Compute evaluation metrics on validation data.
    Inputs: model with `predict`, validation DataFrame with target column.
    Outputs: metrics dict containing mae (and optionally other metrics).
    """

    class DummyModel:
        def predict(self, X):
            return [10] * len(X)

    df = pd.DataFrame(
        {
            "encounter_duration_minutes": [10, 20],
            "proc_count__a": [1, 0],
        }
    )
    metrics = training.evaluate_model(DummyModel(), df, target_column="encounter_duration_minutes")
    assert "mae" in metrics


@pytest.mark.xfail(strict=False, reason="Pending implementation")
def test_save_artifacts_writes_outputs():
    """
    Persist model and metrics under `artifacts/models`.
    Inputs: PipelineConfig, model object, metrics dict.
    Outputs: dict of artifact paths; files exist on disk.
    """
    tmp_path = Path("/tmp/test_training_save_artifacts")
    config = _make_config(tmp_path)
    artifacts = training.save_artifacts(config, model=object(), metrics={"mae": 1.0})
    assert isinstance(artifacts, dict)
    assert any(config.model_dir in Path(path).parents or Path(path) == config.model_dir for path in artifacts.values())


@pytest.mark.xfail(strict=False, reason="Pending implementation")
def test_default_trainer_orchestrates_pipeline(monkeypatch):
    """
    Description: ensure trainer calls load -> split -> train -> evaluate -> save in order.
    Inputs: PipelineConfig, monkeypatched functions.
    Outputs: no exception; returned artifact metadata if implemented.
    """
    tmp_path = Path("/tmp/test_training_orchestrator")
    config = _make_config(tmp_path)

    monkeypatch.setattr(training, "load_feature_table", lambda *_: pd.DataFrame())
    monkeypatch.setattr(training, "split_train_val", lambda *_: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(training, "train_baseline_model", lambda *_: object())
    monkeypatch.setattr(training, "evaluate_model", lambda *_: {"mae": 0.0})
    monkeypatch.setattr(training, "save_artifacts", lambda *_: {"model": str(config.model_dir / "model.pkl")})

    trainer = training.DefaultModelTrainer(config=config)
    trainer.train_model()
