import importlib
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

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


def _install_ml_stubs(monkeypatch: pytest.MonkeyPatch):
    class StubXGBBase:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_called = False
            self._pred = 0

        def fit(self, x, y):
            self.fit_called = True
            self._pred = float(y.iloc[0] if hasattr(y, "iloc") else y[0]) if len(y) else 0.0
            return self

        def predict(self, x):
            return [self._pred for _ in range(len(x))]

        def save_model(self, path: str):
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("stub-model", encoding="utf-8")

    xgb_module = types.ModuleType("xgboost")
    xgb_module.XGBRegressor = StubXGBBase
    xgb_module.XGBClassifier = StubXGBBase

    def _slice(data, start, end):
        if hasattr(data, "iloc"):
            return data.iloc[start:end].copy()
        return data[start:end]

    def train_test_split(x, y, test_size=0.2, random_state=None, stratify=None):
        del random_state, stratify
        n = len(x)
        test_n = max(1, int(round(n * test_size))) if n > 1 else 1
        split = n - test_n
        return _slice(x, 0, split), _slice(x, split, n), _slice(y, 0, split), _slice(y, split, n)

    def mean_absolute_error(y_true, y_pred):
        pairs = list(zip(list(y_true), list(y_pred)))
        return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else 0.0

    def mean_squared_error(y_true, y_pred):
        pairs = list(zip(list(y_true), list(y_pred)))
        return sum((a - b) ** 2 for a, b in pairs) / len(pairs) if pairs else 0.0

    def r2_score(y_true, y_pred):
        y_true = list(y_true)
        y_pred = list(y_pred)
        if not y_true:
            return 0.0
        mean_y = sum(y_true) / len(y_true)
        ss_tot = sum((v - mean_y) ** 2 for v in y_true)
        if ss_tot == 0:
            return 0.0
        ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
        return 1 - (ss_res / ss_tot)

    def accuracy_score(y_true, y_pred):
        y_true = list(y_true)
        y_pred = list(y_pred)
        if not y_true:
            return 0.0
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        return correct / len(y_true)

    class LabelEncoder:
        def fit_transform(self, values):
            uniques = {}
            for value in values:
                if value not in uniques:
                    uniques[value] = len(uniques)
            self.classes_ = list(uniques.keys())
            return [uniques[value] for value in values]

        def transform(self, values):
            lookup = {value: idx for idx, value in enumerate(self.classes_)}
            return [lookup[value] for value in values]

    class StandardScaler:
        def fit_transform(self, x):
            return x

        def transform(self, x):
            return x

    class LinearRegression:
        def fit(self, x, y):
            del x
            self._pred = float(sum(y) / len(y)) if len(y) else 0.0
            return self

        def predict(self, x):
            return [self._pred for _ in range(len(x))]

    class TreeRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x, y):
            del x
            self._pred = float(sum(y) / len(y)) if len(y) else 0.0
            return self

        def predict(self, x):
            return [self._pred for _ in range(len(x))]

    sklearn_module = types.ModuleType("sklearn")
    model_selection_module = types.ModuleType("sklearn.model_selection")
    metrics_module = types.ModuleType("sklearn.metrics")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    linear_model_module = types.ModuleType("sklearn.linear_model")
    ensemble_module = types.ModuleType("sklearn.ensemble")

    model_selection_module.train_test_split = train_test_split
    metrics_module.mean_absolute_error = mean_absolute_error
    metrics_module.mean_squared_error = mean_squared_error
    metrics_module.r2_score = r2_score
    metrics_module.accuracy_score = accuracy_score
    preprocessing_module.LabelEncoder = LabelEncoder
    preprocessing_module.StandardScaler = StandardScaler
    linear_model_module.LinearRegression = LinearRegression
    ensemble_module.RandomForestRegressor = TreeRegressor
    ensemble_module.GradientBoostingRegressor = TreeRegressor

    monkeypatch.setitem(sys.modules, "xgboost", xgb_module)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_module)
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", model_selection_module)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics_module)
    monkeypatch.setitem(sys.modules, "sklearn.preprocessing", preprocessing_module)
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", linear_model_module)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", ensemble_module)


def _import_module(module_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_load_feature_table_reads_features(tmp_path: Path, monkeypatch):
    _install_ml_stubs(monkeypatch)
    training = _import_module("src.train_wait_time")

    config = _make_config(tmp_path)
    feature_path = config.feature_store_dir / config.features_filename
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"Waiting_Time_Mins": [10.0, 20.0], "Day_Index": [1, 2], "Arrival_Hour": [9, 10]}
    ).to_parquet(feature_path)

    result = training.load_feature_table(config)
    assert len(result) == 2
    assert {"Waiting_Time_Mins", "Day_Index", "Arrival_Hour"}.issubset(result.columns)


def test_load_feature_table_validates_required_columns(tmp_path: Path, monkeypatch):
    _install_ml_stubs(monkeypatch)
    training = _import_module("src.train_wait_time")

    config = _make_config(tmp_path)
    feature_path = config.feature_store_dir / config.features_filename
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Waiting_Time_Mins": [1.0]}).to_parquet(feature_path)

    with pytest.raises(ValueError):
        training.load_feature_table(config)


def test_split_train_val_splits_data(tmp_path: Path, monkeypatch):
    _install_ml_stubs(monkeypatch)
    training = _import_module("src.train_wait_time")

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
    _install_ml_stubs(monkeypatch)
    training = _import_module("src.train_wait_time")

    config = _make_config(tmp_path)
    x_train = pd.DataFrame({"Day_Index": [1, 2], "Arrival_Hour": [9, 10]})
    y_train = pd.Series([10.0, 20.0])
    model = training.train_baseline_model(x_train, y_train, config)

    assert model.fit_called is True
    assert model.kwargs["random_state"] == config.random_state


def test_evaluate_model_returns_mae(monkeypatch):
    _install_ml_stubs(monkeypatch)
    training = _import_module("src.train_wait_time")

    class DummyModel:
        def predict(self, x):
            return [10.0 for _ in range(len(x))]

    x_test = pd.DataFrame({"Day_Index": [1, 2], "Arrival_Hour": [9, 10]})
    y_test = pd.Series([10.0, 20.0])

    metrics = training.evaluate_model(DummyModel(), x_test, y_test)
    assert "mae" in metrics
    assert metrics["mae"] == pytest.approx(5.0)


def test_save_artifacts_writes_model_and_metrics(tmp_path: Path, monkeypatch):
    _install_ml_stubs(monkeypatch)
    training = _import_module("src.train_wait_time")

    config = _make_config(tmp_path)

    class DummyModel:
        def save_model(self, path):
            model_path = Path(path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text("model", encoding="utf-8")

    artifacts = training.save_artifacts(config, DummyModel(), {"mae": 1.23})
    assert Path(artifacts["model_path"]).exists()
    assert Path(artifacts["metrics_path"]).exists()


def test_default_trainer_orchestrates_pipeline(tmp_path: Path, monkeypatch):
    _install_ml_stubs(monkeypatch)
    training = _import_module("src.train_wait_time")

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


def test_train_next_activity_saves_artifacts(tmp_path: Path, monkeypatch):
    _install_ml_stubs(monkeypatch)
    train_next_activity = _import_module("src.train_next_activity")

    config = _make_config(tmp_path)
    feature_path = config.feature_store_dir / config.features_filename
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(
        {
            "case:concept:name": ["c1"] * 7 + ["c2"] * 7,
            "concept:name": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
            ],
            "start:timestamp": pd.date_range("2024-01-01", periods=14, freq="h", tz="UTC"),
        }
    )
    data.to_parquet(feature_path)

    train_next_activity.train_next_activity(config)
    assert (config.model_dir / config.next_activity_model_filename).exists()
    assert (config.model_dir / config.next_activity_input_encoder_filename).exists()
    assert (config.model_dir / config.next_activity_output_encoder_filename).exists()


def test_train_next_activity_validates_required_columns(tmp_path: Path, monkeypatch):
    _install_ml_stubs(monkeypatch)
    train_next_activity = _import_module("src.train_next_activity")

    config = _make_config(tmp_path)
    feature_path = config.feature_store_dir / config.features_filename
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case:concept:name": ["c1"]}).to_parquet(feature_path)

    with pytest.raises(ValueError):
        train_next_activity._load_parquet_events(config)


def test_prepare_los_dataset_returns_matrix_and_target(monkeypatch):
    _install_ml_stubs(monkeypatch)
    train_los_models = _import_module("src.train_los_models")

    df = pd.DataFrame(
        {
            "duration_hours_capped": [1.0, 2.0],
            "description_encoded": [0, 1],
            "reason_encoded": [1, 0],
            "time_of_day_encoded": [0, 1],
            "CODE": [101, 102],
            "BASE_COST": [100.0, 200.0],
            "start_hour": [10, 11],
            "start_day_of_week": [1, 2],
            "start_month": [1, 1],
            "start_year": [2024, 2024],
            "season": [0, 0],
            "is_weekend": [0, 0],
            "total_prior_encounters": [0, 1],
            "avg_prior_duration": [0.0, 1.2],
            "avg_prior_cost": [0.0, 150.0],
            "days_since_last_encounter": [0, 7],
            "encounters_last_30_days": [0, 1],
            "encounters_last_90_days": [0, 2],
            "has_reason": [1, 1],
            "is_emergency": [1, 0],
        }
    )

    x, y, encoders = train_los_models._prepare_los_dataset(df)
    assert len(x) == 2
    assert len(y) == 2
    assert {"description_encoded", "reason_encoded", "time_of_day_encoded"} == set(encoders.keys())


def test_train_los_models_writes_artifacts(tmp_path: Path, monkeypatch):
    _install_ml_stubs(monkeypatch)
    train_los_models = _import_module("src.train_los_models")

    config = _make_config(tmp_path)
    feature_path = config.feature_store_dir / config.features_filename
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "duration_hours_capped": [1.0, 2.0, 3.0, 4.0],
            "description_encoded": [0, 1, 0, 1],
            "reason_encoded": [1, 0, 1, 0],
            "time_of_day_encoded": [0, 1, 2, 3],
            "CODE": [101, 102, 103, 104],
            "BASE_COST": [100.0, 200.0, 300.0, 400.0],
            "start_hour": [10, 11, 12, 13],
            "start_day_of_week": [1, 2, 3, 4],
            "start_month": [1, 1, 1, 1],
            "start_year": [2024, 2024, 2024, 2024],
            "season": [0, 0, 0, 0],
            "is_weekend": [0, 0, 0, 0],
            "total_prior_encounters": [0, 1, 2, 3],
            "avg_prior_duration": [0.0, 1.2, 1.3, 1.5],
            "avg_prior_cost": [0.0, 150.0, 175.0, 200.0],
            "days_since_last_encounter": [0, 7, 3, 2],
            "encounters_last_30_days": [0, 1, 1, 2],
            "encounters_last_90_days": [0, 2, 2, 3],
            "has_reason": [1, 1, 1, 1],
            "is_emergency": [1, 0, 1, 0],
        }
    ).to_parquet(feature_path)

    artifacts = train_los_models.train_los_models(config)
    assert artifacts.best_model_path.exists()
    assert artifacts.encoders_path.exists()
    assert artifacts.metrics_path.exists()
    assert artifacts.leaderboard_path.exists()

    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert "best_model" in metrics
    assert "leaderboard" in metrics
