from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import PipelineConfig, default_config


def _load_parquet_events(config: PipelineConfig) -> pd.DataFrame:
    feature_path = config.feature_store_dir / config.features_filename
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found at {feature_path}")

    df = pd.read_parquet(feature_path)
    required = {"case:concept:name", "concept:name", "start:timestamp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in parquet: {sorted(missing)}")

    df["start:timestamp"] = pd.to_datetime(df["start:timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["case:concept:name", "concept:name", "start:timestamp"])
    return df


def train_next_activity(config: PipelineConfig | None = None):
    """Train an XGBoost classifier to predict the next activity from parquet features."""
    if config is None:
        config = default_config(Path("."))

    df = _load_parquet_events(config)
    df = df.sort_values(by=["case:concept:name", "start:timestamp"]).copy()
    df["Next_Activity"] = df.groupby("case:concept:name")["concept:name"].shift(-1)
    df = df.dropna(subset=["Next_Activity"])

    activity_counts = df["Next_Activity"].value_counts()
    common_activities = activity_counts[activity_counts >= 5].index
    df = df[df["Next_Activity"].isin(common_activities)]

    df["Hour"] = df["start:timestamp"].dt.hour
    df["Day_of_Week"] = df["start:timestamp"].dt.dayofweek

    le_input = LabelEncoder()
    df["Current_Activity_Encoded"] = le_input.fit_transform(df["concept:name"])

    le_output = LabelEncoder()
    df["Next_Activity_Encoded"] = le_output.fit_transform(df["Next_Activity"])

    x = df[["Current_Activity_Encoded", "Hour", "Day_of_Week"]]
    y = df["Next_Activity_Encoded"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(x_train, y_train)

    acc = accuracy_score(y_test, model.predict(x_test))
    print(f"Model Accuracy: {acc:.2%}")

    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_dir / config.next_activity_model_filename
    input_encoder_path = config.model_dir / config.next_activity_input_encoder_filename
    output_encoder_path = config.model_dir / config.next_activity_output_encoder_filename

    model.save_model(str(model_path))
    joblib.dump(le_input, input_encoder_path)
    joblib.dump(le_output, output_encoder_path)
    print(f"Success! Model and encoders saved to {config.model_dir}")


if __name__ == "__main__":
    train_next_activity()
