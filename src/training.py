"""
Model training module.
Implements the training pipeline structure defined by the team, 
filling the logic to train and persist the XGBoost model.
"""

import pandas as pd
import xgboost as xgb
import os
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass, field  # <--- AGGIUNTO 'field'
from pathlib import Path
from typing import Protocol

from .config import PipelineConfig, default_config

logger = logging.getLogger(__name__)

class ModelTrainerPort(Protocol):
    """Interface used by pipeline orchestration for trainer components."""

    def train_model(self) -> None:
        """Execute model training workflow."""

def load_feature_table(config: PipelineConfig) -> pd.DataFrame:
    """Load engineered features from disk."""
    data_path = config.feature_store_dir / config.features_filename
    logger.info("Loading features from %s", data_path)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature file not found at {data_path}")

    df = pd.read_parquet(data_path)
    required_columns = {"Waiting_Time_Mins", "Day_Index", "Arrival_Hour"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    
    print(f"[INFO] Data loaded successfully. Shape: {df.shape}")
    return df

def split_train_val(df: pd.DataFrame, config: PipelineConfig):
    """Split features into train and validation sets."""
    X = df[['Day_Index', 'Arrival_Hour']]
    y = df['Waiting_Time_Mins']
    
    print(f"[INFO] Splitting data (Test size: {config.test_size})...")
    return train_test_split(
        X, y, 
        test_size=config.test_size, 
        random_state=config.random_state
    )

def train_baseline_model(X_train, y_train, config: PipelineConfig):
    """Fit the XGBoost regressor model."""
    print("[INFO] Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=config.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on validation data."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"[RESULT] Model Performance - Mean Absolute Error: {mae:.2f} minutes")
    return {"mae": mae}

def save_artifacts(config: PipelineConfig, model, metrics):
    """Persist model and metrics artifacts to disk."""
    os.makedirs(config.model_dir, exist_ok=True)
    model_path = config.model_dir / config.model_filename
    metrics_path = config.model_dir / config.metrics_filename

    model.save_model(str(model_path))
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"[SUCCESS] Model saved to: {model_path}")
    print(f"[SUCCESS] Metrics saved to: {metrics_path}")
    return {"model_path": str(model_path), "metrics_path": str(metrics_path)}

@dataclass
class DefaultModelTrainer:
    """Orchestrator class for the training pipeline."""
    # FIX: Usiamo default_factory per evitare l'errore sui valori mutabili
    config: PipelineConfig = field(default_factory=lambda: default_config(Path(".")))

    def train_model(self):
        """Execute the full training pipeline."""
        try:
            # 1. Load Data
            df = load_feature_table(self.config)
            
            # 2. Split Data
            X_train, X_test, y_train, y_test = split_train_val(df, self.config)
            
            # 3. Train Model
            model = train_baseline_model(X_train, y_train, self.config)

            # 4. Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            
            # 5. Save
            save_artifacts(self.config, model, metrics)

            # 6. logging the process
            logger.info("Training pipeline completed successfully with metrics: %s", metrics)
            
        except Exception:
            logger.exception("Training pipeline failed")
            raise           

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    trainer = DefaultModelTrainer()
    trainer.train_model()
