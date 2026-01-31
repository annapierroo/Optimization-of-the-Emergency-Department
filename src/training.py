"""
Model training module.
Implements the training pipeline structure defined by the team, 
filling the logic to train and persist the XGBoost model.
"""

import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass, field  # <--- AGGIUNTO 'field'

# Configuration Class
@dataclass
class PipelineConfig:
    data_path: str = "data/raw/EventLog.csv"
    model_dir: str = "models"
    model_filename: str = "xgb_model.json"
    test_size: float = 0.2
    random_state: int = 42

def load_feature_table(config: PipelineConfig) -> pd.DataFrame:
    """Load and preprocess the dataset from disk."""
    print(f"[INFO] Loading data from {config.data_path}...")
    
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"Data file not found at {config.data_path}")

    # Load Data
    df = pd.read_csv(config.data_path, sep=";")
    
    # Cleaning & Preprocessing
    df.columns = df.columns.str.strip()
    df['START'] = pd.to_datetime(df['START'], utc=True, errors='coerce')
    df['STOP'] = pd.to_datetime(df['STOP'], utc=True, errors='coerce')
    df = df.dropna(subset=['START', 'STOP'])

    # Target Calculation
    df['Waiting_Time_Mins'] = (df['STOP'] - df['START']).dt.total_seconds() / 60
    df = df[df['Waiting_Time_Mins'] >= 0]

    # Feature Engineering
    df['Day_Index'] = df['START'].dt.dayofweek
    df['Arrival_Hour'] = df['START'].dt.hour
    
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

def save_artifacts(config: PipelineConfig, model):
    """Persist model artifacts to disk."""
    os.makedirs(config.model_dir, exist_ok=True)
    save_path = os.path.join(config.model_dir, config.model_filename)
    
    model.save_model(save_path)
    print(f"[SUCCESS] Model saved to: {save_path}")

@dataclass
class DefaultModelTrainer:
    """Orchestrator class for the training pipeline."""
    # FIX: Usiamo default_factory per evitare l'errore sui valori mutabili
    config: PipelineConfig = field(default_factory=PipelineConfig)

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
            evaluate_model(model, X_test, y_test)
            
            # 5. Save
            save_artifacts(self.config, model)
            
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    trainer = DefaultModelTrainer()
    trainer.train_model()
