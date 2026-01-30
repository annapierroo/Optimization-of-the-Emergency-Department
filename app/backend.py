"""
Backend utilities powering Streamlit predictions.
Handles model loading, input preprocessing, and inference.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to system path to allow importing from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import default_config

def load_latest_model(model_dir=None):
    """
    Loads the most recent model artifact from disk.
    
    Args:
        model_dir (str, optional): Custom path to model directory. 
                                   Defaults to project config path.
    
    Returns:
        model: Deserialized sklearn model object.
    """
    # Use default config if no path is provided
    if model_dir is None:
        config = default_config(PROJECT_ROOT)
        model_path = config.model_dir / "model.joblib"
    else:
        model_path = Path(model_dir) / "model.joblib"

    if not model_path.exists():
        # Fallback to avoid crash if training hasn't run yet
        raise FileNotFoundError(f"🚨 Model not found at {model_path}. Did you run 'python src/training.py'?")
    
    print(f"Loading model from: {model_path}")
    return joblib.load(model_path)


def preprocess_procedures(procedures, feature_columns):
    """
    Converts a list of raw procedure names into a model-ready feature vector.
    
    Args:
        procedures (list): List of procedure strings (e.g., ['Triage', 'X-Ray']).
        feature_columns (list): List of all columns expected by the trained model.
        
    Returns:
        pd.DataFrame: A single-row DataFrame ready for inference.
    """
    # 1. Initialize a dictionary with all expected features set to 0
    input_data = {col: 0 for col in feature_columns}
    
    # 2. Set the selected procedures to 1 (One-Hot Encoding simulation)
    for proc in procedures:
        # Note: The prefix must match the one defined in src/features.py
        feat_name = f"proc_count__{proc}" 
        
        # Only activate the feature if it exists in the model's training set
        if feat_name in input_data:
            input_data[feat_name] = 1
            
    # 3. Return as a DataFrame
    return pd.DataFrame([input_data])


def predict_duration(model, feature_row):
    """
    Runs model inference on the preprocessed data.
    
    Args:
        model: Trained model object.
        feature_row (pd.DataFrame): The preprocessed input row.
        
    Returns:
        float: Predicted duration in minutes (rounded to 2 decimals).
    """
    # Run prediction
    prediction = model.predict(feature_row)
    
    # Return rounded value
    return np.round(prediction[0], 2)
