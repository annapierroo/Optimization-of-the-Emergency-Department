import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Configuration
RAW_DATA_PATH = "data/raw/EventLog.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "next_activity_xgb.json")
INPUT_ENCODER_PATH = os.path.join(MODEL_DIR, "input_encoder.pkl")
OUTPUT_ENCODER_PATH = os.path.join(MODEL_DIR, "output_encoder.pkl")

def train_next_activity():
    """
    Trains an XGBoost Classifier.
    FIX 2.0: Removes rare classes (< 5 occurrences) to prevent gaps in target labels
    and uses stratified splitting.
    """
    print("--- Starting Next Activity Prediction Pipeline ---")

    # 1. Load Data
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} not found.")
        return

    df = pd.read_csv(RAW_DATA_PATH, sep=";")
    
    # 2. Data Cleaning
    df.columns = df.columns.str.strip()
    df['START'] = pd.to_datetime(df['START'], utc=True, errors='coerce')
    df = df.dropna(subset=['START', 'DESCRIPTION', 'ENCOUNTER'])
    df = df.sort_values(by=['ENCOUNTER', 'START'])

    # 3. Feature Engineering
    # Target: Shift 'DESCRIPTION' up by 1
    df['Next_Activity'] = df.groupby('ENCOUNTER')['DESCRIPTION'].shift(-1)
    df = df.dropna(subset=['Next_Activity'])

    # --- FIX START: Remove Rare Classes ---
    # XGBoost crashes if a class is in the Test set but missing in Train set (creating a gap).
    # We remove activities that appear fewer than 5 times to ensure stability.
    print("Filtering rare activities...")
    activity_counts = df['Next_Activity'].value_counts()
    common_activities = activity_counts[activity_counts >= 5].index
    df = df[df['Next_Activity'].isin(common_activities)]
    print(f"   -> Retained {len(common_activities)} unique activities types.")
    # --- FIX END ---

    # Temporal Features
    df['Hour'] = df['START'].dt.hour
    df['Day_of_Week'] = df['START'].dt.dayofweek 
    
    # 4. Encoding
    print("Encoding features and targets...")
    
    le_input = LabelEncoder()
    le_input.fit(df['DESCRIPTION'])
    df['Current_Activity_Encoded'] = le_input.transform(df['DESCRIPTION'])
    
    le_output = LabelEncoder()
    le_output.fit(df['Next_Activity'])
    df['Next_Activity_Encoded'] = le_output.transform(df['Next_Activity'])

    # Split Data
    features = ['Current_Activity_Encoded', 'Hour', 'Day_of_Week']
    X = df[features]
    y = df['Next_Activity_Encoded']

    # Stratify=y ensures that every class in Train is also in Test (no holes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Model Training
    print(f"Training XGBoost on {len(df)} samples...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )
    
    model.fit(X_train, y_train)

    # 6. Evaluation
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {acc:.2%}")

    # 7. Save Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    joblib.dump(le_input, INPUT_ENCODER_PATH)
    joblib.dump(le_output, OUTPUT_ENCODER_PATH)
    
    print(f"Success! Model and encoders saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_next_activity()