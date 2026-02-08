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
ENCODER_PATH = os.path.join(MODEL_DIR, "activity_encoder.pkl")

def train_next_activity():
    """
    Trains and persists an XGBoost Classifier for Next Activity Prediction.
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
    # Create target: Shift 'DESCRIPTION' up by 1 to get the next event
    df['Next_Activity'] = df.groupby('ENCOUNTER')['DESCRIPTION'].shift(-1)
    df = df.dropna(subset=['Next_Activity'])

    # Temporal Features
    df['Hour'] = df['START'].dt.hour
    df['Day_of_Week'] = df['START'].dt.dayofweek 
    
    # 4. Encoding
    le = LabelEncoder()
    # Fit on all unique activities (source + target) to ensure consistent mapping
    all_activities = pd.concat([df['DESCRIPTION'], df['Next_Activity']]).unique()
    le.fit(all_activities)
    
    df['Current_Activity_Encoded'] = le.transform(df['DESCRIPTION'])
    df['Next_Activity_Encoded'] = le.transform(df['Next_Activity'])

    # Split Data
    features = ['Current_Activity_Encoded', 'Hour', 'Day_of_Week']
    X = df[features]
    y = df['Next_Activity_Encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Training
    print(f"Training XGBoost on {len(df)} samples...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softprob', 
        num_class=len(le.classes_),
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)

    # 6. Evaluation
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {acc:.2%}")

    # 7. Save Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    
    print(f"Success. Model and encoder saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_next_activity()
