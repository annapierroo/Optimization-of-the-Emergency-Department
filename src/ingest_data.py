import pandas as pd
import sys
import os

RAW_DATA_PATH = "data/raw/EventLog.csv"
PROCESSED_DATA_PATH = "data/processed/patient_journey_log.csv"

def ingest_and_clean():
    
    try:
        # Read CSV
       cols_to_use = ['PATIENT', 'START', 'STOP', 'DESCRIPTION']
       df = pd.read_csv(RAW_DATA_PATH, sep=',', usecols=cols_to_use)
    except FileNotFoundError:
        print(f"ERROR: File {RAW_DATA_PATH} not found.")
        sys.exit(1)
    
    # Convert both START and STOP 
    df['START'] = pd.to_datetime(df['START'], format="ISO8601", utc=True)
    df['STOP'] = pd.to_datetime(df['STOP'], format="ISO8601", utc=True)
    df.sort_values(by=['PATIENT', 'START'], inplace=True)


    n_patients = 200
    top_patients = df['PATIENT'].unique()[:n_patients]
    df = df[df['PATIENT'].isin(top_patients)].copy()

    # Column Mapping for standard PM4Py
    rename_mapping = {
        'PATIENT': 'case:concept:name',
        'DESCRIPTION': 'concept:name',
        'STOP': 'time:timestamp'
    }
    df.rename(columns=rename_mapping, inplace=True)

    df.dropna(subset=['case:concept:name', 'time:timestamp'], inplace=True)
   
    # HANDLING MISSING ACTIVITIES
    df['concept:name'] = df['concept:name'].fillna("UNKNOWN_ACTIVITY")

    # HANDLING ATTRIBUTES (Costs and Reasons)
    if 'BASE_COST' in df.columns:
        df['BASE_COST'] = df['BASE_COST'].fillna(0)

    if 'REASONDESCRIPTION' in df.columns:
        df['REASONDESCRIPTION'] = df['REASONDESCRIPTION'].fillna("Not specified")


    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Ready dataset saved to: {PROCESSED_DATA_PATH}")
   
if __name__ == "__main__":
    ingest_and_clean()