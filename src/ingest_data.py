import pandas as pd
import sys
import os

RAW_DATA_PATH = "data/raw/EventLog.csv"
PROCESSED_DATA_PATH = "data/processed/patient_journey_log.csv"

def ingest_and_clean():

    try:
        cols_to_use = ['ENCOUNTER', 'START', 'STOP', 'DESCRIPTION']
        df = pd.read_csv(RAW_DATA_PATH, sep=';', usecols=cols_to_use)
    except FileNotFoundError:
        print(f"ERROR: File {RAW_DATA_PATH} not found.")
        sys.exit(1)

    df['START'] = pd.to_datetime(df['START'], utc=True, errors='coerce')
    df['STOP'] = pd.to_datetime(df['STOP'], utc=True, errors='coerce')

    df['DESCRIPTION'] = df['DESCRIPTION'].fillna('UNKNOWN_ACTIVITY')

    df.dropna(subset=['ENCOUNTER', 'START', 'STOP', 'DESCRIPTION'], inplace=True)

    n_patients = 50 # adjust as needed
    encounter_counts = df['ENCOUNTER'].unique()[:n_patients]
    df = df[df['ENCOUNTER'].isin(encounter_counts)].copy()

    df.sort_values(by=['ENCOUNTER', 'START', 'STOP'], inplace=True)

    # Column Mapping for standard PM4Py
    rename_mapping = {
        'ENCOUNTER': 'case:concept:name',
        'DESCRIPTION': 'concept:name',
        'START': 'start:timestamp',
        'STOP': 'end:timestamp'
    }
    df.rename(columns=rename_mapping, inplace=True)
    df['time:timestamp'] = df['start:timestamp']

    df.dropna(subset=['case:concept:name', 'concept:name', 'start:timestamp', 'end:timestamp', 'time:timestamp'], inplace=True)
    df = df.sort_values(by=["case:concept:name", "start:timestamp"])

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Ready dataset saved to: {PROCESSED_DATA_PATH}")
   
if __name__ == "__main__":
    ingest_and_clean()