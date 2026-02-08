import pandas as pd
import sys
import os
import warnings
from time_comp import Timer

RAW_DATA_PATH = "data/raw/EventLog.csv"
PROCESSED_DATA_PATH = "data/processed/patient_journey_log.csv"

def ingest_and_clean():
    timer = Timer()
    timer.start("Data Ingestion")
    try:
        cols_to_use = ['ENCOUNTER', 'START', 'STOP', 'DESCRIPTION']
        df = pd.read_csv(RAW_DATA_PATH, sep=';', usecols=cols_to_use)
    except FileNotFoundError:
        print(f"ERROR: File {RAW_DATA_PATH} not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: One or more columns in the CSV file are missing or have incorrect data types: {e}")
        sys.exit(1)

    missing_data = df.isnull().sum()
    if missing_data.any():
       warnings.warn(f"Missing data detected:\n{missing_data[missing_data > 0]}")

    df['START'] = pd.to_datetime(df['START'], utc=True, errors='coerce')
    df['STOP'] = pd.to_datetime(df['STOP'], utc=True, errors='coerce')
    df['DESCRIPTION'] = df['DESCRIPTION'].fillna('UNKNOWN_ACTIVITY')
    timer.end("Data Ingestion")

    timer.start("Cleaning missing data")
    dataframe_length = len(df)
    df.dropna(subset=['START', 'STOP', 'DESCRIPTION'], inplace=True)
    dropped_rows = dataframe_length - len(df)
    if dropped_rows > 0:
        warnings.warn(f"Dropped {dropped_rows} rows due to insufficient data.")
    timer.end("Cleaning missing data")

    timer.start("Filtering and sorting")
    n_patients = 200 # adjust as needed
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

    df = df.sort_values(by=["case:concept:name", "start:timestamp"])
    timer.end("Filtering and sorting")

    timer.start("Data quality checks")
    inverted_time_events = (df['end:timestamp'] < df['start:timestamp']).sum()
    if inverted_time_events > 0:
        warnings.warn(f"Found {inverted_time_events} events where end time is before start time. This may indicate data quality issues.")

    durations = df['end:timestamp'] - df['start:timestamp']
    time_zero = (durations.dt.total_seconds() == 0)
    zero_time_events = time_zero.sum()
    if zero_time_events > 0:
        warnings.warn(f"Found {zero_time_events} events with zero duration. Deleting them.")
        df = df[~time_zero]

    df['previous_end'] = df.groupby('case:concept:name')['end:timestamp'].shift(1)
    overlapping_events = (df['start:timestamp'] < df['previous_end']) & (df['previous_end'].notna())
    n_overlaps = overlapping_events.sum()
    if n_overlaps > 0:
        warnings.warn(f"Found {n_overlaps} overlapping events for the same patient. This may indicate data quality issues.")

    average_duration = durations.mean()
    if average_duration.total_seconds() < 1:
        warnings.warn("Average duration of events is almost zero. This may indicate an issue with the timestamps.")
    timer.end("Data quality checks")

    timer.start("Saving processed dataset")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Ready dataset saved to: {PROCESSED_DATA_PATH}")
    timer.end("Saving processed dataset")

    total_time = timer.total()
    timer.summary()
    if total_time > 5 :
        warnings.warn(f"WARNING: Process discovery took {total_time:.2f}s, which exceeds the maximum threshold of {MAX_TIME_SECONDS}s.")
    else:
        print(f"Process discovery completed in {total_time:.2f}s.")


if __name__ == "__main__":
    ingest_and_clean()