import pandas as pd
import pm4py
import os
from pm4py.algo.filtering.dfg import dfg_filtering 

PROCESSED_DATA_PATH = "data/processed/patient_journey_log.csv"
OUTPUT_DIR = "reports/figures"
OUTPUT_IMG_PATH = "reports/figures/patient_journey_dfg.png"


def discover_process():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

    # Deleting rare activities
    counter = df['concept:name'].value_counts()
    df = df[df['concept:name'].isin(counter[counter >= 20].index)]

    # Grouping similar activities
    df['concept:name'] = df['concept:name'].str.lower()
    df['concept:name'] = df['concept:name'].replace(r'depression|anxiety|suicide|behavioral', 'Mental health intervention',regex=True)
    df['concept:name'] = df['concept:name'].replace(r'x-ray|ct scan|mri|mammography|ultrasound', 'Imaging tests',regex=True)
    df['concept:name'] = df['concept:name'].replace(r'chlamydia|gonorrhea|hiv|hepatitis|syphilis', 'STD tests',regex=True)
    df['concept:name'] = df['concept:name'].replace(r'pregnancy|prenatal|fetal|mother|childbirth|labor', 'Pregnancy and childbirth',regex=True)
    df['concept:name'] = df['concept:name'].replace(r'vaccination|immunization|immunotherapy', 'Vaccination/Immunization', regex=True)

    # Graph Discovery 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dfg, start_activities, end_activities = pm4py.discover_dfg(df)

    # Filtering for more frequent activities 
    activities_count = df['concept:name'].value_counts().to_dict() 
    dfg_filtered, start_f, end_f, activities_f = dfg_filtering.filter_dfg_on_activities_percentage(
    dfg, start_activities, end_activities, activities_count, percentage=0.6)

    # Filtering for infrequent paths
    dfg_edges_filtered, start_edges, end_edges,activities_edges = dfg_filtering.filter_dfg_on_paths_percentage(
    dfg_filtered, start_f, end_f, activities_count, percentage= 0.3, keep_all_activities= False)

    #Saving Visualization
    os.makedirs(os.path.dirname(OUTPUT_IMG_PATH), exist_ok=True)
    pm4py.save_vis_dfg(dfg_edges_filtered, start_edges, end_edges, OUTPUT_IMG_PATH, variant = "frequency")


if __name__ == "__main__":
    discover_process()