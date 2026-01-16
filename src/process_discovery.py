import os
import re

import pandas as pd
import pm4py
from graphviz import Source
from pm4py.visualization.dfg import visualizer as dfg_visualizer

PROCESSED_DATA_PATH = "data/processed/patient_journey_log.csv"
OUTPUT_DIR = "reports/figures"
OUTPUT_IMG_PATH = "reports/figures/patient_journey_dfg.png"
OUTPUT_IMG_PATH_TIME = "reports/figures/patient_journey_dfg_time.png"

def discover_process():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

    # Grouping similar activities
    s = df['concept:name'].astype(str).str.strip().str.lower()

    def _collapse(rule_regex: str, label: str) -> None:
        nonlocal s
        s = s.replace(rule_regex, label, regex=True)
        mask = s.str.contains(label, regex=False)
        s.loc[mask] = label

    rules = [
        (
            r'pregnan|prenatal|amniotic|fetal|uterine fundal|childbirth|pregnancy test',
            'pregnancy & fetal care',
        ),
        (
            r'depression|patient health questionnaire|phq[- ]?(2|9)?|anxiety|mental health|cognitive and behavioral therapy',
            'mental/behavioral health',
        ),
        (
            r'substance use|drug abuse|alcohol use disorders identification test|audit[- ]?c',
            'substance/abuse screening',
        ),
        (
            r'renal dialysis|hemodialysis|haemodialysis',
            'renal/dialysis',
        ),
        (
            r'chemotherapy|radiation therapy|teleradiotherapy',
            'oncology therapy',
        ),
        (
            r'electrical cardioversion|electrocardiographic|ecg|echocardiography',
            'cardiology procedures/tests',
        ),
        (
            r'\bx-ray\b|radiograph|mammography|bone density scan|computed tomography|magnetic resonance|ultrasound',
            'imaging tests',
        ),
        (
            r'hemoglobin|hematocrit|platelet count|cytopathology|smear|chlamydia|gonorrhea|syphilis|hepatitis|human immunodeficiency virus|\bhiv\b',
            'lab tests / std panel',
        ),
    ]

    for rule, label in rules:
        _collapse(rule, label)

    df['concept:name'] = s

    case_sizes = df.groupby('case:concept:name').size()
    df = df[df['case:concept:name'].isin(case_sizes[case_sizes >= 2].index)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dfg, start_activities, end_activities = pm4py.discover_dfg(df)

    os.makedirs(os.path.dirname(OUTPUT_IMG_PATH), exist_ok=True)
    pm4py.save_vis_dfg(
        dfg,
        start_activities,
        end_activities,
        OUTPUT_IMG_PATH,
        variant="frequency",
    )

    df["start:timestamp"] = pd.to_datetime(df["start:timestamp"], utc=True)
    df["end:timestamp"] = pd.to_datetime(df["end:timestamp"], utc=True)

    perf_dfg, sa, ea = pm4py.discover_performance_dfg(
        df,
        case_id_key="case:concept:name",
        activity_key="concept:name",
        timestamp_key="start:timestamp",
        perf_aggregation_key="mean",
        business_hours=False,
    )

    os.makedirs(os.path.dirname(OUTPUT_IMG_PATH_TIME), exist_ok=True)

    gviz = dfg_visualizer.apply(
        perf_dfg,
        variant=dfg_visualizer.Variants.PERFORMANCE,
        parameters={"start_activities": sa, "end_activities": ea},
    )

    def _edge_to_minutes(m: re.Match) -> str:
        attr = m.group("attr")   
        val = float(m.group("val"))
        unit = m.group("unit").lower()

        if unit == "s":
            minutes = val / 60
        elif unit == "m":
            minutes = val
        elif unit == "h":
            minutes = val * 60
        elif unit == "d":
            minutes = val * 1440
        else:
            return m.group(0)

        return f'{attr}="{minutes} min"'

    dot = re.sub(
        r'(?P<attr>(?:x?label|headlabel|taillabel))\s*=\s*(?P<q>"?)(?P<val>\d+(?:\.\d+)?)(?:\s*)(?P<unit>[smhd])(?P=q)',
        _edge_to_minutes,
        gviz.source,
        flags=re.IGNORECASE,
    )
 
    Source(dot).render(filename=os.path.splitext(OUTPUT_IMG_PATH_TIME)[0], format="png", cleanup=True)

if __name__ == "__main__":
    discover_process()