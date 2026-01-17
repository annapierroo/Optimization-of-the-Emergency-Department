import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import pm4py
from graphviz import Source
from pm4py.visualization.dfg import visualizer as dfg_visualizer

PROCESSED_DATA_PATH = "data/processed/patient_journey_log.csv"
OUTPUT_DIR = "reports/figures"
BOXPLOT_PATH = "reports/figures/waiting_time_boxplot.png"
OUTPUT_IMG_PATH = "reports/figures/patient_journey_dfg.png"
OUTPUT_IMG_PATH_TIME = "reports/figures/patient_journey_dfg_time.png"

def discover_process():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
    df["start:timestamp"] = pd.to_datetime(df["start:timestamp"], utc=True)
    df["end:timestamp"] = pd.to_datetime(df["end:timestamp"], utc=True)

    # Manually computes the waiting time to generate a boxplot to look for outliers
    df["waiting_time"] = (
    df.groupby("case:concept:name")["start:timestamp"]
      .shift(-1) - df["end:timestamp"])
    df["next_activity"] = df.groupby("case:concept:name")["concept:name"].shift(-1)
    df["waiting_time"] = df["waiting_time"].dt.total_seconds() / 3600.0 #in hours
    df_plot = df.dropna(subset=["waiting_time"])
    plt.figure()
    plt.boxplot(df_plot["waiting_time"], vert=True, showfliers=True)
    plt.title("Boxplot of Waiting Times (in hours)")
    plt.ylabel("Waiting Time")
    plt.savefig(BOXPLOT_PATH)
    plt.close()    

    # Analysis of outliers
    outliers = df[df["waiting_time"] > df["waiting_time"].quantile(0.95)]
    outliers_infos = outliers[['case:concept:name','concept:name','next_activity','waiting_time']]

   # Activities that are typically isolated/the first
    first_activities = df.sort_values(['case:concept:name', 'start:timestamp']) \
                     .groupby('case:concept:name').first()

    first_counts = first_activities['concept:name'].value_counts()
    total_counts = df.groupby('concept:name')['case:concept:name'].nunique()
    isolated_acts = (df.sort_values(['case:concept:name', 'start:timestamp'])
                   .groupby('case:concept:name').first()['concept:name']
                   .value_counts() / df.groupby('concept:name')['case:concept:name'].nunique()
                    ).loc[lambda x: x > 0.95].index.tolist()

    

    mean_before_next = df.groupby("next_activity")["waiting_time"].mean().reset_index()
    mean_after_current = df.groupby("concept:name")["waiting_time"].mean().reset_index()
    outliers = outliers.merge(mean_before_next, on = "next_activity", how = "left", suffixes=('', '_mean_before_next'))
    outliers = outliers.merge(mean_after_current, on="concept:name", how="left", suffixes=('', '_mean_after_current'))
    

    real_outliers = outliers[ 
        ((outliers["next_activity"].isin(isolated_acts)) &
        ((outliers["waiting_time"] > 2 * outliers["waiting_time_mean_before_next"]) |
        (outliers["waiting_time"] > 2 * outliers["waiting_time_mean_after_current"]))) |
        ((outliers["waiting_time"] > 2 * outliers["waiting_time_mean_before_next"]) &
        (outliers["waiting_time"] > 2 * outliers["waiting_time_mean_after_current"]))]

    df = df[~df.index.isin(real_outliers.index)]

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