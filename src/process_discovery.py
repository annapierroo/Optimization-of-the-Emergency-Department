import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import pm4py
import holidays
from graphviz import Source
from pm4py.visualization.dfg import visualizer as dfg_visualizer

PROCESSED_DATA_PATH = "data/processed/patient_journey_log.csv"
OUTPUT_DIR = "reports/figures"
OUTPUT_IMG_PATH = "reports/figures/patient_journey_dfg.png"

WAITING_CSV = "reports/waiting_transitions.csv"
WAITING_BOXPLOT_WEEKEND = "reports/figures/waiting_weekday_vs_weekend_boxplot.png"
WAITING_BOXPLOT_DAYTYPE = "reports/figures/waiting_holiday_weekend_weekday_boxplot.png"
WAITING_ECDF_WEEKEND = "reports/figures/waiting_weekday_vs_weekend_ecdf.png"
WAITING_ECDF_DAYTYPE = "reports/figures/waiting_holiday_weekend_weekday_ecdf.png"
WAITING_BOXPLOT_DAYNIGHT = "reports/figures/waiting_day_vs_night_boxplot.png"
WAITING_ECDF_DAYNIGHT = "reports/figures/waiting_day_vs_night_ecdf.png"
OUTPUT_IMG_PATH_TIME = "reports/figures/patient_journey_dfg_time.png"

# Boxplot on log1p scale 
def plot_boxplot_log1p(groups: dict, title: str, output_path: str):
    labels = []
    data = []
    for name, vals in groups.items():
        v = pd.Series(vals).dropna().astype(float)
        v = v[v > 0]
        if len(v) == 0:
            continue
        labels.append(f"{name}\n(n={len(v)})")
        data.append((v + 1.0).apply(lambda x: float(x)).apply(lambda x: __import__('math').log(x)))

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.title(title + "\n(y-axis = log(1 + minutes))")
    plt.ylabel("log(1 + minutes)")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


# ECDF with log-x 
def plot_ecdf_minutes(groups: dict, title: str, output_path: str):
    plt.figure(figsize=(9, 5))

    for name, vals in groups.items():
        v = pd.Series(vals).dropna().astype(float)
        v = v[v > 0]
        if len(v) == 0:
            continue
        v = v.sort_values()
        y = (pd.RangeIndex(1, len(v) + 1) / len(v)).to_numpy()
        plt.step(v.to_numpy(), y, where="post", label=f"{name} (n={len(v)})")

    plt.xscale("log")
    plt.title(title + "\n(x-axis = minutes, log scale)")
    plt.xlabel("Minutes (log scale)")
    plt.ylabel("ECDF")  #ECDF = Empirical Cumulative Distribution Function
    plt.grid(axis="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

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

    # Waiting time
    df = df.sort_values(["case:concept:name", "start:timestamp", "end:timestamp"]).copy()

    df["next_start"] = df.groupby("case:concept:name")["start:timestamp"].shift(-1)
    df["next_activity"] = df.groupby("case:concept:name")["concept:name"].shift(-1)

    df["raw_gap_min"] = ((df["next_start"] - df["end:timestamp"]).dt.total_seconds() / 60.0).round(2)
    df["waiting_min"] = df["raw_gap_min"].clip(lower=0).round(2)
   


    perf_dfg, sa, ea = pm4py.discover_performance_dfg(
        df,
        case_id_key="case:concept:name",
        activity_key="concept:name",
        timestamp_key="start:timestamp",
        perf_aggregation_key="mean",
        business_hours=False,
    )

    os.makedirs(os.path.dirname(OUTPUT_IMG_PATH_TIME), exist_ok=True)

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

        return f'{attr}="{minutes:.2f} min"'

    dot = re.sub(
        r'(?P<attr>(?:x?label|headlabel|taillabel))\s*=\s*(?P<q>"?)(?P<val>\d+(?:\.\d+)?)(?:\s*)(?P<unit>[smhd])(?P=q)',
        _edge_to_minutes,
        gviz.source,
        flags=re.IGNORECASE,
    )
 
    Source(dot).render(filename=os.path.splitext(OUTPUT_IMG_PATH_TIME)[0], format="png", cleanup=True) 
  

    print(f"Saved performance DFG to: {OUTPUT_IMG_PATH_TIME}")
    trans = df.dropna(subset=["next_activity", "raw_gap_min"]).copy()

    years = df["start:timestamp"].dt.year.dropna().unique().tolist()
    ma_holidays = holidays.country_holidays("US", subdiv="MA", years=years)

    trans["date"] = trans["start:timestamp"].dt.date
    trans["is_holiday"] = trans["date"].apply(lambda d: d in ma_holidays)
    trans["is_weekend"] = trans["start:timestamp"].dt.weekday >= 5
    trans["day_type"] = trans.apply(lambda r: "holiday" if r["is_holiday"] else ("weekend" if r["is_weekend"] else "weekday"), axis=1)
    # Day/Night based on start time of the current activity (6-18 = day)
    trans["hour"] = trans["start:timestamp"].dt.hour
    trans["time_of_day"] = trans["hour"].apply(lambda h: "day" if 6 <= h < 18 else "night")

    
    os.makedirs(os.path.dirname(WAITING_CSV), exist_ok=True)
    trans[[
        "case:concept:name",
        "concept:name",
        "next_activity",
        "start:timestamp",
        "end:timestamp",
        "next_start",
        "raw_gap_min",
        "waiting_min",
        "date",
        "is_holiday",
        "is_weekend",
        "day_type",
    ]].to_csv(WAITING_CSV, index=False)

    # Day vs Night analysis
    daynight_groups = {
        "day": trans[trans["time_of_day"] == "day"]["waiting_min"],
        "night": trans[trans["time_of_day"] == "night"]["waiting_min"],
    }

    plot_boxplot_log1p(daynight_groups, "Waiting time by Day vs Night (minutes)", WAITING_BOXPLOT_DAYNIGHT)
    plot_ecdf_minutes(daynight_groups, "Waiting time by Day vs Night (minutes)", WAITING_ECDF_DAYNIGHT)

    def _summary(series: pd.Series) -> dict:
        s = pd.Series(series).dropna().astype(float)
        s = s[s > 0]
        if len(s) == 0:
            return {"n": 0, "median": None, "p90": None, "p95": None, "p99": None}
        return {
            "n": int(len(s)),
            "median": round(float(s.median()), 2),
            "p90": round(float(s.quantile(0.90)), 2),
            "p95": round(float(s.quantile(0.95)), 2),
            "p99": round(float(s.quantile(0.99)), 2),
        }

    summary_dn = pd.DataFrame({k: _summary(v) for k, v in daynight_groups.items()}).T
    print("\nWaiting time summary (minutes, waiting>0) - Day vs Night")
    print(summary_dn)

    # Plots to compare waiting times
    weekend_groups = {
        "weekday": trans[~trans["is_weekend"]]["waiting_min"],
        "weekend": trans[trans["is_weekend"]]["waiting_min"],
    }
    daytype_groups = {
        "holiday": trans[trans["day_type"] == "holiday"]["waiting_min"],
        "weekend": trans[trans["day_type"] == "weekend"]["waiting_min"],
        "weekday": trans[trans["day_type"] == "weekday"]["waiting_min"],
    }

    plot_boxplot_log1p(weekend_groups, "Waiting time by Weekday vs Weekend (minutes)", WAITING_BOXPLOT_WEEKEND)
    plot_ecdf_minutes(weekend_groups, "Waiting time by Weekday vs Weekend (minutes)", WAITING_ECDF_WEEKEND)

    plot_boxplot_log1p(daytype_groups, "Waiting time by Holiday/Weekend/Weekday (minutes)", WAITING_BOXPLOT_DAYTYPE)
    plot_ecdf_minutes(daytype_groups, "Waiting time by Holiday/Weekend/Weekday (minutes)", WAITING_ECDF_DAYTYPE)

    print(f"Saved waiting transitions to: {WAITING_CSV}")
    print(f"Saved plots to: {WAITING_BOXPLOT_WEEKEND}, {WAITING_ECDF_WEEKEND}, {WAITING_BOXPLOT_DAYTYPE}, {WAITING_ECDF_DAYTYPE}")

if __name__ == "__main__":
    discover_process()