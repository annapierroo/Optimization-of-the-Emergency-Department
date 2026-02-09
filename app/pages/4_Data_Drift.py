"""Admin-only data drift dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import auth
from app import backend


def _psi_numeric(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    baseline = baseline.dropna().astype(float)
    current = current.dropna().astype(float)
    if baseline.empty or current.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(baseline, quantiles))
    if len(cuts) < 3:
        return 0.0

    baseline_binned = pd.cut(baseline, bins=cuts, include_lowest=True)
    current_binned = pd.cut(current, bins=cuts, include_lowest=True)
    base_freq = baseline_binned.value_counts(normalize=True).sort_index()
    curr_freq = current_binned.value_counts(normalize=True).sort_index()
    aligned = pd.concat([base_freq, curr_freq], axis=1).fillna(0.0001)
    aligned.columns = ["base", "curr"]
    psi = ((aligned["curr"] - aligned["base"]) * np.log(aligned["curr"] / aligned["base"])).sum()
    return float(psi)


def _js_divergence_from_counts(base_counts: pd.Series, curr_counts: pd.Series) -> float:
    labels = sorted(set(base_counts.index) | set(curr_counts.index))
    if not labels:
        return 0.0
    base = np.array([float(base_counts.get(label, 0.0)) for label in labels], dtype=float)
    curr = np.array([float(curr_counts.get(label, 0.0)) for label in labels], dtype=float)
    if base.sum() == 0 or curr.sum() == 0:
        return 0.0
    p = base / base.sum()
    q = curr / curr.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _severity_from_psi(value: float) -> str:
    if value > 0.25:
        return "high"
    if value > 0.1:
        return "medium"
    return "low"


def _severity_from_js(value: float) -> str:
    if value > 0.2:
        return "high"
    if value > 0.1:
        return "medium"
    return "low"


def _render_wait_time_drift(baseline: pd.DataFrame, logs: pd.DataFrame):
    st.subheader("Wait-Time Drift")
    wt = logs[logs["model_name"] == "wait_time"].copy()
    if wt.empty:
        st.info("No wait-time prediction logs yet.")
        return []

    hour_col = "input__Arrival_Hour"
    day_col = "input__Day_Index"
    alerts = []

    if hour_col in wt.columns:
        psi_hour = _psi_numeric(baseline["Arrival_Hour"], wt[hour_col])
        st.write(f"PSI Arrival_Hour: `{psi_hour:.4f}`")
        alerts.append({"model": "wait_time", "metric": "PSI Arrival_Hour", "value": psi_hour, "severity": _severity_from_psi(psi_hour)})
    if day_col in wt.columns:
        psi_day = _psi_numeric(baseline["Day_Index"], wt[day_col])
        st.write(f"PSI Day_Index: `{psi_day:.4f}`")
        alerts.append({"model": "wait_time", "metric": "PSI Day_Index", "value": psi_day, "severity": _severity_from_psi(psi_day)})

    st.caption(f"Logged predictions: {len(wt)}")
    return alerts


def _render_los_drift(baseline: pd.DataFrame, logs: pd.DataFrame):
    st.subheader("Lenght of Stay Drift")
    los = logs[logs["model_name"] == "los"].copy()
    if los.empty:
        st.info("No LOS prediction logs yet.")
        return []

    alerts = []
    code_col = "input__CODE"
    cost_col = "input__BASE_COST"

    if code_col in los.columns and "CODE" in baseline.columns:
        js_code = _js_divergence_from_counts(baseline["CODE"].value_counts(), los[code_col].value_counts())
        st.write(f"JS CODE distribution: `{js_code:.4f}`")
        alerts.append({"model": "los", "metric": "JS CODE", "value": js_code, "severity": _severity_from_js(js_code)})
    if cost_col in los.columns and "BASE_COST" in baseline.columns:
        psi_cost = _psi_numeric(baseline["BASE_COST"], los[cost_col])
        st.write(f"PSI BASE_COST: `{psi_cost:.4f}`")
        alerts.append({"model": "los", "metric": "PSI BASE_COST", "value": psi_cost, "severity": _severity_from_psi(psi_cost)})

    st.caption(f"Logged predictions: {len(los)}")
    return alerts


def _render_next_activity_drift(baseline: pd.DataFrame, logs: pd.DataFrame):
    st.subheader("Next-Activity Drift")
    nxt = logs[logs["model_name"] == "next_activity"].copy()
    if nxt.empty:
        st.info("No next-activity prediction logs yet.")
        return []

    alerts = []
    current_col = "input__current_activity"
    pred_col = "prediction"
    baseline_current = baseline["concept:name"] if "concept:name" in baseline.columns else pd.Series(dtype=str)

    if current_col in nxt.columns and not baseline_current.empty:
        js_current = _js_divergence_from_counts(baseline_current.value_counts(), nxt[current_col].value_counts())
        st.write(f"JS current-activity distribution: `{js_current:.4f}`")
        alerts.append(
            {
                "model": "next_activity",
                "metric": "JS current_activity",
                "value": js_current,
                "severity": _severity_from_js(js_current),
            }
        )
        unseen = (~nxt[current_col].isin(set(baseline_current.unique()))).mean()
        st.write(f"Unseen current activity rate: `{float(unseen):.4f}`")
        alerts.append(
            {
                "model": "next_activity",
                "metric": "unseen_current_activity_rate",
                "value": float(unseen),
                "severity": "high" if unseen > 0.01 else "low",
            }
        )

    if pred_col in nxt.columns and not baseline_current.empty:
        js_pred = _js_divergence_from_counts(baseline_current.value_counts(), nxt[pred_col].value_counts())
        st.write(f"JS predicted-activity distribution: `{js_pred:.4f}`")
        alerts.append({"model": "next_activity", "metric": "JS predicted_activity", "value": js_pred, "severity": _severity_from_js(js_pred)})

    st.caption(f"Logged predictions: {len(nxt)}")
    return alerts


def main():
    auth.ensure_login()
    auth.render_session_panel()
    auth.require_admin_role()

    st.title("Data Drift Dashboard")
    st.caption("Compares current inference log against training baseline.")

    try:
        baseline = backend.load_baseline_features()
    except Exception as exc:
        st.error(str(exc))
        return

    logs = backend.load_prediction_log()
    if logs.empty:
        st.warning("No prediction log found. Run some predictions first.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Baseline Rows", len(baseline))
    col2.metric("Logged Predictions", len(logs))

    alerts = []
    tabs = st.tabs(["Wait Time", "Lenght of Stay", "Next Activity", "Alerts"])
    with tabs[0]:
        alerts.extend(_render_wait_time_drift(baseline, logs))
    with tabs[1]:
        alerts.extend(_render_los_drift(baseline, logs))
    with tabs[2]:
        alerts.extend(_render_next_activity_drift(baseline, logs))
    with tabs[3]:
        if not alerts:
            st.info("No drift metrics computed yet.")
        else:
            alert_df = pd.DataFrame(alerts)
            st.dataframe(alert_df.sort_values(["severity", "model"], ascending=[False, True]), use_container_width=True)


if __name__ == "__main__":
    main()
