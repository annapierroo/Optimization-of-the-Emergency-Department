"""Lenght of Stay prediction page."""
from __future__ import annotations

import streamlit as st

from app import auth
from app import backend


def main():
    auth.ensure_login()
    auth.render_session_panel()
    auth.require_user_role()

    st.title("Lenght of Stay Prediction")

    try:
        model, scaler, bundle_path = backend.load_los_assets()
    except Exception as exc:
        st.warning(f"LOS model not ready. {exc}")
        st.caption("Run `python run_training.py --trainers los` first.")
        return

    try:
        procedure_options = backend.load_los_procedure_options()
    except Exception as exc:
        st.warning(f"Procedure list unavailable. {exc}")
        return

    st.write(f"Loaded model bundle: `{bundle_path.name}`")

    start_hour = st.slider("Start Hour", min_value=0, max_value=23, value=10, step=1)
    start_day_of_week = st.slider("Day Index (0=Mon ... 6=Sun)", min_value=0, max_value=6, value=1, step=1)
    time_of_day_encoded = st.slider("Time of Day Encoded", min_value=0, max_value=3, value=1, step=1)
    total_prior_encounters = st.number_input("Total Prior Encounters", min_value=0, value=1, step=1)
    avg_prior_duration = st.number_input("Average Prior Duration", min_value=0.0, value=1.0, step=0.1)
    base_cost = st.number_input("Base Cost", min_value=0.0, value=100.0, step=10.0)
    descriptions = procedure_options["DESCRIPTION"].tolist()
    selected_description = st.selectbox("Procedure", options=descriptions, index=0)
    selected_row = procedure_options[procedure_options["DESCRIPTION"] == selected_description].iloc[0]
    code = int(selected_row["CODE"])
    description_encoded = int(selected_row["description_encoded"])
    is_emergency = st.selectbox("Is Emergency", options=[0, 1], index=0)

    values = {
        "start_hour": start_hour,
        "start_day_of_week": start_day_of_week,
        "start_month": 1,
        "start_year": 2024,
        "season": 1,
        "is_weekend": int(start_day_of_week >= 5),
        "time_of_day_encoded": time_of_day_encoded,
        "total_prior_encounters": total_prior_encounters,
        "avg_prior_duration": avg_prior_duration,
        "avg_prior_cost": 1000.0,
        "days_since_last_encounter": 7.0,
        "encounters_last_30_days": 1,
        "encounters_last_90_days": 2,
        "description_encoded": description_encoded,
        "reason_encoded": 0,
        "CODE": code,
        "BASE_COST": base_cost,
        "has_reason": 0,
        "is_emergency": is_emergency,
    }

    if st.button("Predict LOS"):
        prediction = backend.predict_los(model, scaler, values)
        prediction_id = backend.log_prediction_event(
            model_name="los",
            inputs=values | {"DESCRIPTION": selected_description},
            prediction=prediction,
        )
        st.session_state["los_prediction_id"] = prediction_id
        st.session_state["los_prediction_value"] = prediction
        st.metric("Predicted LOS (hours)", f"{prediction:.2f}")

    if "los_prediction_id" in st.session_state:
        st.subheader("Actual Outcome Tracking")
        st.caption(f"Prediction ID: `{st.session_state['los_prediction_id']}`")
        actual_los = st.number_input("Actual LOS (hours)", min_value=0.0, value=0.0, step=0.1)
        if st.button("Save Actual LOS"):
            abs_error = backend.attach_actual_outcome(st.session_state["los_prediction_id"], actual_los)
            if abs_error is None:
                st.success("Actual outcome saved.")
            else:
                st.success(f"Actual outcome saved. Absolute error: {abs_error:.2f}")


if __name__ == "__main__":
    main()
