"""Wait-time prediction page."""
from __future__ import annotations

import streamlit as st

from app import auth
from app import backend


def main():
    auth.ensure_login()
    auth.render_session_panel()
    auth.require_user_role()

    st.title("Wait-Time Prediction")

    day_index = st.number_input("Day Index (0=Mon ... 6=Sun)", min_value=0, max_value=6, value=0, step=1)
    arrival_hour = st.number_input("Arrival Hour (0-23)", min_value=0, max_value=23, value=10, step=1)

    if st.button("Predict Wait Time"):
        try:
            model = backend.load_latest_model()
            minutes = backend.predict_duration(model, int(day_index), int(arrival_hour))
            prediction_id = backend.log_prediction_event(
                model_name="wait_time",
                inputs={"Day_Index": int(day_index), "Arrival_Hour": int(arrival_hour)},
                prediction=minutes,
            )
            st.session_state["wait_time_prediction_id"] = prediction_id
            st.session_state["wait_time_prediction_value"] = minutes
            st.metric("Predicted Wait Time (minutes)", f"{minutes:.2f}")
        except Exception as exc:
            st.error(str(exc))
            st.caption("Run `python run_training.py --trainers wait_time`.")
    else:
        st.info("Set inputs and click Predict Wait Time.")

    if "wait_time_prediction_id" in st.session_state:
        st.subheader("Actual Outcome Tracking")
        st.caption(f"Prediction ID: `{st.session_state['wait_time_prediction_id']}`")
        actual_wait = st.number_input("Actual Wait Time (minutes)", min_value=0.0, value=0.0, step=1.0)
        if st.button("Save Actual Wait Time"):
            abs_error = backend.attach_actual_outcome(st.session_state["wait_time_prediction_id"], actual_wait)
            if abs_error is None:
                st.success("Actual outcome saved.")
            else:
                st.success(f"Actual outcome saved. Absolute error: {abs_error:.2f}")


if __name__ == "__main__":
    main()
