"""Next-activity prediction page."""
from __future__ import annotations

import streamlit as st

from app import auth
from app import backend


def main():
    auth.ensure_login()
    auth.render_session_panel()
    auth.require_user_role()

    st.title("Next Activity Prediction")

    try:
        model, input_encoder, output_encoder, model_path = backend.load_next_activity_assets()
    except Exception as exc:
        st.warning(f"Next Activity Model not ready. {exc}")
        st.caption("Run `python run_training.py --trainers next_activity` first.")
        return

    activities = list(getattr(input_encoder, "classes_", []))
    if not activities:
        st.error("Input encoder has no known activities.")
        return

    st.write(f"Loaded model: `{model_path.name}`")
    current_activity = st.selectbox("Current Activity", options=activities, index=0)
    hour = st.slider("Current Hour", min_value=0, max_value=23, value=10, step=1)
    day_index = st.slider("Day Index (0=Mon ... 6=Sun)", min_value=0, max_value=6, value=1, step=1)

    if st.button("Predict Next Activity"):
        prediction = backend.predict_next_activity(
            model=model,
            input_encoder=input_encoder,
            output_encoder=output_encoder,
            current_activity=current_activity,
            hour=hour,
            day_index=day_index,
        )
        prediction_id = backend.log_prediction_event(
            model_name="next_activity",
            inputs={
                "current_activity": current_activity,
                "Hour": int(hour),
                "Day_of_Week": int(day_index),
            },
            prediction=prediction,
        )
        st.session_state["next_activity_prediction_id"] = prediction_id
        st.session_state["next_activity_prediction_value"] = prediction
        st.success(f"Predicted next activity: {prediction}")

    if "next_activity_prediction_id" in st.session_state:
        st.subheader("Actual Outcome Tracking")
        st.caption(f"Prediction ID: `{st.session_state['next_activity_prediction_id']}`")
        actual_options = list(getattr(output_encoder, "classes_", []))
        if not actual_options:
            st.error("Output encoder has no known activity classes.")
            return
        actual_next = st.selectbox("Actual Next Activity", options=actual_options, index=0)
        if st.button("Save Actual Next Activity"):
            backend.attach_actual_outcome(st.session_state["next_activity_prediction_id"], actual_next)
            st.success("Actual outcome saved.")


if __name__ == "__main__":
    main()
