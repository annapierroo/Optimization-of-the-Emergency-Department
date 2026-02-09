"""Dedicated Streamlit dashboard for next-activity prediction."""
from __future__ import annotations

import streamlit as st

from app import backend


def main():
    st.set_page_config(page_title="Next Activity Dashboard", layout="centered")
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
        st.success(f"Predicted next activity: {prediction}")


if __name__ == "__main__":
    main()
