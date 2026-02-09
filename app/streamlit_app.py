"""Landing page for Streamlit multipage dashboard."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

from app import auth


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    st.set_page_config(page_title="ED Dashboard", layout="centered")
    auth.ensure_login()
    if auth.get_role() != "user":
        auth.hide_sidebar_navigation()
    auth.render_session_panel()

    st.title("Emergency Department Dashboard")
    st.write("Use the left sidebar to switch pages.")
    st.write("Available pages:")
    st.write("- Wait Time")
    st.write("- Next Activity")
    st.write("- Lenght of Stay")
    if auth.is_admin():
        st.write("- Data Drift")

    if auth.is_admin():
        st.subheader("Admin Controls")
        if st.button("Open Data Drift Dashboard"):
            st.switch_page("pages/4_Data_Drift.py")

        trainers = st.multiselect(
            "Retrain models",
            options=["wait_time", "los", "next_activity"],
            default=["wait_time", "los", "next_activity"],
        )
        skip_ingestion = st.checkbox("Skip ingestion", value=False)
        skip_features = st.checkbox("Skip features", value=False)

        if st.button("Run Retraining"):
            cmd = [sys.executable, "run_training.py", "--trainers", *trainers]
            if skip_ingestion:
                cmd.append("--skip-ingestion")
            if skip_features:
                cmd.append("--skip-features")

            with st.spinner("Running retraining..."):
                completed = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                )
            if completed.returncode == 0:
                st.success("Retraining completed.")
            else:
                st.error("Retraining failed.")
            if completed.stdout:
                st.code(completed.stdout[-6000:])
            if completed.stderr:
                st.code(completed.stderr[-6000:])
    else:
        st.info("User role can run predictions only.")


if __name__ == "__main__":
    main()
