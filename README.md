Optimization of the Emergency Department

End-to-end MLOps project to analyze, visualize, and predict patient flows in an Emergency Department (ED) by combining Process Mining (PM4Py) and Predictive Modeling (XGBoost). The pipeline shows graphs of the patient journey, waiting times, and identifies bottlenecks; it also trains a model to predict the next activity in a patient pathway and the estimated waiting time.  ￼


Repository structure (high level)

app/        # Streamlit dashboard + inference
data/       # data storage (DVC-managed)
models/     # trained models and encoders
reports/    # figures and CSV outputs
src/        # ingestion, process mining, training pipeline code
tests/      # unit tests
dvc.yaml    # DVC pipeline stages
Dockerfile  # reproducible runtime environment

⸻

Quick start (recommended): Docker + DVC pipeline

Prerequisites

Install:
	•	Git
	•	Docker Desktop (must be running)

1) Clone the repository

git clone https://github.com/annapierroo/Optimization-of-the-Emergency-Department.git
cd Optimization-of-the-Emergency-Department

2) Build the Docker image

docker build -t ed-optimizer .



3) Run the full pipeline (one command)

This mounts your project into the container, runs the DVC pipeline, and ensures outputs are writable on the host.

docker run -v "$(pwd)":/app ed-optimizer /bin/bash -c "dvc repro && chmod -R 777 reports"

If you are on Windows PowerShell, use:

docker run -v ${PWD}:/app ed-optimizer /bin/bash -c "dvc repro && chmod -R 777 reports"

￼

4) Launch the Streamlit dashboard

Option A (run locally, if your local env has dependencies installed):

streamlit run app/streamlit_app.py

Option B (run via Docker; exposes Streamlit on localhost:8501):

docker run -it -p 8501:8501 -v "$(pwd)":/app ed-optimizer \
  /bin/bash -c "streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501"

￼

Open in browser:
	•	http://localhost:8501

⸻

Alternative: run step-by-step (inside Docker)

If you want to debug individual steps:

docker run -it -v "$(pwd)":/app ed-optimizer /bin/bash

Then inside the container:

python src/ingest_data.py
python src/train_next_activity.py

￼

⸻

Alternative: run locally (without Docker)

Use this if you prefer a native setup. Docker is still recommended for reproducibility.

1) Create and activate a virtual environment

macOS/Linux

python -m venv .venv
source .venv/bin/activate

Windows (PowerShell)

python -m venv .venv
.\.venv\Scripts\Activate.ps1

2) Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

3) Run the pipeline

If you have DVC installed:

dvc repro

Or run scripts directly:

python src/ingest_data.py
python src/train_next_activity.py

4) Run the dashboard

streamlit run app/streamlit_app.py

----

Configuration

Change cohort size / number of cases

In src/ingest_data.py, edit:

n_cases = 1000  # set to None for all data

Then re-run.

