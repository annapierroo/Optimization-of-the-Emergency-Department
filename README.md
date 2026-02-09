# Optimization of the Emergency Department

## Project Overview
This repository implements an end-to-end **MLOps framework** designed to analyze, visualize, and predict patient flows within an Emergency Department (ED). By integrating **Process Mining** (PM4Py) with **Predictive Analytics** (XGBoost), the system identifies operational bottlenecks and predicts future patient states to support resource allocation decisions.

Unlike standard data analysis scripts, this project is engineered with a **modular architecture**, ensuring reproducibility via **Docker** and data versioning via **DVC**.

## Core Capabilities

### 1. Process Discovery & Mining
* **Graph Reconstruction:** Utilizes the *Directly-Follows Graph (DFG)* algorithm to map patient journeys from Triage to Discharge.
* **Bottleneck Analysis:** Quantifies transition times between hospital activities to pinpoint structural inefficiencies.

### 2. Predictive Modeling (Next Activity Prediction)
* **Algorithm:** Implements a Gradient Boosting classifier (**XGBoost**) to predict the next clinical step for a patient based on their current trajectory.
* **Feature Engineering:** Extracts temporal patterns (hour, day of week) and sequential lag features.
* **Robust Training:** Includes stratified sampling and rare-class filtering to handle the inherent class imbalance of hospital event logs.

### 3. Software Engineering
* **Hexagonal/Modular Architecture:** The codebase uses a Protocol-based design (Ports & Adapters pattern) seen in `src/pipeline_architecture.py`, decoupling the core logic (Ingestion, Training, Evaluation) from specific implementations.
* **Reproducibility:** Fully containerized environment using Docker.

## Tech Stack

* **Language:** Python 3.9+
* **Process Mining:** PM4Py
* **Machine Learning:** XGBoost, Scikit-Learn
* **Orchestration & MLOps:** DVC (Data Version Control), Docker
* **Visualization:** Streamlit (Interactive Dashboard)

## Project Structure

```text
├── app/                 # Streamlit Dashboard and Inference Backend
├── data/                # Data storage (managed by DVC)
├── models/              # Serialized XGBoost models and Encoders
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── src/                 # Core Source Code
│   ├── pipeline_architecture.py  # Protocol definitions (Ports)
│   ├── train_next_activity.py    # Training logic
│   ├── ingest_data.py            # ETL pipelines
│   └── process_discovery.py      # PM4Py logic
├── tests/               # Unit tests
├── Dockerfile           # Environment definition
└── dvc.yaml             # Pipeline stage definitions
```
## Quick Start

### Prerequisites
Before running the pipeline, ensure your environment meets the following requirements:
* **Docker Desktop**: Must be installed and running (the project runs entirely in containers to avoid dependency hell).
* **Git**: For version control.
* **Make** (Optional): If you prefer using a Makefile for commands.

### 1. Setup & Installation
Build the Docker image containing the full runtime environment (Python 3.9, PM4Py, XGBoost).

```bash
# Build the image with the tag 'ed-optimizer'
docker build -t ed-optimizer .
```

## Quick Start

### Prerequisites

Before running the pipeline, ensure your environment meets the following requirements:

* **Docker Desktop**: Must be installed and running (the project runs entirely in containers to avoid dependency hell).
* **Git**: For version control.

### 1. Setup & Installation

Build the Docker image containing the full runtime environment (Python 3.9, PM4Py, XGBoost).

```bash
# Build the image with the tag 'ed-optimizer'
docker build -t ed-optimizer .

```

### 2. Running the Pipeline

You can execute the full MLOps workflow (Ingestion → Processing → Training) using a single Docker command. We use **DVC** (Data Version Control) to manage the pipeline stages.

**Option A: Automated Pipeline (Recommended)**
This command mounts your current directory (`$(pwd)`) to the container, runs the DVC reproduction, and ensures the output files are accessible.

```bash
docker run -v $(pwd):/app ed-optimizer /bin/bash -c "dvc repro && chmod -R 777 reports"

```

**Option B: Manual Execution**
If you need to debug specific steps:

```bash
docker run -it -v $(pwd):/app ed-optimizer /bin/bash
# Inside the container:
python src/ingest_data.py    # Step 1: Ingest & Clean
python src/train_next_activity.py # Step 2: Train XGBoost

```

### 3. Launching the Dashboard

To visualize the Process Map (DFG) and interact with the Prediction Model:

```bash
# Run Streamlit on localhost:8501
streamlit run app/streamlit_app.py

```

---

## Configuration & Customization

### Changing the Cohort Size

By default, the pipeline processes a subset of patients for speed. To analyze the full dataset or a different sample size:

1. Open `src/ingest_data.py`.
2. Locate the `n_cases` variable configuration:
```python
# src/ingest_data.py
n_cases = 1000  # <--- Update this value (e.g., set to None for all data)

```


3. Re-run the pipeline: `dvc repro`.

### Model Hyperparameters

The XGBoost configuration is decoupled from the training logic. You can adjust hyperparameters (learning rate, max depth) directly in `src/train_next_activity.py` or move them to a `params.yaml` file for better MLOps practice.

---

## Outputs & Artifacts

After a successful run, the system generates the following artifacts in the `reports/` and `models/` directories:

| Artifact | Path | Description |
| --- | --- | --- |
| **Process Map** | `reports/figures/patient_journey_dfg.png` | Visual representation of the Directly-Follows Graph showing patient flow. |
| **Transition Stats** | `reports/waiting_transitions.csv` | Statistical breakdown of waiting times between activities. |
| **XGBoost Model** | `models/next_activity_xgb.json` | The trained predictive model for next-activity classification. |
| **Encoders** | `models/*.pkl` | Serialized LabelEncoders for categorical features. |

---

```

```
