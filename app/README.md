# App Quick Guide

## 1. Run

From project root:

```bash
source /mnt/c/_migration/Dealings/Zetten_v0.9.7.7/Slip_Box/Refer_Nodes/Lecture_Nodes/_venv/bin/activate
python run_training.py
python -m streamlit run app/streamlit_app.py
```

Open:

- `http://localhost:8501`

## 2. Login Roles

- `admin / admin`
  - Landing page only
  - Can start retraining
  - Can open Data Drift dashboard
- `user / user`
  - Prediction pages:
    - Wait Time
    - Next Activity
    - Lenght of Stay

## 3. Prediction Tracking

After each prediction, an **Actual Outcome Tracking** section appears.

- Save actual outcome to append feedback into:
  - `data/monitoring/prediction_log.parquet`
- Numeric models (Wait Time, Lenght of Stay) also store `abs_error`.

## 4. Data Drift Interpretation

Data Drift dashboard compares current prediction logs vs training baseline.

Key metrics:

- `PSI` (numeric feature drift)
  - `<= 0.10`: low
  - `0.10 - 0.25`: medium
  - `> 0.25`: high
- `JS` divergence (categorical distribution drift)
  - Higher means stronger distribution shift.

Example:

- `next_activity | JS current_activity = 0.60 | high`
- `next_activity | JS predicted_activity = 0.54 | high`

Meaning: current process behavior is far from training behavior; next-activity model reliability is degraded.

## 5. Common Debug Checks

If app shows model-not-ready warnings:

1. Confirm artifacts exist:
   - `artifacts/models/xgb_model.joblib`
   - `artifacts/models/best_los_model.joblib`
   - `artifacts/models/next_activity_xgb.joblib` (or `.json`)
   - `artifacts/models/input_encoder.pkl`
   - `artifacts/models/output_encoder.pkl`
2. Re-run training:
   - `python run_training.py`
3. Restart Streamlit after code changes.

If `localhost:8501` is unreachable:

- Ensure Streamlit process is still running.
- Relaunch:
  - `python -m streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501`
