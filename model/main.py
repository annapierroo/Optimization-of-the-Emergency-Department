from fastapi import FastAPI, UploadFile, File
import shutil
import joblib
import os

from retraining.retraining import retrain_random_forest, retrain_xgboost

app = FastAPI()

UPLOAD_DIR = "uploads"
SAVED_DIR = "saved"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/retrain/random-forest")
async def retrain_rf(file: UploadFile = File(...)):
    excel_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(excel_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    X_old = joblib.load(f"{SAVED_DIR}/X_old.joblib")
    y_old = joblib.load(f"{SAVED_DIR}/y_old.joblib")

    retrain_random_forest(excel_path, X_old, y_old)

    return {
        "status": "success",
        "model": "random_forest"
    }


@app.post("/retrain/xgboost")
async def retrain_xgb(file: UploadFile = File(...)):
    excel_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(excel_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    X_old = joblib.load(f"{SAVED_DIR}/X_old.joblib")
    y_old = joblib.load(f"{SAVED_DIR}/y_old.joblib")

    retrain_xgboost(excel_path, X_old, y_old)

    return {
        "status": "success",
        "model": "xgboost"
    }
