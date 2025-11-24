from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


class FraudRequest(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float


app = FastAPI(title="Fraud Detection API")

MODEL_PATH = "fraud_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Run src/model.py first.")

model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}


@app.post("/predict")
def predict(data: FraudRequest):
    df = pd.DataFrame([data.dict()])
    preds = model.predict(df)
    df['is_fraud'] = [1 if p == -1 else 0 for p in preds]
    return {"prediction": int(df['is_fraud'].iloc[0])}
