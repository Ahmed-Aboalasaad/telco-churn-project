"""
Telco Customer Churn Prediction API
=====================================
FastAPI-based REST API for predicting customer churn.
Load the trained pipeline from models/final_model.pkl and expose prediction endpoints.

Run:
    uvicorn api.main:app --reload

Docs:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
from typing import Literal

# ── Load model once at startup ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")

try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. "
        "Please run the Jupyter notebook first to train and save the model."
    )

# ── Pydantic input schema ───────────────────────────────────────────────────────
class CustomerData(BaseModel):
    gender:           Literal["Male", "Female"]         = Field(..., example="Female")
    SeniorCitizen:    Literal["Yes", "No"]              = Field(..., example="No",
                          description="'Yes' or 'No' (not 0/1)")
    Partner:          Literal["Yes", "No"]              = Field(..., example="Yes")
    Dependents:       Literal["Yes", "No"]              = Field(..., example="No")
    tenure:           int                               = Field(..., ge=0, le=72, example=12)
    PhoneService:     Literal["Yes", "No"]              = Field(..., example="Yes")
    MultipleLines:    Literal["Yes", "No", "No phone service"] = Field(..., example="No")
    InternetService:  Literal["DSL", "Fiber optic", "No"]     = Field(..., example="Fiber optic")
    OnlineSecurity:   Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    OnlineBackup:     Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    TechSupport:      Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingTV:      Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")
    StreamingMovies:  Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")
    Contract:         Literal["Month-to-month", "One year", "Two year"] = Field(..., example="Month-to-month")
    PaperlessBilling: Literal["Yes", "No"]              = Field(..., example="Yes")
    PaymentMethod:    Literal[
                          "Electronic check",
                          "Mailed check",
                          "Bank transfer (automatic)",
                          "Credit card (automatic)"
                      ]                                 = Field(..., example="Electronic check")
    MonthlyCharges:   float                             = Field(..., ge=0, le=500, example=85.5)
    TotalCharges:     float                             = Field(..., ge=0, example=1026.0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 1026.0
            }
        }

# ── Response schema ─────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction:        int   = Field(..., description="0 = No Churn, 1 = Churn")
    prediction_label:  str   = Field(..., description="'Churn' or 'No Churn'")
    churn_probability: float = Field(..., description="Probability of churn (0–1)")
    risk_level:        str   = Field(..., description="'High Risk', 'Medium Risk', or 'Low Risk'")


def get_risk_level(prob: float) -> str:
    """Classify churn probability into risk tiers."""
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


# ── App ─────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "REST API for predicting whether a telecom customer will churn. "
        "The model is a scikit-learn Pipeline trained on the Telco Customer Churn dataset. "
        "Visit /docs for interactive Swagger UI."
    ),
    version="1.0.0",
    contact={"name": "Data Exploration & Preparation Course"},
)


# ── Health check ────────────────────────────────────────────────────────────────
@app.get("/", summary="Health Check")
def root():
    """
    Verify that the API is running and the model is loaded.
    """
    return {"message": "Telco Churn Prediction API is running"}


@app.get("/health", summary="Detailed Health Check")
def health():
    """
    Returns model type and API status.
    """
    return {
        "status": "ok",
        "model_type": type(pipeline.named_steps.get("model", pipeline)).__name__,
        "api_version": "1.0.0"
    }


# ── Prediction endpoint ─────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, summary="Predict Customer Churn")
def predict(customer: CustomerData):
    """
    Predict whether a telecom customer will churn.

    - **Input**: Full customer profile (demographics, account, services, billing).
    - **Output**: Churn prediction (0/1), label, probability, and risk level.

    The model pipeline handles all preprocessing internally — no manual encoding needed.
    """
    try:
        # Convert Pydantic model → single-row DataFrame
        data = customer.model_dump()
        df   = pd.DataFrame([data])

        # Ensure correct column order (matches training)
        expected_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]
        df = df[expected_cols]

        # Predict
        churn_class = int(pipeline.predict(df)[0])
        churn_prob  = float(pipeline.predict_proba(df)[0][1])

        return PredictionResponse(
            prediction        = churn_class,
            prediction_label  = "Churn" if churn_class == 1 else "No Churn",
            churn_probability = round(churn_prob, 4),
            risk_level        = get_risk_level(churn_prob)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── Batch prediction endpoint ───────────────────────────────────────────────────
@app.post("/predict/batch", summary="Batch Predict Customer Churn")
def predict_batch(customers: list[CustomerData]):
    """
    Predict churn for multiple customers at once.

    - **Input**: List of customer profiles (max 100 per request).
    - **Output**: List of predictions.
    """
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 customers per batch request.")

    try:
        rows = [c.model_dump() for c in customers]
        df   = pd.DataFrame(rows)

        expected_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]
        df = df[expected_cols]

        classes = pipeline.predict(df).tolist()
        probs   = pipeline.predict_proba(df)[:, 1].tolist()

        return [
            {
                "index":             i,
                "prediction":        classes[i],
                "prediction_label":  "Churn" if classes[i] == 1 else "No Churn",
                "churn_probability": round(probs[i], 4),
                "risk_level":        get_risk_level(probs[i])
            }
            for i in range(len(customers))
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
