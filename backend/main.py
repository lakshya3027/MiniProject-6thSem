# ===============================
# IMPORTS
# ===============================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os

# ===============================
# CREATE FASTAPI APP
# ===============================
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML API for fraud prediction using FastAPI",
    version="1.0"
)

# ===============================
# ENABLE CORS (Frontend Connection)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# REQUEST SCHEMA (VALIDATION)
# ===============================
class Transaction(BaseModel):
    time: float
    amount: float
    V_features: list[float]


# ===============================
# LOAD MODEL & SCALERS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    time_scaler = joblib.load(os.path.join(BASE_DIR, "time_scaler.pkl"))
    amount_scaler = joblib.load(os.path.join(BASE_DIR, "amount_scaler.pkl"))
except Exception as e:
    raise RuntimeError(f"Error loading model or scalers: {e}")


# ===============================
# ROOT ROUTE
# ===============================
@app.get("/")
def home():
    return {"message": "Fraud Detection API Running Successfully 🚀"}


# ===============================
# HEALTH CHECK (Optional but Professional)
# ===============================
@app.get("/health")
def health_check():
    return {"status": "API is healthy"}


# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.post("/predict")
def predict(transaction: Transaction):

    try:
        # -----------------------------
        # EXTRACT DATA
        # -----------------------------
        time = transaction.time
        amount = transaction.amount
        V_features = transaction.V_features

        # Validate PCA features
        if len(V_features) != 28:
            raise HTTPException(
                status_code=400,
                detail="V_features must contain exactly 28 values"
            )

        # -----------------------------
        # PREPARE INPUT ARRAY
        # -----------------------------
        input_data = np.array(
            [time] + V_features + [amount]
        ).reshape(1, -1)

        # -----------------------------
        # SCALE FEATURES
        # -----------------------------
        input_data[0, 0] = time_scaler.transform([[time]])[0][0]
        input_data[0, -1] = amount_scaler.transform([[amount]])[0][0]

        # -----------------------------
        # MODEL PREDICTION
        # -----------------------------
        fraud_prob = model.predict_proba(input_data)[0][1]

        threshold = 0.3
        prediction = 1 if fraud_prob >= threshold else 0

        # -----------------------------
        # RESPONSE
        # -----------------------------
        return {
            "fraud_probability": float(fraud_prob),
            "prediction": prediction,
            "status": "FRAUD" if prediction == 1 else "SAFE"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))