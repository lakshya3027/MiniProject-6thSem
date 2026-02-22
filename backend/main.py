# ===============================
# IMPORTS
# ===============================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List

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
    V_features: List[float]


# ===============================
# LOAD MODEL & SCALERS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
time_scaler = None
amount_scaler = None

@app.on_event("startup")
def load_model():
    global model, time_scaler, amount_scaler

    try:
        model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
        time_scaler = joblib.load(os.path.join(BASE_DIR, "time_scaler.pkl"))
        amount_scaler = joblib.load(os.path.join(BASE_DIR, "amount_scaler.pkl"))
        print("✅ Model and scalers loaded successfully")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise e


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

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
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
    

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)