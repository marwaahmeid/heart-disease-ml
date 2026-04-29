# =========================================
# ❤️ HEART DISEASE FASTAPI (IMPROVED)
# =========================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os

# =========================================
# 🔹 Load Model
# =========================================

MODEL_PATH = "models/heart_rf_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found")

data = joblib.load(MODEL_PATH)

model = data["model"]
features = data["features"]

# ⚠️ scaler اختياري فقط (إذا استخدمته بالتدريب)
scaler = data.get("scaler", None)

# =========================================
# 🔹 App
# =========================================

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)

# =========================================
# 🔹 CORS (مهم للواجهة)
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# 🔹 Input Schema
# =========================================

class Patient(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# =========================================
# 🔹 Home
# =========================================

@app.get("/")
def home():
    return {"message": "❤️ API Running"}

# =========================================
# 🔹 Predict
# =========================================

@app.post("/predict")
def predict(p: Patient):

    try:
        # تحويل البيانات
        input_data = np.array([[getattr(p, f) for f in features]])

        # Scaling (إذا موجود)
        if scaler:
            input_data = scaler.transform(input_data)

        # Prediction
        pred = int(model.predict(input_data)[0])

        # Probability
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(input_data)[0][1])

        return {
            "prediction": pred,
            "probability": prob,
            "result": "Heart Disease" if pred == 1 else "Healthy",
            "risk_level": "HIGH" if pred == 1 else "LOW"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    import uvicorn

