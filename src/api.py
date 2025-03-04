import os
import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# Get the absolute path to models folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to "src/"
MODEL_RF_PATH = os.path.join(BASE_DIR, "..", "models", "model_rf.pkl")
MODEL_XGB_PATH = os.path.join(BASE_DIR, "..", "models", "model_xgb.pkl")
MODEL_ENSEMBLE_PATH = os.path.join(BASE_DIR, "..", "models", "model_ensemble.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "feature_names.pkl")

# Load models
rf_model = joblib.load(MODEL_RF_PATH)
xgb_model = joblib.load(MODEL_XGB_PATH)
ensemble_model = joblib.load(MODEL_ENSEMBLE_PATH)

# Load the expected feature names
expected_features = joblib.load(FEATURES_PATH)

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Ensure all expected features exist
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  # Default value for missing columns

    # Ensure feature order matches training
    df = df[expected_features]

    # Convert data to numeric (if necessary)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Make predictions
    rf_pred = rf_model.predict(df)[0]
    xgb_pred = xgb_model.predict(df)[0]
    ensemble_pred = ensemble_model.predict(df)[0]

    print(" Debug - Raw Predictions:")
    print(f"Random Forest: {rf_pred}, XGBoost: {xgb_pred}, Ensemble: {ensemble_pred}")  # Print predictions

    return {
        "Random Forest Prediction": int(rf_pred),
        "XGBoost Prediction": int(xgb_pred),
        "Ensemble Model Prediction": int(ensemble_pred)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
