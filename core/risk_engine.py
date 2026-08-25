import joblib
import numpy as np
import pandas as pd
from pathlib import Path    

MODEL_DIR = Path(__file__).parent.parent / "models"

model = joblib.load(MODEL_DIR / "risk_classifier_v2.pkl")
scaler = joblib.load(MODEL_DIR / "scaler_v2.pkl")
encoder = joblib.load(MODEL_DIR / "label_encoder_v2.pkl")

def predict_risk(no_helmet_count, no_vest_count, past_incidents, high_risk_zone):
    df = pd.DataFrame({
        "no_helmet_count": [no_helmet_count],
        "no_vest_count": [no_vest_count],
        "past_incidents": [past_incidents],
        "high_risk_zone": [high_risk_zone]
    })
    X_scaled = scaler.transform(df)
    tahmin = model.predict(X_scaled)
    # encoder.inverse_transform(tahmin)[0]
    
    return encoder.inverse_transform(tahmin)[0]


if __name__ == "__main__": 
    no_helmet_count = 2
    no_vest_count = 1
    past_incidents = 3
    high_risk_zone = 1

    risk_prediction = predict_risk(no_helmet_count, no_vest_count, past_incidents, high_risk_zone)
    print(f"Risk Tahmini: {risk_prediction}")
