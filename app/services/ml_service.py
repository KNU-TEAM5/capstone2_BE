# app/services/ml_service.py
import pandas as pd
import os

ART_DIR = "artifacts"

def load_feature_importance():
    path = os.path.join(ART_DIR, "feature_importance_rf.csv")
    
    if not os.path.exists(path):
        return {"error": "feature importance file not found"}
    
    df = pd.read_csv(path)
    return df.to_dict(orient="records")