# app/api/v1/qc.py
from fastapi import APIRouter
from app.services.ml_service import load_feature_importance

router = APIRouter()

@router.get("/feature-importance")
def get_feature_importance():
    """
    저장된 CSV 파일(feature importance)을 JSON으로 반환
    """
    data = load_feature_importance()
    return {"feature_importance": data}