# app/api/v1/results.py
"""
분석 결과 조회 관련 엔드포인트
artifacts 폴더에 저장된 분석 결과를 조회합니다.
"""
from fastapi import APIRouter, HTTPException
from app.services.artifact_service import (
    load_feature_importance,
    load_confusion_matrix,
    load_classification_report_rf,
    load_safe_region_result,
)

router = APIRouter()


@router.get("/feature-importance")
def get_feature_importance():
    """
    저장된 CSV 파일(feature importance)을 JSON으로 반환
    """
    data = load_feature_importance()
    return {"feature_importance": data}


@router.get("/confusion-matrix")
def get_confusion_matrix():
    """
    혼동 행렬(Confusion Matrix) 데이터를 반환
    """
    csv_path = "artifacts/confusion_matrix_randomforest.csv"
    cm_data = load_confusion_matrix(csv_path)

    return {
        "confusion_matrix": cm_data
    }


@router.get("/classification-report-rf")
def get_classification_report_rf():
    """
    RandomForest 분류 리포트 JSON 반환 엔드포인트
    (artifacts/classification_report_randomforest.json)
    """
    try:
        report = load_classification_report_rf()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 그대로 반환하면 프론트에서 바로 사용 가능
    return report


@router.get("/safe-region")
def get_safe_region_result():
    """
    공정 안전 구간(safe region) 결과 반환 엔드포인트
    → artifacts/safe_region_result.json 읽어서 그대로 반환
    """
    try:
        result = load_safe_region_result()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return result
