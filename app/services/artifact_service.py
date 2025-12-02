# app/services/artifact_service.py
"""
분석 결과 아티팩트 로딩 서비스
artifacts/ 폴더에 저장된 ML 분석 결과 파일들을 읽어서 반환합니다.
"""
import pandas as pd
import os
from typing import Any, Dict
import json

ART_DIR = "artifacts"

def load_feature_importance():
    """
    특성 중요도(Feature Importance) 데이터 로딩

    Returns:
        특성별 중요도 리스트 또는 에러 딕셔너리
    """
    path = os.path.join(ART_DIR, "feature_importance_rf.csv")

    if not os.path.exists(path):
        return {"error": "feature importance file not found"}

    # 첫 번째 컬럼을 인덱스로 읽기
    df = pd.read_csv(path, index_col=0)

    # 인덱스를 feature 컬럼으로 변환
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "feature"})

    # feature와 importance만 반환
    return df[["feature", "importance"]].to_dict(orient="records")

def load_confusion_matrix(csv_path: str):
    """
    혼동 행렬(Confusion Matrix) 데이터 로딩

    Args:
        csv_path: CSV 파일 경로

    Returns:
        혼동 행렬 값들을 의미론적 키로 매핑한 딕셔너리
    """
    df = pd.read_csv(csv_path)

    # CSV 첫 번째 컬럼을 index로 설정
    df = df.set_index(df.columns[0])

    tn = int(df.loc["true_0", "pred_0"])   # 정상→정상
    fp = int(df.loc["true_0", "pred_1"])   # 정상→불량
    fn = int(df.loc["true_1", "pred_0"])   # 불량→정상
    tp = int(df.loc["true_1", "pred_1"])   # 불량→불량

    return {
        "normal_to_normal": tn,
        "normal_to_defect": fp,
        "defect_to_normal": fn,
        "defect_to_defect": tp
    }

def load_classification_report_rf() -> Dict[str, Any]:
    """
    분류 리포트(Classification Report) 로딩
    artifacts/classification_report_rf.json 파일을 읽어서 dict로 반환

    Returns:
        분류 리포트 딕셔너리

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
    """
    json_path = os.path.join(ART_DIR, "classification_report_rf.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def load_safe_region_result() -> Dict[str, Any]:
    """
    안전 영역 분석 결과 로딩
    artifacts/safe_region_result.json 파일을 읽어서 dict로 반환

    Returns:
        안전 영역 분석 결과 딕셔너리

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
    """
    json_path = os.path.join(ART_DIR, "safe_region_result.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data
