# app/services/ml_service.py
import pandas as pd
import os
from typing import Any, Dict
import json

ART_DIR = "artifacts"

def load_feature_importance():
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
    confusion_matrix_randomforest.csv 파일을 읽어서
    프론트 용도로 가공된 dict 형태로 반환한다.
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
    artifacts/classification_report_rf.json 파일을 읽어서 dict로 반환
    """
    json_path = os.path.join(ART_DIR, "classification_report_rf.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def load_safe_region_result() -> Dict[str, Any]:
    """
    artifacts/safe_region_result.json 파일을 읽어서 dict로 반환
    """
    json_path = os.path.join(ART_DIR, "safe_region_result.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data