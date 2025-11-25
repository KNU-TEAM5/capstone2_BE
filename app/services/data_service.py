# app/services/data_service.py
import pandas as pd
import os
from typing import Any, Dict
from fastapi import UploadFile
from datetime import datetime
import io

DATA_DIR = "data"

async def process_uploaded_csv(file: UploadFile) -> Dict[str, Any]:
    """
    업로드된 CSV 파일을 처리하고 기본 분석 정보를 반환

    Args:
        file: 업로드된 CSV 파일

    Returns:
        파일 정보 및 기본 분석 결과를 담은 딕셔너리
    """
    # data 디렉토리가 없으면 생성
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 파일 읽기
    contents = await file.read()

    # pandas로 CSV 파싱
    df = pd.read_csv(io.BytesIO(contents))

    # 타임스탬프를 포함한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"uploaded_{timestamp}_{file.filename}"
    save_path = os.path.join(DATA_DIR, saved_filename)

    # 파일 저장
    df.to_csv(save_path, index=False)

    # 기본 통계 정보 생성
    analysis = {
        "saved_path": save_path,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "basic_stats": df.describe().to_dict() if len(df) > 0 else {},
        "preview": df.head(5).to_dict(orient="records")  # 처음 5개 행 미리보기
    }

    return analysis
