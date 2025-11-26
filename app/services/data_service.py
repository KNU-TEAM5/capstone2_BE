# app/services/data_service.py
import pandas as pd
import os
import re
from typing import Any, Dict, List
from pathlib import Path
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

    # pandas로 CSV 파싱 (여러 인코딩 시도)
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
    df = None
    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
            break  # 성공하면 루프 종료
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue

    if df is None:
        raise ValueError(f"CSV 파일 인코딩을 감지할 수 없습니다. 시도한 인코딩: {encodings}. 마지막 오류: {last_error}")

    # 타임스탬프를 포함한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"uploaded_{timestamp}_{file.filename}"
    save_path = os.path.join(DATA_DIR, saved_filename)

    # 파일 저장
    df.to_csv(save_path, index=False)

    # 업로드된 파일이 센서 파일인지 확인
    is_sensor = is_sensor_file(file.filename)

    # 전체 센서 파일 정보 가져오기
    sensor_info = get_sensor_file_info(DATA_DIR)

    # 기본 통계 정보 생성
    analysis = {
        "saved_path": save_path,
        "is_sensor_file": is_sensor,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "basic_stats": df.describe().to_dict() if len(df) > 0 else {},
        "preview": df.head(5).to_dict(orient="records"),  # 처음 5개 행 미리보기
        "sensor_file_info": sensor_info  # 전체 센서 파일 정보
    }

    return analysis


def is_sensor_file(filename: str) -> bool:
    """
    파일명이 센서 파일인지 확인 (파일명에 'error'가 포함되지 않으면 센서 파일)

    Args:
        filename: 확인할 파일명

    Returns:
        센서 파일이면 True, 아니면 False
    """
    return not re.search(r'error', filename, flags=re.IGNORECASE)


def get_sensor_files(data_dir: str = DATA_DIR) -> List[str]:
    """
    데이터 디렉토리에서 센서 CSV 파일 목록을 반환

    Args:
        data_dir: 데이터 디렉토리 경로

    Returns:
        센서 CSV 파일명 리스트
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        return []

    # CSV 파일 중 파일명에 'error'가 포함되지 않은 것만 센서 파일로 분류
    sensor_paths = [
        p for p in data_path.glob("*.csv")
        if is_sensor_file(p.name)
    ]

    sensor_filenames = [p.name for p in sensor_paths]
    return sensor_filenames


def get_sensor_file_info(data_dir: str = DATA_DIR) -> Dict[str, Any]:
    """
    센서 파일 정보를 반환

    Args:
        data_dir: 데이터 디렉토리 경로

    Returns:
        센서 파일 개수 및 파일명 리스트
    """
    sensor_files = get_sensor_files(data_dir)

    return {
        "sensor_file_count": len(sensor_files),
        "sensor_files": sensor_files
    }
