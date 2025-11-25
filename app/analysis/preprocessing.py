# app/analysis/preprocessing.py

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from scipy import stats


def read_csv_safely(path: str) -> pd.DataFrame:
    """
    여러 인코딩을 시도하여 CSV 파일을 안전하게 읽기

    Args:
        path: CSV 파일 경로

    Returns:
        읽은 DataFrame
    """
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    # 위 인코딩으로도 안 되면 최후 시도
    return pd.read_csv(path, low_memory=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 컬럼명을 정규화 (소문자, 공백/특수문자 제거)

    Args:
        df: 정규화할 DataFrame

    Returns:
        컬럼명이 정규화된 DataFrame
    """
    df = df.copy()
    df.columns = (
        pd.Series(df.columns)
        .astype(str).str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.lower()
    )
    return df


def rename_sensor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    센서 데이터의 컬럼명을 표준 이름으로 변경

    Args:
        df: 변경할 DataFrame

    Returns:
        컬럼명이 변경된 DataFrame
    """
    df = df.rename(columns={
        "온도": "temp",
        "temp": "temp",
        "습도": "humid",
        "압력": "press",
        "불량여부": "error"
    })
    return df


def process_sensor_files(sensor_paths: List[Path], num_files: int = 2) -> Dict[str, Any]:
    """
    센서 파일들을 읽고 처리하여 정보 반환

    Args:
        sensor_paths: 센서 파일 경로 리스트
        num_files: 처리할 파일 개수 (기본값: 2)

    Returns:
        처리된 DataFrame 리스트와 파일 정보
    """
    dfs_for_merge = []

    for idx, path in enumerate(sensor_paths[:num_files], start=1):
        df = read_csv_safely(str(path))
        df = normalize_columns(df)
        df = rename_sensor_columns(df)

        print(f"\n[단일 파일 확인 {idx}] {path.name}")
        print("행 수:", len(df))
        print("열 수:", df.shape[1])
        print(df.tail(3))   # 끝 3행 미리보기

        dfs_for_merge.append(df)

    return {
        "dataframes": dfs_for_merge,
        "file_count": len(dfs_for_merge)
    }


def get_sensor_files_summary(sensor_paths: List[Path]) -> Dict[str, Any]:
    """
    모든 센서 CSV 파일의 요약 정보 반환

    Args:
        sensor_paths: 센서 파일 경로 리스트

    Returns:
        파일별 정보 및 총 행 수
    """
    per_file = []
    for p in sensor_paths:
        dfi = read_csv_safely(str(p))
        per_file.append({"file": p.name, "rows": len(dfi), "cols": dfi.shape[1]})

    total_rows = sum(x["rows"] for x in per_file)

    print("\n[전체 센서 CSV 요약]")
    print("파일 수:", len(per_file))
    print("총 행 수 합계:", total_rows)

    return {
        "file_count": len(per_file),
        "total_rows": total_rows,
        "per_file": per_file
    }


def merge_sensor_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    여러 DataFrame을 병합

    Args:
        dfs: 병합할 DataFrame 리스트

    Returns:
        병합된 DataFrame
    """
    combined_df = pd.concat(dfs, ignore_index=True)

    print("\n[파일 병합 결과]")
    print("새 데이터프레임 행 수:", len(combined_df))
    print("새 데이터프레임 열 수:", combined_df.shape[1])
    print(combined_df.head(5))

    return combined_df


def save_combined_data(df: pd.DataFrame, output_dir: str = "artifacts", filename: str = "combined_data.csv") -> str:
    """
    병합된 데이터프레임을 CSV로 저장

    Args:
        df: 저장할 DataFrame
        output_dir: 저장할 디렉토리
        filename: 파일명

    Returns:
        저장된 파일 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n[저장 완료] {output_path}")
    return output_path


def preprocess_and_merge_sensors(
    data_dir: str = "data",
    num_files: int = 2,
    output_dir: str = "artifacts"
) -> Dict[str, Any]:
    """
    센서 데이터 전처리 및 병합 전체 파이프라인

    Args:
        data_dir: 데이터 디렉토리
        num_files: 병합할 파일 개수
        output_dir: 결과 저장 디렉토리

    Returns:
        처리 결과 정보
    """
    from pathlib import Path
    import re

    # 센서 파일 수집
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"[ERROR] 데이터 폴더가 없습니다: {data_path}")

    sensor_paths = [
        p for p in data_path.glob("*.csv")
        if not re.search(r'error', p.name, flags=re.IGNORECASE)
    ]

    if len(sensor_paths) == 0:
        raise ValueError("센서 파일이 없습니다.")

    # 파일 처리
    processed = process_sensor_files(sensor_paths, num_files)

    # 전체 요약
    summary = get_sensor_files_summary(sensor_paths)

    # 병합
    combined_df = merge_sensor_dataframes(processed["dataframes"])

    # 저장
    output_path = save_combined_data(combined_df, output_dir)

    return {
        "sensor_file_count": len(sensor_paths),
        "processed_file_count": processed["file_count"],
        "combined_rows": len(combined_df),
        "combined_cols": combined_df.shape[1],
        "output_path": output_path,
        "summary": summary
    }


# =========================================
#  데이터 정제 함수들
# =========================================

def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    결측치 확인

    Args:
        df: 확인할 DataFrame

    Returns:
        결측치 정보
    """
    missing_counts = df.isna().sum()
    total_missing = missing_counts.sum()

    print("📋 [결측치 개수 확인]")
    print(missing_counts)
    print(f"\n총 결측치 개수: {total_missing}")

    return {
        "missing_counts": missing_counts.to_dict(),
        "total_missing": int(total_missing)
    }


def remove_missing_values(df: pd.DataFrame, method: str = "drop") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    결측치 제거 또는 대체

    Args:
        df: 처리할 DataFrame
        method: 'drop' (제거) 또는 'fill' (평균값으로 대체)

    Returns:
        처리된 DataFrame과 처리 정보
    """
    missing_before = df.isnull().sum().sum()

    if method == "drop":
        df_clean = df.dropna().reset_index(drop=True)
    elif method == "fill":
        df_clean = df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("method는 'drop' 또는 'fill'이어야 합니다.")

    missing_after = df_clean.isnull().sum().sum()

    info = {
        "method": method,
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
        "rows_before": len(df),
        "rows_after": len(df_clean),
        "removed_rows": len(df) - len(df_clean)
    }

    print(f"\n결측치 {method} 전: {missing_before}, 후: {missing_after}")
    print(f"결측치 처리 후 행 수: {len(df_clean)}\n")

    return df_clean, info


def remove_outliers_zscore(df: pd.DataFrame, cols: List[str], threshold: float = 3.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Z-score 기반 이상치 제거

    Args:
        df: 처리할 DataFrame
        cols: 이상치를 확인할 컬럼 리스트
        threshold: Z-score 임계값 (기본값: 3.0)

    Returns:
        정제된 DataFrame과 처리 정보
    """
    df = df.copy()
    z_scores = np.abs(stats.zscore(df[cols], nan_policy='omit'))
    mask = (z_scores < threshold).all(axis=1)
    clean_df = df[mask].reset_index(drop=True)

    removed_count = len(df) - len(clean_df)
    removal_rate = (removed_count / len(df)) * 100 if len(df) > 0 else 0

    info = {
        "threshold": threshold,
        "columns": cols,
        "rows_before": len(df),
        "rows_after": len(clean_df),
        "removed_rows": removed_count,
        "removal_rate": round(removal_rate, 2)
    }

    print("📊 [Z-score 이상치 제거 결과]")
    print(f"제거 기준: |z| > {threshold}")
    print(f"이상치 제거 전 행 수: {len(df)}")
    print(f"이상치 제거 후 행 수: {len(clean_df)}")
    print(f"총 제거된 행 수: {removed_count}")
    print(f"제거 비율: {removal_rate:.2f}%\n")

    return clean_df, info


def clean_data(
    df: pd.DataFrame,
    num_cols: List[str],
    missing_method: str = "drop",
    outlier_threshold: float = 3.0,
    output_path: str = None
) -> Dict[str, Any]:
    """
    데이터 정제 전체 파이프라인 (결측치 제거 + 이상치 제거)

    Args:
        df: 정제할 DataFrame
        num_cols: 이상치 제거 대상 수치 컬럼 리스트
        missing_method: 결측치 처리 방법 ('drop' 또는 'fill')
        outlier_threshold: Z-score 임계값
        output_path: 저장할 파일 경로 (선택)

    Returns:
        정제 결과 정보
    """
    print(f"원본 데이터 행 수: {len(df)}")
    print(f"원본 데이터 열 수: {df.shape[1]}\n")

    # 사본 생성
    base_df = df.copy()

    # 결측치 확인
    missing_info = check_missing_values(base_df)

    # 결측치 제거/대체
    df_no_missing, missing_result = remove_missing_values(base_df, method=missing_method)

    # 이상치 제거
    df_clean, outlier_result = remove_outliers_zscore(df_no_missing, num_cols, threshold=outlier_threshold)

    # 최종 데이터 정보 확인
    print("✅ [최종 정제 데이터 정보]")
    print(df_clean.info())
    print("\n결측치 개수:")
    print(df_clean.isna().sum())
    print("\n통계 요약:")
    print(df_clean[num_cols].describe().round(3))

    # 저장 (선택)
    if output_path:
        df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n[저장 완료] {output_path}")

    return {
        "original_rows": len(df),
        "final_rows": len(df_clean),
        "total_removed_rows": len(df) - len(df_clean),
        "missing_info": missing_info,
        "missing_result": missing_result,
        "outlier_result": outlier_result,
        "cleaned_dataframe": df_clean,
        "output_path": output_path
    }
