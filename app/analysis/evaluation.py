# app/analysis/evaluation.py
# 데이터 품질 평가 및 안전 영역 분석 - DAI 계산, 안전 영역 추정 로직

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


def compute_quality_indices(
    df: pd.DataFrame,
    cols: List[str],
    valid_ranges: Optional[Dict[str, tuple]] = None
) -> pd.DataFrame:
    """
    데이터 품질 지수 계산 (CQI, UQI, VQI, DQI)

    Args:
        df: 평가할 데이터프레임
        cols: 평가할 컬럼 리스트 (예: ["temp", "humid", "press"])
        valid_ranges: {컬럼명: (min, max)} 딕셔너리. 없으면 IQR 기반 범위로 자동 설정

    Returns:
        품질 지수가 포함된 DataFrame

    품질 지수:
        - CQI (Completeness Quality Index): 완전성 지수 (결측치가 없는 정도)
        - UQI (Uniqueness Quality Index): 고유성 지수 (중복되지 않은 값의 비율)
        - VQI (Validity Quality Index): 유효성 지수 (유효 범위 내 값의 비율)
        - DQI (Data Quality Index): 데이터 품질 지수 (위 3개의 평균)
    """
    results = []

    for col in cols:
        series = df[col]
        total_count = len(series)
        missing_count = series.isnull().sum()
        unique_count = series.nunique(dropna=True)

        # (1) Completeness Quality Index (CQI)
        cqi = (1 - missing_count / total_count) * 100 if total_count > 0 else 0

        # (2) Uniqueness Quality Index (UQI)
        uqi = (unique_count / total_count) * 100 if total_count > 0 else 0

        # (3) Validity Quality Index (VQI)
        if valid_ranges and col in valid_ranges:
            vmin, vmax = valid_ranges[col]
        else:
            # IQR 기반 유효 범위 설정
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            vmin, vmax = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        valid_count = series.between(vmin, vmax).sum()
        vqi = (valid_count / total_count) * 100 if total_count > 0 else 0

        # (4) Data Quality Index (DQI) = 세 지수 평균
        dqi = (cqi + uqi + vqi) / 3

        results.append({
            "column": col,
            "CQI": round(cqi, 2),
            "UQI": round(uqi, 2),
            "VQI": round(vqi, 2),
            "DQI": round(dqi, 2),
            "valid_range": f"[{vmin:.2f}, {vmax:.2f}]"
        })

    return pd.DataFrame(results)


def evaluate_data_quality(
    df: pd.DataFrame,
    num_cols: List[str],
    valid_ranges: Optional[Dict[str, tuple]] = None
) -> Dict[str, Any]:
    """
    데이터 품질 평가 전체 파이프라인

    Args:
        df: 평가할 데이터프레임
        num_cols: 평가할 수치 컬럼 리스트
        valid_ranges: 유효 범위 딕셔너리 (선택)

    Returns:
        품질 평가 결과
    """
    # 품질 지수 계산
    quality_df = compute_quality_indices(df, num_cols, valid_ranges)

    print("📊 [데이터 품질 지수 (CQI / UQI / VQI / DQI)]")
    print(quality_df)

    # 전체 평균 DQI 계산
    overall_dqi = quality_df["DQI"].mean()
    print(f"\n➡ 평균 데이터 품질 지수 (DQI): {overall_dqi:.2f} / 100")

    return {
        "quality_indices": quality_df.to_dict(orient="records"),
        "overall_dqi": round(overall_dqi, 2),
        "evaluation_summary": {
            "avg_cqi": round(quality_df["CQI"].mean(), 2),
            "avg_uqi": round(quality_df["UQI"].mean(), 2),
            "avg_vqi": round(quality_df["VQI"].mean(), 2),
            "avg_dqi": round(overall_dqi, 2)
        }
    }


def get_quality_grade(dqi: float) -> str:
    """
    DQI 점수를 기반으로 품질 등급 반환

    Args:
        dqi: 데이터 품질 지수 (0-100)

    Returns:
        품질 등급 (Excellent, Good, Fair, Poor)
    """
    if dqi >= 90:
        return "Excellent"
    elif dqi >= 75:
        return "Good"
    elif dqi >= 60:
        return "Fair"
    else:
        return "Poor"


def generate_quality_report(
    df: pd.DataFrame,
    num_cols: List[str],
    valid_ranges: Optional[Dict[str, tuple]] = None
) -> Dict[str, Any]:
    """
    데이터 품질 리포트 생성

    Args:
        df: 평가할 데이터프레임
        num_cols: 평가할 수치 컬럼 리스트
        valid_ranges: 유효 범위 딕셔너리 (선택)

    Returns:
        품질 리포트
    """
    # 품질 평가
    evaluation = evaluate_data_quality(df, num_cols, valid_ranges)

    # 품질 등급 결정
    overall_dqi = evaluation["overall_dqi"]
    quality_grade = get_quality_grade(overall_dqi)

    # 컬럼별 품질 등급
    column_grades = {}
    for item in evaluation["quality_indices"]:
        col_name = item["column"]
        col_dqi = item["DQI"]
        column_grades[col_name] = {
            "dqi": col_dqi,
            "grade": get_quality_grade(col_dqi)
        }

    report = {
        "dataset_info": {
            "total_rows": len(df),
            "total_columns": len(num_cols),
            "columns_evaluated": num_cols
        },
        "overall_quality": {
            "dqi": overall_dqi,
            "grade": quality_grade
        },
        "quality_indices": evaluation["quality_indices"],
        "evaluation_summary": evaluation["evaluation_summary"],
        "column_grades": column_grades
    }

    print("\n" + "="*50)
    print("📋 데이터 품질 리포트")
    print("="*50)
    print(f"전체 품질 점수: {overall_dqi:.2f} / 100")
    print(f"품질 등급: {quality_grade}")
    print("\n컬럼별 품질 등급:")
    for col, grade_info in column_grades.items():
        print(f"  - {col}: {grade_info['dqi']:.2f} ({grade_info['grade']})")
    print("="*50)

    return report


# =========================================
#  안전 영역(Safe Region) 추정
# =========================================

def estimate_safe_region(
    model,
    scaler,
    temp_range: tuple,
    humid_range: tuple,
    press_range: tuple,
    n_temp: int = 20,
    n_humid: int = 20,
    n_press: int = 10,
    prob_threshold: float = 0.05
) -> tuple:
    """
    temp/humid/press 범위 안에서 격자를 만들어
    모델이 예측한 불량 확률 < prob_threshold인 점들만 모아서
    안전 영역(min/max)을 추정

    Args:
        model: 학습된 모델
        scaler: 데이터 스케일러
        temp_range: 온도 범위 (min, max)
        humid_range: 습도 범위 (min, max)
        press_range: 압력 범위 (min, max)
        n_temp: 온도 격자 개수
        n_humid: 습도 격자 개수
        n_press: 압력 격자 개수
        prob_threshold: 불량 확률 임계값 (기본값: 0.05)

    Returns:
        (safe_summary, safe_points) - 안전 영역 요약 정보와 안전 격자점들
    """
    # 격자 생성
    temps = np.linspace(*temp_range, n_temp)
    humids = np.linspace(*humid_range, n_humid)
    presses = np.linspace(*press_range, n_press)

    grid = []
    for t in temps:
        for h in humids:
            for p in presses:
                grid.append([t, h, p])
    grid = np.array(grid)

    print(f"🔍 안전 영역 추정 중...")
    print(f"격자점 개수: {len(grid)}")
    print(f"불량 확률 임계값: {prob_threshold}")

    # feature name을 가진 DataFrame으로 변환 후 transform
    grid_df = pd.DataFrame(grid, columns=["temp", "humid", "press"])
    grid_scaled = scaler.transform(grid_df)

    # 불량 확률 예측
    probs = model.predict_proba(grid_scaled)[:, 1]

    # 안전 영역 필터링
    safe_mask = probs < prob_threshold
    safe_points = grid[safe_mask]

    if len(safe_points) == 0:
        print("⚠️ 이 범위 안에서는 안전 영역을 찾지 못했습니다.")
        return None, None

    # 안전 영역 요약
    safe_summary = {
        "temp_min": float(safe_points[:, 0].min()),
        "temp_max": float(safe_points[:, 0].max()),
        "humid_min": float(safe_points[:, 1].min()),
        "humid_max": float(safe_points[:, 1].max()),
        "press_min": float(safe_points[:, 2].min()),
        "press_max": float(safe_points[:, 2].max()),
        "n_safe_points": int(len(safe_points)),
        "n_total_points": int(len(grid)),
        "safe_ratio": round(len(safe_points) / len(grid) * 100, 2)
    }

    print(f"\n✅ 안전 영역 추정 완료")
    print(f"안전 격자점 개수: {safe_summary['n_safe_points']} / {safe_summary['n_total_points']}")
    print(f"안전 비율: {safe_summary['safe_ratio']}%")
    print(f"\n안전 영역 범위:")
    print(f"  온도(temp):  [{safe_summary['temp_min']:.2f}, {safe_summary['temp_max']:.2f}]")
    print(f"  습도(humid): [{safe_summary['humid_min']:.2f}, {safe_summary['humid_max']:.2f}]")
    print(f"  압력(press): [{safe_summary['press_min']:.2f}, {safe_summary['press_max']:.2f}]")

    return safe_summary, safe_points


def print_safe_summary(
    safe_summary: Optional[Dict[str, Any]],
    prob_threshold: float = 0.05
) -> None:
    """
    안전 영역 요약 정보를 출력

    Args:
        safe_summary: 안전 영역 요약 정보
        prob_threshold: 불량 확률 임계값
    """
    if safe_summary is None:
        print("⚠️ 이 범위 안에서는 안전 운전 범위를 찾지 못했습니다.")
        return

    print("\n" + "="*60)
    print("📊 안전 영역 요약")
    print("="*60)
    print(f"불량 확률 임계값: {prob_threshold}")
    print(f"\n안전 영역 범위:")
    print(f"  온도(temp):  [{safe_summary['temp_min']:.2f}, {safe_summary['temp_max']:.2f}]")
    print(f"  습도(humid): [{safe_summary['humid_min']:.2f}, {safe_summary['humid_max']:.2f}]")
    print(f"  압력(press): [{safe_summary['press_min']:.2f}, {safe_summary['press_max']:.2f}]")
    print(f"\n안전 격자점: {safe_summary['n_safe_points']}")
    if "n_total_points" in safe_summary:
        print(f"전체 격자점: {safe_summary['n_total_points']}")
        print(f"안전 비율: {safe_summary.get('safe_ratio', 0):.2f}%")
    print("="*60)


def safe_summary_to_json(
    safe_summary: Optional[Dict[str, Any]],
    prob_threshold: float = 0.05
) -> str:
    """
    안전 영역 요약 정보를 JSON 문자열로 변환

    Args:
        safe_summary: 안전 영역 요약 정보
        prob_threshold: 불량 확률 임계값

    Returns:
        JSON 문자열
    """
    import json

    if safe_summary is None:
        return json.dumps({
            "status": "no_safe_region",
            "message": "이 범위 안에서는 안전 운전 범위를 찾지 못했습니다."
        }, ensure_ascii=False)

    result = {
        "status": "success",
        "prob_threshold": prob_threshold,  # 불량 확률 기준
        "safe_range": {
            "temp": {
                "min": safe_summary["temp_min"],
                "max": safe_summary["temp_max"],
            },
            "humid": {
                "min": safe_summary["humid_min"],
                "max": safe_summary["humid_max"],
            },
            "press": {
                "min": safe_summary["press_min"],
                "max": safe_summary["press_max"],
            },
        },
        "n_safe_points": safe_summary["n_safe_points"],
        "n_total_points": safe_summary.get("n_total_points", 0),
        "safe_ratio": safe_summary.get("safe_ratio", 0)
    }
    # 문자열(JSON)로 반환
    return json.dumps(result, ensure_ascii=False, indent=2)


def save_safe_region_result(
    safe_summary: Dict[str, Any],
    output_path: str = "artifacts/safe_region_result.json"
) -> str:
    """
    안전 영역 결과를 JSON 파일로 저장 (원본 형식)

    Args:
        safe_summary: 안전 영역 요약 정보
        output_path: 저장 경로

    Returns:
        저장된 파일 경로
    """
    import json
    import os

    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[저장 완료] {output_path}")
    return output_path


def save_safe_region_json(
    safe_summary: Optional[Dict[str, Any]],
    prob_threshold: float = 0.05,
    output_dir: str = "artifacts",
    filename: str = "safe_region_result.json"
) -> str:
    """
    안전 영역 결과를 구조화된 JSON 파일로 저장

    Args:
        safe_summary: 안전 영역 요약 정보
        prob_threshold: 불량 확률 임계값
        output_dir: 출력 디렉토리
        filename: 파일명

    Returns:
        저장된 파일 경로
    """
    import os

    # JSON 문자열 생성
    safe_json = safe_summary_to_json(safe_summary, prob_threshold)

    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 파일 경로
    output_path = os.path.join(output_dir, filename)

    # JSON 파일로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(safe_json)

    print(f"[저장 완료] {filename}")
    return output_path


def analyze_safe_region(
    model,
    scaler,
    df: pd.DataFrame,
    feature_cols: List[str] = ["temp", "humid", "press"],
    prob_threshold: float = 0.05,
    output_path: str = "artifacts/safe_region_result.json"
) -> Dict[str, Any]:
    """
    데이터프레임 기반으로 안전 영역 분석 전체 파이프라인

    Args:
        model: 학습된 모델
        scaler: 데이터 스케일러
        df: 분석할 데이터프레임
        feature_cols: 특성 컬럼 리스트
        prob_threshold: 불량 확률 임계값
        output_path: 결과 저장 경로

    Returns:
        안전 영역 분석 결과
    """
    print("\n" + "="*60)
    print("🔍 안전 영역 분석 시작")
    print("="*60)

    # 각 특성의 범위를 데이터에서 추출
    temp_range = (df["temp"].min(), df["temp"].max())
    humid_range = (df["humid"].min(), df["humid"].max())
    press_range = (df["press"].min(), df["press"].max())

    print(f"\n데이터 범위:")
    print(f"  온도(temp):  [{temp_range[0]:.2f}, {temp_range[1]:.2f}]")
    print(f"  습도(humid): [{humid_range[0]:.2f}, {humid_range[1]:.2f}]")
    print(f"  압력(press): [{press_range[0]:.2f}, {press_range[1]:.2f}]\n")

    # 안전 영역 추정
    safe_summary, safe_points = estimate_safe_region(
        model=model,
        scaler=scaler,
        temp_range=temp_range,
        humid_range=humid_range,
        press_range=press_range,
        prob_threshold=prob_threshold
    )

    if safe_summary is None:
        return {"error": "안전 영역을 찾지 못했습니다."}

    # 결과 저장
    save_path = save_safe_region_result(safe_summary, output_path)

    print("\n" + "="*60)
    print("✅ 안전 영역 분석 완료")
    print("="*60)

    return {
        "safe_summary": safe_summary,
        "safe_points_count": len(safe_points),
        "output_path": save_path
    }
