# app/api/v1/qc.py
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.services.ml_service import (
    load_feature_importance,
    load_confusion_matrix,
    load_classification_report_rf,
    load_safe_region_result,
)
from app.services.data_service import process_uploaded_csv, get_sensor_file_info, delete_file
from app.services.analysis_service import run_analysis_pipeline
import os
import json

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
    csv_path = "artifacts/confusion_matrix_rf.csv"
    cm_data = load_confusion_matrix(csv_path)

    return {
        "confusion_matrix": cm_data
    }

@router.get("/classification-report-rf")
def get_classification_report_rf():
    """
    RandomForest 분류 리포트 JSON 반환 엔드포인트
    (artifacts/classification_report_rf.json)
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


@router.post("/upload-csv")
async def upload_csv_file(file: UploadFile = File(...)):
    """
    단일 CSV 파일 업로드 엔드포인트

    프론트엔드에서 CSV 파일 1개를 업로드하면 파일을 저장하고
    기본 분석 정보를 반환합니다.

    반환 정보에는 업로드된 파일이 센서 파일인지 여부와
    전체 센서 파일 개수 및 목록이 포함됩니다.
    """
    # CSV 파일인지 확인
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="CSV 파일만 업로드 가능합니다."
        )

    try:
        result = await process_uploaded_csv(file)
        return {
            "message": "파일 업로드 성공",
            "filename": file.filename,
            "analysis": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 처리 중 오류 발생: {str(e)}"
        )


@router.post("/upload-multiple-csv")
async def upload_multiple_csv_files(files: List[UploadFile] = File(...)):
    """
    여러 CSV 파일 일괄 업로드 엔드포인트

    프론트엔드에서 여러 CSV 파일을 한 번에 업로드합니다.
    각 파일을 개별적으로 처리하며, 일부 파일이 실패해도
    성공한 파일들은 정상적으로 저장됩니다.

    Returns:
        성공한 파일과 실패한 파일 정보를 포함한 응답
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="업로드할 파일이 없습니다."
        )

    successful_files = []
    failed_files = []

    for file in files:
        # CSV 파일만 처리
        if not file.filename.endswith('.csv'):
            failed_files.append({
                "filename": file.filename,
                "error": "CSV 파일만 업로드 가능합니다."
            })
            continue

        try:
            result = await process_uploaded_csv(file)
            successful_files.append({
                "filename": file.filename,
                "analysis": result
            })
        except Exception as e:
            failed_files.append({
                "filename": file.filename,
                "error": str(e)
            })

    # 응답 생성
    total_count = len(files)
    success_count = len(successful_files)
    failed_count = len(failed_files)

    response = {
        "message": f"총 {total_count}개 파일 중 {success_count}개 성공, {failed_count}개 실패",
        "total_files": total_count,
        "successful_count": success_count,
        "failed_count": failed_count,
        "successful_files": successful_files,
    }

    # 실패한 파일이 있으면 포함
    if failed_files:
        response["failed_files"] = failed_files

    # 모든 파일이 실패한 경우 에러 처리
    if success_count == 0:
        raise HTTPException(
            status_code=400,
            detail=response
        )

    return response


@router.get("/sensor-files")
def get_sensor_files():
    """
    업로드된 센서 파일 정보를 조회하는 엔드포인트

    파일명에 'error'가 포함되지 않은 CSV 파일을 센서 파일로 분류합니다.
    센서 파일 개수와 파일명 리스트를 반환합니다.
    """
    try:
        sensor_info = get_sensor_file_info()
        return sensor_info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"센서 파일 정보 조회 중 오류 발생: {str(e)}"
        )


@router.delete("/sensor-files/{filename}")
def delete_sensor_file(filename: str):
    """
    특정 파일을 삭제하는 엔드포인트

    프론트엔드에서 파일명을 전달하면 해당 파일을 삭제합니다.

    Args:
        filename: 삭제할 파일명 (예: uploaded_20251127_143022_sensor1.csv)

    Returns:
        삭제 결과 정보
    """
    try:
        result = delete_file(filename)
        return {
            "message": "파일 삭제 성공",
            **result
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=403,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 삭제 중 오류 발생: {str(e)}"
        )


# 분석 상태를 저장할 전역 변수 (실제로는 Redis 등 사용 권장)
analysis_status = {"status": "idle", "result": None}


def run_analysis_task():
    """백그라운드에서 분석 실행"""
    global analysis_status
    try:
        analysis_status["status"] = "running"
        result = run_analysis_pipeline(
            data_dir="data",
            output_dir="artifacts"
        )
        analysis_status["status"] = "completed"
        analysis_status["result"] = result
    except Exception as e:
        analysis_status["status"] = "error"
        analysis_status["result"] = {"error": str(e)}


@router.post("/start-analysis")
async def start_analysis(background_tasks: BackgroundTasks):
    """
    데이터 분석을 시작하는 엔드포인트

    백그라운드에서 전체 분석 파이프라인을 실행합니다:
    1. 데이터 전처리 및 병합
    2. 데이터 정제
    3. 품질 평가
    4. EDA 리포트 생성
    5. 모델 학습 및 평가
    6. 안전 영역 분석
    """
    global analysis_status

    if analysis_status["status"] == "running":
        raise HTTPException(
            status_code=400,
            detail="분석이 이미 실행 중입니다."
        )

    background_tasks.add_task(run_analysis_task)

    return {
        "message": "분석이 시작되었습니다.",
        "status": "running"
    }


@router.get("/analysis-status")
def get_analysis_status():
    """
    분석 상태를 조회하는 엔드포인트

    Returns:
        - status: idle, running, completed, error
        - result: 분석 완료 시 결과 반환
    """
    return analysis_status