# app/api/v1/analysis.py
"""
데이터 분석 실행 및 상태 관리 엔드포인트
전체 분석 파이프라인을 백그라운드에서 실행하고 상태를 조회합니다.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.services.analysis_service import run_analysis_pipeline

router = APIRouter()

# 분석 상태를 저장할 전역 변수 
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
