# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

경북대학교 종합설계프로젝트2 - 머신러닝 제조 분야 적용 연구(데이터 분석 및 시각화) 백엔드 서비스입니다. 전해탈지 공정의 품질 예측 및 데이터 분석을 위한 FastAPI 기반 API를 제공합니다.

참여 기업: 초록에이아이
팀원: 정유현, 이미진, 곽민서

Python 3.13 및 FastAPI 사용

## 개발 명령어

### 환경 설정
```bash
# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
pip install -r requirement.txt
```

### 애플리케이션 실행
```bash
# 개발 서버 (자동 리로드)
uvicorn app.main:app --reload

# 프로덕션 서버
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 린팅 및 포맷팅
현재 프로젝트에 린팅/포맷팅 도구가 설정되어 있지 않습니다.

## 아키텍처

### 애플리케이션 구조

```
app/
├── main.py              # FastAPI 애플리케이션 진입점
├── api/
│   └── v1/
│       └── qc.py        # 품질 관리 엔드포인트
└── services/
    └── ml_service.py    # ML 아티팩트 로딩 로직
```

### 주요 설계 패턴

**서비스 레이어 패턴**: ML 관련 파일 작업은 `app/services/ml_service.py`에 중앙화되어 있습니다. 모든 아티팩트 로딩 함수는 일관된 패턴을 따릅니다:
- `artifacts/` 디렉토리에서 파일 읽기
- 파일이 없을 때 적절한 처리 (`FileNotFoundError` 발생 또는 에러 딕셔너리 반환)
- JSON 직렬화에 적합한 딕셔너리 형태로 데이터 반환

**API 버저닝**: 엔드포인트는 `/api/v1/` 접두사로 버전 관리됩니다. 라우터는 `app/api/v1/qc.py`에 정의되고 main 앱에 prefix와 tags와 함께 포함됩니다.

### Artifacts 디렉토리

`artifacts/` 디렉토리는 사전 생성된 ML 모델 출력물을 포함합니다:
- `feature_importance_rf.csv` - Random Forest 특성 중요도 점수
- `confusion_matrix_rf.csv` - 혼동 행렬 데이터 (2x2: true_0/true_1 vs pred_0/pred_1)
- `classification_report_rf.json` - 분류 메트릭 (precision, recall, f1-score)
- `metrics_summary_randomforest.json` - 모델 성능 요약
- `safe_region_result.json` - 공정 안전 구간 분석 결과
- `combined_data.csv` - 통합 데이터셋 (425KB)

**중요**: 이 파일들은 API 엔드포인트에서 읽어서 사용하지만, 이 백엔드에서 생성하지 않습니다. ML 파이프라인에서 사전 생성하여 artifacts 디렉토리에 배치해야 합니다.

### 데이터 처리 참고사항

**특성 중요도 로딩** (`ml_service.py:9-24`): CSV 파일에 이름 없는 인덱스 컬럼("Unnamed: 0")이 있어서 이를 "feature"로 이름 변경하여 더 깔끔한 JSON 출력을 생성합니다.

**혼동 행렬 파싱** (`ml_service.py:26-46`): CSV는 "true_0", "true_1", "pred_0", "pred_1" 레이블을 사용하며, 이를 의미론적 키로 매핑합니다: `normal_to_normal`, `normal_to_defect`, `defect_to_normal`, `defect_to_defect`.

## API 엔드포인트

모든 엔드포인트는 `/api/v1/` 하위에 "Quality" 태그로 구성:

- `GET /api/v1/feature-importance` - RandomForest 특성 중요도 데이터 반환
- `GET /api/v1/confusion-matrix` - 의미론적 형식의 혼동 행렬 반환
- `GET /api/v1/classification-report-rf` - 분류 리포트 JSON 반환
- `GET /api/v1/safe-region` - 안전 구간 분석 결과 반환

루트 엔드포인트:
- `GET /` - API 실행 확인
- `GET /health` - 헬스 체크 엔드포인트

## 주의사항

- `app/api/v1/qc.py`의 38번, 52번 줄에서 `HTTPException`을 사용하지만 import하지 않았습니다. 필요시 `from fastapi import HTTPException`을 추가해야 합니다.
- 현재 데이터베이스가 설정되어 있지 않으며, 모든 데이터는 정적 아티팩트 파일에서 읽어옵니다.
- 환경 변수 및 비밀 정보는 `.gitignore` 패턴을 따릅니다 (`.env` 파일은 제외됨)
- `.gitignore`는 대부분의 아티팩트 파일 타입(CSV, PNG, TXT, JOBLIB)을 제외하지만, artifacts의 JSON 파일은 git에서 추적됩니다.
