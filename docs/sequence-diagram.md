# 시퀀스 다이어그램

## 전체 워크플로우: CSV 업로드부터 결과 조회까지

```mermaid
sequenceDiagram
    actor User as 사용자
    participant FE as Frontend
    participant API as FastAPI
    participant FilesAPI as Files API
    participant AnalysisAPI as Analysis API
    participant ResultsAPI as Results API
    participant DataSvc as data_service
    participant AnalysisSvc as analysis_service
    participant MLSvc as ml_service
    participant FS as File System

    Note over User,FS: 1️⃣ CSV 파일 업로드 단계
    User->>FE: CSV 파일 선택
    FE->>+API: POST /api/v1/upload-csv
    API->>+FilesAPI: upload_csv_file()
    FilesAPI->>+DataSvc: process_uploaded_csv()
    DataSvc->>FS: 파일 저장 (data/)
    FS-->>DataSvc: 저장 완료
    DataSvc->>DataSvc: 기본 분석 (행/열 수, 통계)
    DataSvc-->>-FilesAPI: 분석 결과 반환
    FilesAPI-->>-API: 200 OK + 파일 정보
    API-->>-FE: 업로드 성공 응답
    FE-->>User: 업로드 완료 표시

    Note over User,FS: 2️⃣ 데이터 분석 실행 단계
    User->>FE: "분석 시작" 버튼 클릭
    FE->>+API: POST /api/v1/start-analysis
    API->>+AnalysisAPI: start_analysis()
    AnalysisAPI->>AnalysisAPI: 상태 확인 (이미 실행 중?)
    AnalysisAPI->>AnalysisAPI: 백그라운드 작업 등록
    AnalysisAPI-->>-API: 200 OK (분석 시작됨)
    API-->>-FE: {"status": "running"}
    FE-->>User: "분석 중..." 표시

    Note over AnalysisAPI,FS: 🔄 백그라운드에서 분석 실행 (비동기)
    AnalysisAPI->>+AnalysisSvc: run_analysis_pipeline()

    AnalysisSvc->>FS: data/ 폴더에서 CSV 읽기
    FS-->>AnalysisSvc: CSV 데이터

    AnalysisSvc->>AnalysisSvc: 1. 데이터 전처리 & 병합
    AnalysisSvc->>AnalysisSvc: 2. 데이터 정제 (결측치, 이상치)
    AnalysisSvc->>AnalysisSvc: 3. 품질 평가 (DQI)
    AnalysisSvc->>AnalysisSvc: 4. EDA 리포트 생성
    AnalysisSvc->>AnalysisSvc: 5. 모델 학습 (RandomForest)
    AnalysisSvc->>AnalysisSvc: 6. 모델 평가
    AnalysisSvc->>AnalysisSvc: 7. 안전 영역 분석

    AnalysisSvc->>FS: artifacts/ 폴더에 결과 저장
    Note right of FS: confusion_matrix_rf.csv<br/>feature_importance_rf.csv<br/>classification_report_rf.json<br/>safe_region_result.json
    FS-->>AnalysisSvc: 저장 완료

    AnalysisSvc-->>-AnalysisAPI: 분석 완료 + 요약 결과
    AnalysisAPI->>AnalysisAPI: 상태 업데이트 (completed)

    Note over User,FS: 3️⃣ 분석 상태 조회 단계 (폴링)
    loop 주기적으로 상태 확인
        FE->>+API: GET /api/v1/analysis-status
        API->>+AnalysisAPI: get_analysis_status()
        AnalysisAPI-->>-API: {"status": "running/completed"}
        API-->>-FE: 현재 상태
        alt 분석 완료
            FE-->>User: "분석 완료!" 표시
        else 아직 실행 중
            FE-->>User: 진행 중 표시
            Note over FE: 3초 대기 후 재시도
        end
    end

    Note over User,FS: 4️⃣ 분석 결과 조회 단계
    User->>FE: 결과 화면 이동

    par 병렬로 여러 결과 조회
        FE->>+API: GET /api/v1/feature-importance
        API->>+ResultsAPI: get_feature_importance()
        ResultsAPI->>+MLSvc: load_feature_importance()
        MLSvc->>FS: artifacts/feature_importance_rf.csv 읽기
        FS-->>MLSvc: CSV 데이터
        MLSvc-->>-ResultsAPI: JSON 변환 후 반환
        ResultsAPI-->>-API: 200 OK + 데이터
        API-->>-FE: 특성 중요도 데이터
        FE-->>User: 차트 표시
    and
        FE->>+API: GET /api/v1/confusion-matrix
        API->>+ResultsAPI: get_confusion_matrix()
        ResultsAPI->>+MLSvc: load_confusion_matrix()
        MLSvc->>FS: artifacts/confusion_matrix_rf.csv 읽기
        FS-->>MLSvc: CSV 데이터
        MLSvc-->>-ResultsAPI: JSON 변환 후 반환
        ResultsAPI-->>-API: 200 OK + 데이터
        API-->>-FE: 혼동 행렬 데이터
        FE-->>User: 혼동 행렬 표시
    and
        FE->>+API: GET /api/v1/classification-report-rf
        API->>+ResultsAPI: get_classification_report_rf()
        ResultsAPI->>+MLSvc: load_classification_report_rf()
        MLSvc->>FS: artifacts/classification_report_rf.json 읽기
        FS-->>MLSvc: JSON 데이터
        MLSvc-->>-ResultsAPI: JSON 반환
        ResultsAPI-->>-API: 200 OK + 데이터
        API-->>-FE: 분류 리포트
        FE-->>User: 테이블 표시
    and
        FE->>+API: GET /api/v1/safe-region
        API->>+ResultsAPI: get_safe_region_result()
        ResultsAPI->>+MLSvc: load_safe_region_result()
        MLSvc->>FS: artifacts/safe_region_result.json 읽기
        FS-->>MLSvc: JSON 데이터
        MLSvc-->>-ResultsAPI: JSON 반환
        ResultsAPI-->>-API: 200 OK + 데이터
        API-->>-FE: 안전 영역 데이터
        FE-->>User: 안전 범위 표시
    end

    Note over User,FS: ✅ 전체 워크플로우 완료
```

---

## 상세 시퀀스 다이어그램 (개별)

### 1. CSV 업로드 플로우

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend
    participant API as Files API
    participant Svc as data_service
    participant FS as data/ 폴더

    User->>FE: CSV 파일 선택
    FE->>+API: POST /api/v1/upload-csv
    Note right of API: Content-Type:<br/>multipart/form-data

    API->>API: 파일 확장자 검증 (.csv)

    API->>+Svc: process_uploaded_csv(file)

    Svc->>Svc: 여러 인코딩 시도<br/>(utf-8, cp949, euc-kr)
    Svc->>Svc: pandas로 CSV 파싱
    Svc->>Svc: 타임스탬프 생성

    Svc->>FS: uploaded_YYYYMMDD_HHMMSS_filename.csv 저장
    FS-->>Svc: 저장 완료

    Svc->>Svc: 센서 파일 여부 확인<br/>(파일명에 'error' 없음)
    Svc->>FS: data/ 전체 스캔
    FS-->>Svc: 센서 파일 목록

    Svc->>Svc: 기본 분석 수행<br/>(행/열 수, 결측치, 통계)

    Svc-->>-API: 분석 결과 반환

    API-->>-FE: 200 OK
    Note right of FE: {<br/>  "message": "파일 업로드 성공",<br/>  "filename": "...",<br/>  "analysis": {...}<br/>}

    FE-->>User: 업로드 완료 + 미리보기
```

### 2. 분석 실행 플로우 (백그라운드)

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant API as Analysis API
    participant BG as 백그라운드 작업
    participant Svc as analysis_service
    participant Domain as analysis/*<br/>(도메인 로직)
    participant FS as 파일 시스템

    FE->>+API: POST /api/v1/start-analysis

    API->>API: 현재 상태 확인
    alt 이미 실행 중
        API-->>FE: 400 Bad Request<br/>"분석이 이미 실행 중입니다"
    else idle 상태
        API->>BG: run_analysis_task() 등록
        API->>API: 상태 = "running"
        API-->>-FE: 200 OK {"status": "running"}

        Note over BG,FS: 🔄 비동기로 백그라운드 실행

        activate BG
        BG->>+Svc: run_analysis_pipeline(<br/>  data_dir="data",<br/>  output_dir="artifacts"<br/>)

        Note over Svc,Domain: 1️⃣ 데이터 전처리
        Svc->>Domain: preprocess_and_merge_sensors()
        Domain->>FS: data/*.csv 파일들 읽기
        FS-->>Domain: 센서 CSV 데이터
        Domain->>Domain: 컬럼 정규화, 병합
        Domain->>FS: artifacts/combined_data.csv 저장
        Domain-->>Svc: 병합 완료

        Note over Svc,Domain: 2️⃣ 데이터 정제
        Svc->>Domain: clean_data()
        Domain->>Domain: 결측치 제거
        Domain->>Domain: Z-score 이상치 제거
        Domain->>FS: artifacts/cleaned_data.csv 저장
        Domain-->>Svc: 정제 완료

        Note over Svc,Domain: 3️⃣ 품질 평가
        Svc->>Domain: evaluate_data_quality()
        Domain->>Domain: DQI 계산
        Domain-->>Svc: 품질 지수 반환

        Note over Svc,Domain: 4️⃣ EDA 리포트
        Svc->>Domain: generate_eda_report()
        Domain->>Domain: 시각화 생성
        Domain->>FS: artifacts/eda/*.png 저장
        Domain-->>Svc: EDA 완료

        Note over Svc,Domain: 5️⃣ 모델 학습
        Svc->>Domain: train_and_evaluate_rf()
        Domain->>Domain: 데이터 분할 (train/test)
        Domain->>Domain: StandardScaler 적용
        Domain->>Domain: RandomForest 학습
        Domain->>Domain: 모델 평가
        Domain-->>Svc: 모델 & 평가 결과

        Note over Svc,Domain: 6️⃣ 결과 저장
        Svc->>Domain: save_model_artifacts()
        Domain->>FS: confusion_matrix_rf.csv
        Domain->>FS: feature_importance_rf.csv
        Domain->>FS: classification_report_rf.json
        Domain->>FS: metrics_summary_randomforest.json
        Domain->>FS: model_rf.joblib
        Domain-->>Svc: 저장 완료

        Note over Svc,Domain: 7️⃣ 안전 영역 분석
        Svc->>Domain: estimate_safe_region()
        Domain->>Domain: 격자점 생성 & 예측
        Domain->>Domain: 안전 영역 계산
        Domain->>FS: safe_region_result.json
        Domain-->>Svc: 안전 영역 결과

        Svc-->>-BG: 분석 완료 + 요약

        BG->>API: 상태 = "completed"
        BG->>API: result = {...}
        deactivate BG
    end
```

### 3. 결과 조회 플로우

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant API as Results API
    participant MLSvc as ml_service
    participant FS as artifacts/

    Note over FE: 사용자가 결과 화면 진입

    FE->>+API: GET /api/v1/feature-importance
    API->>+MLSvc: load_feature_importance()
    MLSvc->>FS: artifacts/feature_importance_rf.csv

    alt 파일 존재
        FS-->>MLSvc: CSV 데이터
        MLSvc->>MLSvc: pandas로 읽기
        MLSvc->>MLSvc: 인덱스 → "feature" 컬럼 변환
        MLSvc->>MLSvc: dict로 변환
        MLSvc-->>-API: [{"feature": "temp", "importance": 0.45}, ...]
        API-->>-FE: 200 OK + JSON 데이터
        FE->>FE: 차트 렌더링
    else 파일 없음
        FS-->>MLSvc: FileNotFoundError
        MLSvc-->>API: {"error": "file not found"}
        API-->>FE: 200 OK (에러 포함)
        FE->>FE: 에러 메시지 표시
    end
```

---

## 에러 처리 플로우

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant API as Analysis API
    participant BG as 백그라운드
    participant Svc as analysis_service

    FE->>+API: POST /api/v1/start-analysis
    API->>BG: 분석 시작
    API-->>-FE: 200 OK {"status": "running"}

    activate BG
    BG->>+Svc: run_analysis_pipeline()

    Svc->>Svc: 분석 진행 중...

    alt 에러 발생 (예: 파일 없음)
        Svc-->>-BG: Exception 발생
        BG->>API: 상태 = "error"
        BG->>API: result = {"error": "..."}
        deactivate BG
    end

    loop 상태 조회
        FE->>+API: GET /api/v1/analysis-status
        API-->>-FE: {"status": "error", "result": {"error": "..."}}
    end

    FE->>FE: 에러 메시지 표시
    FE->>FE: 재시도 버튼 활성화
```

---

## 다이어그램 렌더링 방법

### 1. GitHub에서 보기
- 이 파일을 GitHub에 push하면 자동으로 렌더링됩니다

### 2. VS Code에서 보기
- Mermaid 플러그인 설치: `Markdown Preview Mermaid Support`
- 마크다운 미리보기 열기 (Cmd+Shift+V)

### 3. 온라인 에디터
- https://mermaid.live/ 에서 코드 붙여넣기

### 4. 이미지로 변환
```bash
# mermaid-cli 설치
npm install -g @mermaid-js/mermaid-cli

# PNG로 변환
mmdc -i sequence-diagram.md -o sequence-diagram.png
```
