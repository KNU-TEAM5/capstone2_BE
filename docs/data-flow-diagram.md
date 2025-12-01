# 데이터 플로우 다이어그램

전해탈지 공정 데이터가 수집되어 ML 분석을 거쳐 최종 사용자에게 전달되는 전체 흐름을 나타냅니다.

## 1. 전체 데이터 파이프라인

```mermaid
flowchart LR
    subgraph "데이터 수집"
        RAW[원본 센서 데이터<br/>CSV 파일]
        ERROR[에러 로그<br/>CSV 파일]
    end

    subgraph "데이터 업로드 (백엔드)"
        UPLOAD[파일 업로드<br/>POST /upload-csv]
        STORAGE[(data/ 디렉토리)]
    end

    subgraph "ML Pipeline (외부/백그라운드)"
        PREPROCESS[데이터 전처리<br/>병합 및 정제]
        QUALITY[품질 평가<br/>불량/정상 분류]
        TRAIN[모델 학습<br/>RandomForest]
        EVAL[모델 평가<br/>성능 분석]
        SAFE[안전 영역 분석<br/>공정 최적화]
    end

    subgraph "Artifacts 저장"
        ART[(artifacts/<br/>ML 결과물)]
    end

    subgraph "API 서비스"
        API[FastAPI 백엔드<br/>GET /api/v1/*]
    end

    subgraph "데이터 시각화"
        DASH[프론트엔드<br/>대시보드]
        CHART[차트/그래프<br/>인사이트]
    end

    %% 데이터 흐름
    RAW --> UPLOAD
    ERROR --> UPLOAD
    UPLOAD --> STORAGE

    STORAGE --> PREPROCESS
    PREPROCESS --> QUALITY
    QUALITY --> TRAIN
    TRAIN --> EVAL
    EVAL --> SAFE
    SAFE --> ART

    ART --> API
    API --> DASH
    DASH --> CHART

    style RAW fill:#e3f2fd
    style ERROR fill:#ffebee
    style UPLOAD fill:#fff3e0
    style STORAGE fill:#f3e5f5
    style PREPROCESS fill:#e8f5e9
    style QUALITY fill:#e8f5e9
    style TRAIN fill:#e8f5e9
    style EVAL fill:#e8f5e9
    style SAFE fill:#e8f5e9
    style ART fill:#fff9c4
    style API fill:#ffe0b2
    style DASH fill:#e1f5fe
    style CHART fill:#e1f5fe
```

## 2. API 요청 플로우 (상세)

### 2.1 ML 아티팩트 조회 플로우

```mermaid
sequenceDiagram
    participant User as 사용자
    participant FE as 프론트엔드
    participant API as FastAPI
    participant Router as QC Router
    participant Service as ML Service
    participant FS as Artifacts 파일

    User->>FE: 데이터 조회 요청
    FE->>API: GET /api/v1/feature-importance
    API->>Router: 라우팅
    Router->>Service: load_feature_importance()
    Service->>FS: feature_importance_rf.csv 읽기
    FS-->>Service: CSV 데이터
    Service->>Service: DataFrame → dict 변환
    Service-->>Router: JSON 데이터
    Router-->>API: HTTP Response
    API-->>FE: JSON Response
    FE->>FE: 차트 렌더링
    FE-->>User: 시각화 표시

    Note over Service,FS: 동일한 플로우:<br/>confusion-matrix<br/>classification-report<br/>safe-region
```

### 2.2 CSV 파일 업로드 플로우

```mermaid
sequenceDiagram
    participant User as 사용자
    participant FE as 프론트엔드
    participant API as FastAPI
    participant Router as QC Router
    participant Service as Data Service
    participant FS as data/ 디렉토리

    User->>FE: CSV 파일 선택
    FE->>API: POST /api/v1/upload-csv<br/>(multipart/form-data)
    API->>Router: 라우팅
    Router->>Router: CSV 파일 검증
    Router->>Service: process_uploaded_csv()
    Service->>Service: 파일명 생성<br/>(timestamp 추가)
    Service->>FS: 파일 저장
    Service->>Service: 센서 파일 분석<br/>(error 포함 여부 확인)
    Service-->>Router: 업로드 결과
    Router-->>API: HTTP Response
    API-->>FE: JSON Response<br/>(filename, analysis)
    FE-->>User: 업로드 성공 메시지
```

### 2.3 백그라운드 분석 실행 플로우

```mermaid
sequenceDiagram
    participant User as 사용자
    participant FE as 프론트엔드
    participant API as FastAPI
    participant Router as QC Router
    participant BG as Background Task
    participant Analysis as Analysis Service
    participant Data as data/ 디렉토리
    participant Artifacts as artifacts/ 디렉토리

    User->>FE: 분석 시작 요청
    FE->>API: POST /api/v1/start-analysis
    API->>Router: 라우팅
    Router->>Router: 분석 상태 확인<br/>(idle/running)
    Router->>BG: BackgroundTask 등록
    Router-->>API: 즉시 응답
    API-->>FE: {"status": "running"}
    FE-->>User: 분석 시작 알림

    par 백그라운드 실행
        BG->>Analysis: run_analysis_pipeline()
        Analysis->>Data: 업로드된 CSV 파일 읽기
        Analysis->>Analysis: 데이터 전처리 및 병합
        Analysis->>Analysis: 품질 평가
        Analysis->>Analysis: 모델 학습 (RandomForest)
        Analysis->>Analysis: 모델 평가
        Analysis->>Analysis: 안전 영역 분석
        Analysis->>Artifacts: 결과물 저장<br/>(CSV, JSON)
        Analysis-->>BG: 분석 완료
    end

    loop 상태 폴링
        FE->>API: GET /api/v1/analysis-status
        API->>Router: 라우팅
        Router-->>API: {"status": "running"}
        API-->>FE: 현재 상태
    end

    FE->>API: GET /api/v1/analysis-status
    API->>Router: 라우팅
    Router-->>API: {"status": "completed", "result": {...}}
    API-->>FE: 분석 완료
    FE-->>User: 완료 알림 + 결과 표시
```

## 3. 데이터 변환 과정

### 3.1 특성 중요도 (Feature Importance)

```mermaid
graph LR
    A[CSV 파일<br/>feature_importance_rf.csv] --> B[Pandas DataFrame]
    B --> C[인덱스 리셋]
    C --> D[컬럼명 변경<br/>Unnamed:0 → feature]
    D --> E[dict 변환<br/>orient=records]
    E --> F[JSON Response<br/>feature, importance]

    style A fill:#e3f2fd
    style F fill:#c8e6c9
```

**예시 변환:**
```
CSV:                        JSON:
Unnamed: 0, importance  →  [
current, 0.45              {"feature": "current", "importance": 0.45},
voltage, 0.32              {"feature": "voltage", "importance": 0.32},
temp, 0.23                 {"feature": "temp", "importance": 0.23}
                           ]
```

### 3.2 혼동 행렬 (Confusion Matrix)

```mermaid
graph LR
    A[CSV 파일<br/>confusion_matrix_rf.csv] --> B[Pandas DataFrame]
    B --> C[인덱스 설정<br/>true_0, true_1]
    C --> D[값 추출<br/>tn, fp, fn, tp]
    D --> E[의미론적 키 매핑]
    E --> F[JSON Response<br/>normal/defect]

    style A fill:#e3f2fd
    style F fill:#c8e6c9
```

**예시 변환:**
```
CSV:                          JSON:
       pred_0  pred_1     →  {
true_0   850     15           "normal_to_normal": 850,
true_1    12     98           "normal_to_defect": 15,
                              "defect_to_normal": 12,
                              "defect_to_defect": 98
                             }
```

## 4. 주요 데이터 파일

### Artifacts 디렉토리 (ML 결과물)
| 파일명 | 형식 | 용도 | 크기 |
|--------|------|------|------|
| `feature_importance_rf.csv` | CSV | 특성 중요도 점수 | ~1KB |
| `confusion_matrix_rf.csv` | CSV | 혼동 행렬 (2x2) | <1KB |
| `classification_report_rf.json` | JSON | 분류 메트릭 (정밀도, 재현율, F1) | ~2KB |
| `metrics_summary_randomforest.json` | JSON | 모델 성능 요약 | ~1KB |
| `safe_region_result.json` | JSON | 공정 안전 구간 분석 | ~5KB |
| `combined_data.csv` | CSV | 통합 데이터셋 | 425KB |

### Data 디렉토리 (업로드 파일)
- 패턴: `uploaded_YYYYMMDD_HHMMSS_원본파일명.csv`
- 센서 파일: 파일명에 'error' 미포함
- 에러 파일: 파일명에 'error' 포함

## 5. 데이터 처리 특징

### 서비스 레이어 패턴
모든 아티팩트 로딩 함수는 일관된 패턴:
1. `artifacts/` 디렉토리에서 파일 읽기
2. 파일 없을 시 적절한 에러 처리 (`FileNotFoundError` 또는 에러 딕셔너리)
3. JSON 직렬화 가능한 dict 형태로 반환

### 데이터 정규화
- CSV 파일의 인덱스 컬럼 처리
- 의미론적 키 이름 사용 (true_0 → normal_to_normal)
- 프론트엔드 친화적 형식 변환

### 파일 명명 규칙
- 타임스탬프 기반 고유 파일명
- 원본 파일명 보존
- 파일 타입별 분류 (센서/에러)

## 6. 확장 가능성

현재 시스템은 다음과 같은 확장이 가능합니다:

```mermaid
graph TB
    subgraph "현재 시스템"
        CURRENT[정적 아티팩트<br/>파일 기반]
    end

    subgraph "확장 가능한 기능"
        DB[(데이터베이스<br/>PostgreSQL/MongoDB)]
        CACHE[(캐시<br/>Redis)]
        QUEUE[작업 큐<br/>Celery/RQ]
        RT[실시간 분석<br/>WebSocket]
    end

    CURRENT -.-> DB
    CURRENT -.-> CACHE
    CURRENT -.-> QUEUE
    CURRENT -.-> RT

    style CURRENT fill:#c8e6c9
    style DB fill:#fff9c4
    style CACHE fill:#fff9c4
    style QUEUE fill:#fff9c4
    style RT fill:#fff9c4
```
