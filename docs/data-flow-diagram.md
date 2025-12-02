# 데이터 플로우 다이어그램

## 전체 데이터 흐름 개요

```mermaid
flowchart TB
    subgraph Input["📥 입력 단계"]
        A[사용자 CSV 파일]
    end

    subgraph Upload["1️⃣ 업로드 & 저장"]
        B[POST /upload-csv]
        C[data_service.process_uploaded_csv]
        D[data/ 폴더에 저장<br/>uploaded_YYYYMMDD_HHMMSS_*.csv]
    end

    subgraph Analysis["2️⃣ 분석 파이프라인 실행"]
        E[POST /start-analysis]
        F[BackgroundTasks 등록]
        G[analysis_service.run_analysis_pipeline]

        subgraph Pipeline["7단계 순차 실행"]
            G1[1. 데이터 전처리 & 병합<br/>→ combined_data.csv]
            G2[2. 데이터 정제<br/>→ cleaned_data.csv]
            G3[3. 품질 평가<br/>→ DQI 계산]
            G4[4. EDA 리포트<br/>→ eda/*.png]
            G5[5. 모델 학습<br/>→ model_rf.joblib]
            G6[6. 결과 저장<br/>→ 4개 파일]
            G7[7. 안전 영역 분석<br/>→ safe_region_result.json]
        end
    end

    subgraph Artifacts["📊 분석 결과 (artifacts/)"]
        H1[feature_importance_rf.csv]
        H2[confusion_matrix_rf.csv]
        H3[classification_report_rf.json]
        H4[safe_region_result.json]
        H5[combined_data.csv]
        H6[cleaned_data.csv]
        H7[model_rf.joblib]
    end

    subgraph Query["3️⃣ 결과 조회"]
        I[GET /feature-importance<br/>GET /confusion-matrix<br/>GET /classification-report-rf<br/>GET /safe-region]
        J[artifact_service]
        K[JSON 응답]
    end

    subgraph Frontend["💻 프론트엔드"]
        L[결과 시각화]
    end

    A --> B
    B --> C
    C --> D

    E --> F
    F --> G
    G --> Pipeline

    G1 --> G2
    G2 --> G3
    G3 --> G4
    G4 --> G5
    G5 --> G6
    G6 --> G7

    Pipeline --> Artifacts

    Artifacts --> J
    I --> J
    J --> K
    K --> L

    style Input fill:#e1f5ff
    style Upload fill:#fff4e1
    style Analysis fill:#ffe1f5
    style Artifacts fill:#e1ffe1
    style Query fill:#f5e1ff
    style Frontend fill:#ffe1e1
```

---

## 상세 데이터 플로우

### Flow 1: CSV 업로드 → data/ 저장

```mermaid
flowchart LR
    A[사용자 CSV 파일] -->|multipart/form-data| B[POST /api/v1/upload-csv]
    B --> C{파일 확장자 검증}
    C -->|.csv 아님| D[400 Error]
    C -->|.csv 맞음| E[data_service.process_uploaded_csv]

    E --> F[여러 인코딩 시도<br/>utf-8, cp949, euc-kr]
    F --> G[pandas DataFrame 생성]
    G --> H[타임스탬프 생성<br/>YYYYMMDD_HHMMSS]
    H --> I[파일명 생성<br/>uploaded_TIMESTAMP_원본명.csv]
    I --> J[data/ 폴더에 저장]

    J --> K[기본 분석 수행<br/>행/열 수, 결측치, 통계]
    K --> L[200 OK<br/>분석 정보 반환]

    style D fill:#ffcccc
    style L fill:#ccffcc
```

**데이터 변환:**
- 입력: Binary file (CSV)
- 중간: pandas DataFrame (메모리)
- 출력: CSV 파일 (data/uploaded_YYYYMMDD_HHMMSS_원본명.csv)

---

### Flow 2: 분석 실행 → artifacts/ 생성

```mermaid
flowchart TB
    Start[POST /api/v1/start-analysis] --> Check{상태 확인}
    Check -->|이미 실행 중| Error[400 Error]
    Check -->|idle| BG[BackgroundTasks 등록]

    BG --> Status1[status = 'running']
    Status1 --> Response[200 OK 즉시 반환]

    BG --> Pipeline[run_analysis_pipeline 백그라운드 실행]

    Pipeline --> Step1[1️⃣ 전처리 & 병합]
    Step1 --> File1[combined_data.csv]

    File1 --> Step2[2️⃣ 데이터 정제]
    Step2 --> File2[cleaned_data.csv]

    File2 --> Step3[3️⃣ 품질 평가]
    Step3 --> Calc1[DQI 계산<br/>메모리만]

    Calc1 --> Step4[4️⃣ EDA 리포트]
    Step4 --> Files4[eda/*.png<br/>시각화 파일들]

    Files4 --> Step5[5️⃣ 모델 학습]
    Step5 --> File5[model_rf.joblib]

    File5 --> Step6[6️⃣ 결과 저장]
    Step6 --> Files6A[feature_importance_rf.csv]
    Step6 --> Files6B[confusion_matrix_rf.csv]
    Step6 --> Files6C[classification_report_rf.json]
    Step6 --> Files6D[metrics_summary_randomforest.json]

    Files6D --> Step7[7️⃣ 안전 영역 분석]
    Step7 --> File7[safe_region_result.json]

    File7 --> Status2[status = 'completed']
    Status2 --> Result[result = summary]

    style Error fill:#ffcccc
    style Response fill:#ccffcc
    style Status2 fill:#ccffcc
```

**데이터 변환 상세:**

| 단계 | 입력 | 처리 | 출력 |
|-----|------|------|------|
| 1. 전처리 | data/*.csv 여러 파일 | 컬럼 정규화, 병합 | combined_data.csv |
| 2. 정제 | combined_data.csv | 결측치 제거, Z-score 이상치 제거 | cleaned_data.csv |
| 3. 품질 평가 | cleaned_data.csv | DQI 계산 (0~1 점수) | 메모리 (파일 저장 안함) |
| 4. EDA | cleaned_data.csv | matplotlib 시각화 | eda/*.png (5~10개 그래프) |
| 5. 모델 학습 | cleaned_data.csv | RandomForest 학습 | model_rf.joblib, scaler.joblib |
| 6. 평가 결과 | 학습 완료 모델 | 예측 & 평가 메트릭 | 4개 CSV/JSON 파일 |
| 7. 안전 영역 | 학습 완료 모델 | 격자점 예측 & 영역 추정 | safe_region_result.json |

---

### Flow 3: 결과 조회 → JSON 응답

```mermaid
flowchart LR
    A[GET /api/v1/feature-importance] --> B[artifact_service.load_feature_importance]
    B --> C[artifacts/feature_importance_rf.csv 읽기]
    C --> D{파일 존재?}
    D -->|없음| E[error: file not found 반환]
    D -->|있음| F[pandas로 CSV 읽기]
    F --> G[인덱스 → feature 컬럼 변환]
    G --> H[.to_dict orient=records]
    H --> I[JSON 응답<br/>feature, importance 배열]

    style E fill:#ffcccc
    style I fill:#ccffcc
```

```mermaid
flowchart LR
    A[GET /api/v1/confusion-matrix] --> B[artifact_service.load_confusion_matrix]
    B --> C[artifacts/confusion_matrix_rf.csv 읽기]
    C --> D[pandas DataFrame]
    D --> E[true_0/pred_0 → normal_to_normal]
    E --> F[true_0/pred_1 → normal_to_defect]
    F --> G[true_1/pred_0 → defect_to_normal]
    G --> H[true_1/pred_1 → defect_to_defect]
    H --> I[JSON 응답<br/>의미론적 키]

    style I fill:#ccffcc
```

```mermaid
flowchart LR
    A[GET /api/v1/classification-report-rf] --> B[artifact_service.load_classification_report_rf]
    B --> C[artifacts/classification_report_rf.json]
    C --> D{파일 존재?}
    D -->|없음| E[404 FileNotFoundError]
    D -->|있음| F[json.load]
    F --> G[JSON 응답<br/>precision, recall, f1-score]

    style E fill:#ffcccc
    style G fill:#ccffcc
```

```mermaid
flowchart LR
    A[GET /api/v1/safe-region] --> B[artifact_service.load_safe_region_result]
    B --> C[artifacts/safe_region_result.json]
    C --> D{파일 존재?}
    D -->|없음| E[404 FileNotFoundError]
    D -->|있음| F[json.load]
    F --> G[JSON 응답<br/>안전 범위 정보]

    style E fill:#ffcccc
    style G fill:#ccffcc
```

**데이터 변환:**
- CSV 파일 → pandas DataFrame → Python dict → JSON
- JSON 파일 → Python dict → JSON (그대로 전달)

---

### Flow 4: 상태 조회 (폴링)

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant API as Analysis API
    participant Mem as 메모리<br/>(analysis_status)

    Note over FE: 분석 시작 후 매 3초마다 폴링

    loop 주기적 상태 확인
        FE->>+API: GET /api/v1/analysis-status
        API->>Mem: analysis_status 딕셔너리 읽기
        Mem-->>API: {"status": "running", "result": null}
        API-->>-FE: 현재 상태 반환

        alt status == "completed"
            FE->>FE: 폴링 중지
            FE->>FE: 결과 화면으로 이동
        else status == "running"
            FE->>FE: 3초 대기
            Note over FE: 다음 폴링 준비
        else status == "error"
            FE->>FE: 에러 메시지 표시
            FE->>FE: 폴링 중지
        end
    end
```

**상태 데이터 구조:**
```json
{
  "status": "idle | running | completed | error",
  "result": null | {
    "status": "success",
    "data_summary": {...},
    "model": {...},
    "safe_region": {...}
  }
}
```

---

## 파일 생성 타임라인

분석 파이프라인 실행 시 파일들이 생성되는 순서와 예상 소요 시간:

```
t=0s     │ POST /start-analysis 호출
         │ └─ 200 OK 즉시 반환
         │
t=0~5s   │ 🔄 1단계: 전처리 & 병합
         │ └─ artifacts/combined_data.csv 생성
         │
t=5~10s  │ 🔄 2단계: 데이터 정제
         │ └─ artifacts/cleaned_data.csv 생성
         │
t=10~15s │ 🔄 3단계: 품질 평가
         │ └─ (파일 생성 없음, 메모리만)
         │
t=15~30s │ 🔄 4단계: EDA 리포트
         │ └─ artifacts/eda/*.png (여러 파일)
         │
t=30~50s │ 🔄 5단계: 모델 학습
         │ └─ artifacts/model_rf.joblib
         │ └─ artifacts/scaler.joblib
         │
t=50~55s │ 🔄 6단계: 평가 결과 저장
         │ ├─ artifacts/feature_importance_rf.csv
         │ ├─ artifacts/confusion_matrix_rf.csv
         │ ├─ artifacts/classification_report_rf.json
         │ └─ artifacts/metrics_summary_randomforest.json
         │
t=55~65s │ 🔄 7단계: 안전 영역 분석
         │ └─ artifacts/safe_region_result.json
         │
t=65s    │ ✅ 분석 완료
         │ └─ status = "completed"
```

**총 소요 시간**: 약 60~90초 (데이터 크기에 따라 변동)

---

## 데이터 생명주기 상태 다이어그램

```mermaid
stateDiagram-v2
    [*] --> 원본CSV: 사용자 업로드

    원본CSV --> 저장됨: data/ 저장

    저장됨 --> 병합중: 분석 시작
    병합중 --> combined_data: 1단계 완료

    combined_data --> 정제중: 2단계 시작
    정제중 --> cleaned_data: 2단계 완료

    cleaned_data --> 품질평가: 3단계 시작
    품질평가 --> DQI계산완료: 3단계 완료

    DQI계산완료 --> EDA생성중: 4단계 시작
    EDA생성중 --> EDA완료: PNG 파일들 생성

    EDA완료 --> 모델학습중: 5단계 시작
    모델학습중 --> 모델완료: joblib 저장

    모델완료 --> 평가중: 6단계 시작
    평가중 --> 평가완료: CSV/JSON 저장

    평가완료 --> 안전영역분석: 7단계 시작
    안전영역분석 --> 분석완료: JSON 저장

    분석완료 --> 조회가능: API 엔드포인트로 제공

    조회가능 --> [*]: 프론트엔드 시각화
```

---

## 저장소별 역할

| 저장소 | 경로 | 용도 | 생성 시점 | 소비 주체 |
|--------|------|------|----------|-----------|
| **원본 데이터** | `data/uploaded_*.csv` | 사용자 업로드 CSV | 업로드 API 호출 시 | analysis_service |
| **병합 데이터** | `artifacts/combined_data.csv` | 여러 센서 데이터 병합 | 분석 1단계 | 분석 2~7단계 |
| **정제 데이터** | `artifacts/cleaned_data.csv` | 결측치/이상치 제거 | 분석 2단계 | 분석 3~7단계 |
| **시각화** | `artifacts/eda/*.png` | 탐색적 데이터 분석 | 분석 4단계 | (프론트 직접 조회 가능) |
| **모델** | `artifacts/model_rf.joblib` | 학습된 RandomForest | 분석 5단계 | 분석 7단계 (안전 영역) |
| **평가 결과** | `artifacts/*_rf.{csv,json}` | 모델 성능 메트릭 | 분석 6단계 | artifact_service → API |
| **안전 영역** | `artifacts/safe_region_result.json` | 공정 안전 파라미터 | 분석 7단계 | artifact_service → API |
| **상태 정보** | 메모리 (analysis_status) | 분석 실행 상태 | 분석 시작/완료 | Analysis API |

---

## 데이터 변환 요약

### 인코딩 & 파싱
- **입력**: Binary CSV file
- **처리**:
  1. UTF-8 시도
  2. CP949 시도 (한글 Windows)
  3. EUC-KR 시도 (레거시 한글)
- **출력**: pandas DataFrame

### 정규화 & 병합
- **입력**: 여러 센서 CSV 파일들
- **처리**:
  1. 컬럼명 정규화 (소문자, 공백 제거)
  2. 타임스탬프 기준 병합
  3. 중복 제거
- **출력**: combined_data.csv (단일 DataFrame)

### 정제
- **입력**: combined_data.csv
- **처리**:
  1. 결측치 행 제거
  2. Z-score > 3 이상치 제거
  3. 인덱스 리셋
- **출력**: cleaned_data.csv

### ML 학습
- **입력**: cleaned_data.csv
- **처리**:
  1. train_test_split (80:20)
  2. StandardScaler 적용
  3. RandomForestClassifier 학습
- **출력**:
  - model_rf.joblib (모델)
  - scaler.joblib (스케일러)
  - 예측 결과 (메모리)

### 평가 메트릭
- **입력**: 예측 결과 (y_test vs y_pred)
- **처리**:
  1. confusion_matrix 계산
  2. classification_report 생성
  3. feature_importances_ 추출
- **출력**:
  - CSV (confusion matrix, feature importance)
  - JSON (classification report, metrics summary)

### 안전 영역 추정
- **입력**: 학습된 모델
- **처리**:
  1. 주요 특성 2개 선택
  2. 격자점 생성 (100x100)
  3. 각 격자점 예측
  4. "정상" 예측 영역 계산
- **출력**: safe_region_result.json
  - 안전 범위 (min/max)
  - 중심점
  - 영역 비율

### API 응답 변환
- **CSV → JSON**:
  - pandas → `.to_dict(orient="records")`
  - 의미론적 키로 매핑
- **JSON → JSON**:
  - 파일 그대로 전달
  - 추가 변환 없음

---

## 다이어그램 렌더링 방법

### 1. GitHub에서 보기
- 이 파일을 GitHub에 push하면 자동으로 렌더링됩니다

### 2. VS Code에서 보기
- Mermaid 플러그인 설치: `Markdown Preview Mermaid Support`
- 마크다운 미리보기 열기 (Cmd+Shift+V)

### 3. 온라인 에디터
- https://mermaid.live/ 에서 코드 붙여넣기
