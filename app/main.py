# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import results, files, analysis

app = FastAPI(
    title="Capstone QC API",
    version="0.1.0",
    description="전해탈지 공정 품질 예측 / 데이터 분석용 FastAPI 백엔드",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://samsung-lac.vercel.app",
        "http://localhost:3000",  # 로컬 개발용
        "http://localhost:5173",  # Vite 개발 서버용
        "http://127.0.0.1:5500",  # LiveServer
    ],
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

@app.get("/")
def read_root():
    return {"message": "Capstone QC API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# API 라우터 등록
app.include_router(results.router, prefix="/api/v1", tags=["Results"])
app.include_router(files.router, prefix="/api/v1", tags=["Files"])
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])