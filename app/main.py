# app/main.py
from fastapi import FastAPI

app = FastAPI(
    title="Capstone QC API",
    version="0.1.0",
    description="전해탈지 공정 품질 예측 / 데이터 분석용 FastAPI 백엔드",
)

@app.get("/")
def read_root():
    return {"message": "Capstone QC API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}