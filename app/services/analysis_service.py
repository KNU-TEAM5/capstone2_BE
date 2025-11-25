# app/services/analysis_service.py

import os
import pandas as pd
from typing import Dict, Any

from app.analysis.preprocessing import (
    preprocess_and_merge_sensors,
    clean_data
)
from app.analysis.evaluation import (
    evaluate_data_quality,
    estimate_safe_region,
    save_safe_region_json
)
from app.analysis.model_training import (
    train_and_evaluate_rf,
    save_model_artifacts
)
from app.analysis.visualization import generate_eda_report


class AnalysisService:
    """데이터 분석 전체 파이프라인을 관리하는 서비스"""

    def __init__(self, data_dir: str = "data", output_dir: str = "artifacts"):
        self.data_dir = data_dir
        self.output_dir = output_dir

    def run_full_pipeline(
        self,
        feature_cols: list = ["temp", "humid", "press"],
        target_col: str = "error",
        num_files: int = 2,
        prob_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        전체 분석 파이프라인 실행

        Returns:
            분석 결과 요약
        """
        try:
            # 1. 데이터 전처리 및 병합
            merge_result = preprocess_and_merge_sensors(
                data_dir=self.data_dir,
                num_files=num_files,
                output_dir=self.output_dir
            )

            # 병합된 데이터 로드
            combined_df = pd.read_csv(merge_result["output_path"])

            # 2. 데이터 정제
            clean_result = clean_data(
                df=combined_df,
                num_cols=feature_cols,
                missing_method="drop",
                outlier_threshold=3.0,
                output_path=os.path.join(self.output_dir, "cleaned_data.csv")
            )

            df_clean = clean_result["cleaned_dataframe"]

            # 3. 데이터 품질 평가
            quality_result = evaluate_data_quality(
                df=df_clean,
                num_cols=feature_cols
            )

            # 4. EDA 리포트 생성
            eda_result = generate_eda_report(
                df=df_clean,
                num_cols=feature_cols,
                label_col=target_col,
                output_dir=os.path.join(self.output_dir, "eda")
            )

            # 5. 모델 학습 및 평가
            model_result = train_and_evaluate_rf(
                df=df_clean,
                feature_cols=feature_cols,
                target_col=target_col,
                test_size=0.2,
                n_estimators=200,
                random_state=42
            )

            # 6. 모델 결과물 저장
            # X_test를 다시 생성해야 함
            from sklearn.model_selection import train_test_split
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_test_scaled = model_result["scaler"].transform(X_test)

            save_result = save_model_artifacts(
                model=model_result["model"],
                model_name="RandomForestClassifier",
                X_test=X_test_scaled,
                y_test=y_test,
                feature_names=feature_cols,
                output_dir=self.output_dir
            )

            # 7. 안전 영역 분석
            temp_range = (df_clean["temp"].quantile(0.05), df_clean["temp"].quantile(0.95))
            humid_range = (df_clean["humid"].quantile(0.05), df_clean["humid"].quantile(0.95))
            press_range = (df_clean["press"].quantile(0.05), df_clean["press"].quantile(0.95))

            safe_summary, _ = estimate_safe_region(
                model=model_result["model"],
                scaler=model_result["scaler"],
                temp_range=temp_range,
                humid_range=humid_range,
                press_range=press_range,
                prob_threshold=prob_threshold
            )

            if safe_summary:
                save_safe_region_json(
                    safe_summary=safe_summary,
                    prob_threshold=prob_threshold,
                    output_dir=self.output_dir,
                    filename="safe_region_result.json"
                )

            # 8. 결과 요약
            return {
                "status": "success",
                "data_summary": {
                    "original_rows": merge_result["combined_rows"],
                    "cleaned_rows": clean_result["final_rows"],
                    "removed_rows": clean_result["total_removed_rows"]
                },
                "quality": {
                    "overall_dqi": quality_result["overall_dqi"]
                },
                "model": {
                    "accuracy": model_result["evaluation"]["accuracy"],
                    "precision": model_result["evaluation"]["precision"],
                    "recall": model_result["evaluation"]["recall"],
                    "f1_score": model_result["evaluation"]["f1_score"],
                    "auc": model_result["evaluation"]["auc"]
                },
                "safe_region": safe_summary,
                "artifacts": {
                    "cleaned_data": os.path.join(self.output_dir, "cleaned_data.csv"),
                    "model_file": save_result.get("model_file"),
                    "eda_dir": eda_result["output_dir"]
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


def run_analysis_pipeline(
    data_dir: str = "data",
    output_dir: str = "artifacts",
    **kwargs
) -> Dict[str, Any]:
    """
    분석 파이프라인 실행 (함수형 인터페이스)

    Returns:
        분석 결과
    """
    service = AnalysisService(data_dir=data_dir, output_dir=output_dir)
    return service.run_full_pipeline(**kwargs)
