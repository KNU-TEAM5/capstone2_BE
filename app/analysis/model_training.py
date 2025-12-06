# app/analysis/model_training.py
# ml 모델 학습 및 저장 - RandomForest 학습, 모델 평가 및 artifacts 생성

import os
import datetime
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 머신러닝 관련
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score, classification_report,
    average_precision_score, balanced_accuracy_score, ConfusionMatrixDisplay,
    precision_recall_curve, auc
)
import joblib
import json

# 이곳에 모델 학습 및 분석 함수들을 작성하세요
from typing import Dict, Any, Tuple


def prepare_data(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    데이터 준비: 입력(X), 타깃(y) 분리 및 학습/테스트 분할

    Args:
        df: 데이터프레임
        feature_cols: 특성 컬럼 리스트
        target_col: 타깃 컬럼명
        test_size: 테스트 데이터 비율 (기본값: 0.2)
        random_state: 랜덤 시드

    Returns:
        X_train, X_test, y_train, y_test
    """
    # 입력(X), 타깃(y) 분리
    X = df[feature_cols]
    y = df[target_col]

    # 학습용/테스트용 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"✅ 데이터 분할 완료")
    print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    print(f"훈련 라벨 분포:\n{y_train.value_counts()}")
    print(f"테스트 라벨 분포:\n{y_test.value_counts()}")

    return X_train, X_test, y_train, y_test


def scale_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    데이터 스케일링 (표준화)

    Args:
        X_train: 훈련 데이터
        X_test: 테스트 데이터

    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n✅ 데이터 스케일링 완료")
    print(f"훈련 데이터 shape: {X_train_scaled.shape}")
    print(f"테스트 데이터 shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, scaler


def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    RandomForest 모델 학습

    Args:
        X_train: 훈련 데이터
        y_train: 훈련 라벨
        n_estimators: 트리 개수
        max_depth: 최대 깊이
        min_samples_split: 분할을 위한 최소 샘플 수
        min_samples_leaf: 리프 노드의 최소 샘플 수
        random_state: 랜덤 시드

    Returns:
        학습된 RandomForest 모델
    """
    print(f"\n🌲 RandomForest 모델 학습 중...")

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    print(f"✅ 모델 학습 완료")
    print(f"트리 개수: {n_estimators}")
    print(f"최대 깊이: {max_depth}")

    return rf_model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    모델 평가 및 성능 지표 계산

    Args:
        model: 학습된 모델
        X_test: 테스트 데이터
        y_test: 테스트 라벨

    Returns:
        성능 평가 결과
    """
    # 예측
    y_pred = model.predict(X_test)

    # 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f"\n📊 [RandomForest 성능 요약]")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"\nDetailed report:\n{classification_report(y_test, y_pred, digits=4)}")

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auc": round(auc, 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "predictions": y_pred.tolist()
    }


def train_and_evaluate_rf(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    RandomForest 모델 학습 및 평가 전체 파이프라인

    Args:
        df: 데이터프레임
        feature_cols: 특성 컬럼 리스트
        target_col: 타깃 컬럼명
        test_size: 테스트 데이터 비율
        n_estimators: 트리 개수
        random_state: 랜덤 시드

    Returns:
        학습 및 평가 결과
    """
    print("\n" + "="*60)
    print("🚀 RandomForest 모델 학습 및 평가 시작")
    print("="*60)

    # 1. 데이터 준비
    X_train, X_test, y_train, y_test = prepare_data(
        df, feature_cols, target_col, test_size, random_state
    )

    # 2. 데이터 스케일링
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # 3. 모델 학습
    model = train_random_forest(
        X_train_scaled, y_train, n_estimators=n_estimators, random_state=random_state
    )

    # 4. 모델 평가
    evaluation = evaluate_model(model, X_test_scaled, y_test)

    print("\n" + "="*60)
    print("✅ RandomForest 모델 학습 및 평가 완료")
    print("="*60)

    return {
        "model": model,
        "scaler": scaler,
        "evaluation": evaluation,
        "data_split": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_cols": feature_cols,
            "target_col": target_col
        }
    }


# =========================================
#  모델 결과물 저장
# =========================================

def save_model_artifacts(
    model,
    model_name: str,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: list,
    output_dir: str = "artifacts"
) -> Dict[str, str]:
    """
    모델의 모든 결과물(지표, 이미지, 모델 파일)을 저장

    Args:
        model: 학습된 모델
        model_name: 모델 이름
        X_test: 테스트 데이터
        y_test: 테스트 라벨
        feature_names: 특성 이름 리스트
        output_dir: 출력 디렉토리

    Returns:
        저장된 파일 경로들
    """
    print("\n" + "="*60)
    print(f"💾 {model_name} 결과물 저장 시작")
    print("="*60)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    model_tag = model_name.lower().replace("classifier", "").replace(" ", "")

    # ===== 1) 예측 및 점수 계산 =====
    y_true = y_test.values if isinstance(y_test, pd.Series) else np.asarray(y_test)
    y_pred = model.predict(X_test)

    # 양성 클래스(1)에 대한 확률 또는 점수
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    # 혼동행렬
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 주요 지표 계산
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "avg_precision_AP": average_precision_score(y_true, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "specificity": tn / (tn + fp + 1e-12),
        "n_test_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
    }

    saved_files = {}

    # ===== 2) metrics 저장 =====
    metrics_path = os.path.join(output_dir, f"metrics_{model_tag}.csv")
    pd.DataFrame(list(metrics.items()), columns=["metric", "value"]).to_csv(
        metrics_path, index=False
    )
    saved_files["metrics_csv"] = metrics_path
    print(f"[저장 완료] 평가 지표 → {metrics_path}")

    # ===== 2-1) classification_report JSON 저장 =====
    report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_json_path = os.path.join(output_dir, f"classification_report_{model_tag}.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    saved_files["classification_report_json"] = report_json_path
    print(f"[저장 완료] 분류 리포트(JSON) → {report_json_path}")

    # ===== 3) 테스트 예측 결과 저장 =====
    pred_df = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "y_score": y_score}
    )
    pred_path = os.path.join(output_dir, f"test_predictions_{model_tag}.csv")
    pred_df.to_csv(pred_path, index=False)
    saved_files["predictions"] = pred_path
    print(f"[저장 완료] 예측 결과 → {pred_path}")

    # ===== 4) 혼동행렬 이미지 & CSV =====
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f"Confusion Matrix ({model_name})")
    cm_img_path = os.path.join(output_dir, f"confusion_matrix_{model_tag}.png")
    plt.savefig(cm_img_path, bbox_inches="tight")
    plt.close()
    saved_files["confusion_matrix_img"] = cm_img_path
    print(f"[저장 완료] 혼동행렬 이미지 → {cm_img_path}")

    cm_df = pd.DataFrame(
        cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
    )
    cm_csv_path = os.path.join(output_dir, f"confusion_matrix_{model_tag}.csv")
    cm_df.to_csv(cm_csv_path)
    saved_files["confusion_matrix_csv"] = cm_csv_path
    print(f"[저장 완료] 혼동행렬 값 → {cm_csv_path}")

    # ===== 5) ROC Curve 이미지 =====
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_val:.3f}", color="darkorange")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_name})")
    plt.legend()
    plt.grid(True)
    roc_path = os.path.join(output_dir, f"roc_curve_{model_tag}.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    saved_files["roc_curve"] = roc_path
    print(f"[저장 완료] ROC Curve 이미지 → {roc_path}")

    # ===== 6) Precision–Recall Curve 이미지 =====
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, color="blue", label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({model_name})")
    plt.legend()
    plt.grid(True)
    pr_path = os.path.join(output_dir, f"pr_curve_{model_tag}.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    saved_files["pr_curve"] = pr_path
    print(f"[저장 완료] PR Curve 이미지 → {pr_path}")

    # ===== 7) Feature Importance (트리 계열만) =====
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(
            ascending=False
        )
        fi_path = os.path.join(output_dir, f"feature_importance_{model_tag}.csv")
        fi.to_csv(fi_path, header=["importance"])
        saved_files["feature_importance"] = fi_path
        print(f"[저장 완료] Feature Importance → {fi_path}")

    # ===== 8) 모델 파일(joblib) =====
    model_path = os.path.join(output_dir, f"model_{model_tag}.joblib")
    joblib.dump(model, model_path)
    saved_files["model_file"] = model_path
    print(f"[저장 완료] 모델 파일 → {model_path}")

    print("="*60)
    print(f"✅ {model_name} 결과물 저장 완료")
    print(f"저장 위치: {os.path.abspath(output_dir)}")
    print("="*60 + "\n")

    return saved_files
