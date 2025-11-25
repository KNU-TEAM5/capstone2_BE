# app/analysis/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os


def print_data_summary(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Any]:
    """
    최종 정제 데이터의 기본 정보 요약

    Args:
        df: 요약할 DataFrame
        num_cols: 수치 컬럼 리스트

    Returns:
        요약 정보 딕셔너리
    """
    print("✅ [최종 정제 데이터 기본 정보]")
    print("shape:", df.shape)
    print("\ncolumns:", list(df.columns))
    print("\ninfo:")
    print(df.info())
    print("\nNA count:")
    print(df.isna().sum())

    # 수치 컬럼만 따로
    df_num = df[num_cols]

    print("\n📈 [수치 컬럼 기술 통계량]")
    stats = df_num.describe().round(3)
    print(stats)

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "na_count": df.isna().sum().to_dict(),
        "statistics": stats.to_dict()
    }


def plot_histograms(df: pd.DataFrame, num_cols: List[str], save_path: str = None) -> str:
    """
    수치 컬럼의 히스토그램 생성 (분포 확인)

    Args:
        df: 데이터프레임
        num_cols: 수치 컬럼 리스트
        save_path: 저장 경로 (선택)

    Returns:
        저장된 파일 경로
    """
    df_num = df[num_cols]

    df_num.hist(bins=30, figsize=(10, 6))
    plt.suptitle("Distribution of temp / humid / press (after cleaning)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[저장 완료] {save_path}")

    plt.show()
    return save_path


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: List[str], save_path: str = None) -> Dict[str, Any]:
    """
    상관관계 히트맵 생성

    Args:
        df: 데이터프레임
        num_cols: 수치 컬럼 리스트
        save_path: 저장 경로 (선택)

    Returns:
        상관관계 행렬과 저장 경로
    """
    df_num = df[num_cols]
    corr = df_num.corr()

    plt.figure(figsize=(4, 3))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", square=True)
    plt.title("Correlation (temp - humid - press)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[저장 완료] {save_path}")

    plt.show()

    return {
        "correlation_matrix": corr.to_dict(),
        "save_path": save_path
    }


def plot_boxplot(df: pd.DataFrame, num_cols: List[str], save_path: str = None) -> str:
    """
    박스플롯 생성 (이상치 제거 후 분포 확인)

    Args:
        df: 데이터프레임
        num_cols: 수치 컬럼 리스트
        save_path: 저장 경로 (선택)

    Returns:
        저장된 파일 경로
    """
    df_num = df[num_cols]

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_num, orient="h")
    plt.title("Boxplot of temp / humid / press (after cleaning)")
    plt.xlabel("value")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[저장 완료] {save_path}")

    plt.show()
    return save_path


def plot_label_distribution(df: pd.DataFrame, label_col: str = "error", save_path: str = None) -> Dict[str, Any]:
    """
    불량 여부 라벨 분포 시각화

    Args:
        df: 데이터프레임
        label_col: 라벨 컬럼명
        save_path: 저장 경로 (선택)

    Returns:
        라벨 분포 정보 및 저장 경로
    """
    label_counts = df[label_col].value_counts().sort_index()

    plt.figure(figsize=(4, 3))
    label_counts.plot(kind="bar")
    plt.xticks([0, 1], ["normal(0)", "error(1)"], rotation=0)
    plt.title("Label Distribution (error)")
    plt.ylabel("count")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[저장 완료] {save_path}")

    plt.show()

    return {
        "label_counts": label_counts.to_dict(),
        "total": len(df),
        "normal_count": int(label_counts.get(0, 0)),
        "error_count": int(label_counts.get(1, 0)),
        "error_rate": round((label_counts.get(1, 0) / len(df)) * 100, 2),
        "save_path": save_path
    }


def generate_eda_report(
    df: pd.DataFrame,
    num_cols: List[str],
    label_col: str = "error",
    output_dir: str = "artifacts/eda"
) -> Dict[str, Any]:
    """
    탐색적 데이터 분석(EDA) 전체 리포트 생성

    Args:
        df: 분석할 데이터프레임
        num_cols: 수치 컬럼 리스트
        label_col: 라벨 컬럼명
        output_dir: 시각화 저장 디렉토리

    Returns:
        EDA 리포트
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("📊 탐색적 데이터 분석(EDA) 리포트 생성")
    print("="*60 + "\n")

    # 1. 기본 정보 요약
    summary = print_data_summary(df, num_cols)

    # 2. 히스토그램
    print("\n[1/4] 히스토그램 생성 중...")
    hist_path = os.path.join(output_dir, "histogram.png")
    plot_histograms(df, num_cols, hist_path)

    # 3. 상관관계 히트맵
    print("\n[2/4] 상관관계 히트맵 생성 중...")
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    corr_result = plot_correlation_heatmap(df, num_cols, corr_path)

    # 4. 박스플롯
    print("\n[3/4] 박스플롯 생성 중...")
    box_path = os.path.join(output_dir, "boxplot.png")
    plot_boxplot(df, num_cols, box_path)

    # 5. 라벨 분포
    print("\n[4/4] 라벨 분포 시각화 중...")
    label_path = os.path.join(output_dir, "label_distribution.png")
    label_result = plot_label_distribution(df, label_col, label_path)

    print("\n" + "="*60)
    print("✅ EDA 리포트 생성 완료")
    print(f"저장 위치: {output_dir}")
    print("="*60 + "\n")

    return {
        "summary": summary,
        "correlation": corr_result,
        "label_distribution": label_result,
        "visualizations": {
            "histogram": hist_path,
            "correlation_heatmap": corr_path,
            "boxplot": box_path,
            "label_distribution": label_path
        },
        "output_dir": output_dir
    }


# =========================================
#  모델 관련 시각화
# =========================================

def plot_feature_importance(
    model,
    feature_names: List[str],
    save_path: str = None,
    top_n: int = None
) -> Dict[str, Any]:
    """
    RandomForest 변수 중요도 시각화

    Args:
        model: 학습된 RandomForest 모델
        feature_names: 특성 이름 리스트
        save_path: 저장 경로 (선택)
        top_n: 상위 N개 특성만 표시 (선택)

    Returns:
        변수 중요도 정보
    """
    # 변수 중요도 DataFrame 생성
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    # top_n 설정이 있으면 상위 N개만 선택
    if top_n:
        importance_df = importance_df.head(top_n)

    # 시각화
    plt.figure(figsize=(5, 3))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title("Feature Importance - RandomForest")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[저장 완료] {save_path}")

    plt.show()

    print("\n📊 [변수 중요도]")
    print(importance_df.to_string(index=False))

    return {
        "feature_importance": importance_df.to_dict(orient="records"),
        "save_path": save_path
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: List[str] = None,
    save_path: str = None
) -> Dict[str, Any]:
    """
    혼동 행렬(Confusion Matrix) 시각화

    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        labels: 라벨 이름 리스트 (기본값: ["Normal", "Error"])
        save_path: 저장 경로 (선택)

    Returns:
        혼동 행렬 정보
    """
    from sklearn.metrics import confusion_matrix

    if labels is None:
        labels = ["Normal", "Error"]

    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)

    # 시각화
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[저장 완료] {save_path}")

    plt.show()

    print("\n📊 [혼동 행렬]")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

    return {
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "save_path": save_path
    }


def plot_roc_curve(
    y_true,
    y_pred_proba,
    save_path: str = None
) -> Dict[str, Any]:
    """
    ROC 곡선 시각화

    Args:
        y_true: 실제 라벨
        y_pred_proba: 예측 확률 (positive class)
        save_path: 저장 경로 (선택)

    Returns:
        ROC 정보
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    # ROC 곡선 계산
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    # 시각화
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[저장 완료] {save_path}")

    plt.show()

    print(f"\n📊 [ROC AUC Score]: {auc_score:.4f}")

    return {
        "auc_score": round(auc_score, 4),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "save_path": save_path
    }


def plot_performance_curves(
    y_true,
    y_pred_proba,
    model_name: str = "RandomForest",
    save_path: str = None
) -> Dict[str, Any]:
    """
    ROC Curve와 Precision-Recall Curve를 함께 시각화

    Args:
        y_true: 실제 라벨
        y_pred_proba: 예측 확률 (positive class)
        model_name: 모델 이름 (기본값: "RandomForest")
        save_path: 저장 경로 (선택)

    Returns:
        성능 곡선 정보
    """
    from sklearn.metrics import (
        roc_curve, auc,
        precision_recall_curve, average_precision_score
    )

    # ROC Curve 계산
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve 계산
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)

    # 그래프 시각화
    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="blue", lw=2,
             label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.title("ROC Curve", fontsize=13)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="blue", lw=2,
             label=f"{model_name} (AP = {ap_score:.4f})")
    plt.title("Precision-Recall Curve", fontsize=13)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[저장 완료] {save_path}")

    plt.show()

    print(f"\n📊 [성능 곡선]")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {ap_score:.4f}")

    return {
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(ap_score, 4),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "save_path": save_path
    }


def generate_model_report(
    model,
    feature_names: List[str],
    y_true,
    y_pred,
    y_pred_proba=None,
    output_dir: str = "artifacts/model_report"
) -> Dict[str, Any]:
    """
    모델 평가 시각화 리포트 생성

    Args:
        model: 학습된 모델
        feature_names: 특성 이름 리스트
        y_true: 실제 라벨
        y_pred: 예측 라벨
        y_pred_proba: 예측 확률 (선택, ROC 곡선용)
        output_dir: 저장 디렉토리

    Returns:
        모델 리포트
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("📊 모델 평가 시각화 리포트 생성")
    print("="*60 + "\n")

    # 1. Feature Importance
    print("[1/3] 변수 중요도 시각화 중...")
    importance_path = os.path.join(output_dir, "feature_importance.png")
    importance_result = plot_feature_importance(model, feature_names, importance_path)

    # 2. Confusion Matrix
    print("\n[2/3] 혼동 행렬 시각화 중...")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    cm_result = plot_confusion_matrix(y_true, y_pred, save_path=cm_path)

    # 3. ROC Curve (확률이 제공된 경우)
    roc_result = None
    if y_pred_proba is not None:
        print("\n[3/3] ROC 곡선 시각화 중...")
        roc_path = os.path.join(output_dir, "roc_curve.png")
        roc_result = plot_roc_curve(y_true, y_pred_proba, roc_path)

    print("\n" + "="*60)
    print("✅ 모델 평가 리포트 생성 완료")
    print(f"저장 위치: {output_dir}")
    print("="*60 + "\n")

    return {
        "feature_importance": importance_result,
        "confusion_matrix": cm_result,
        "roc_curve": roc_result,
        "output_dir": output_dir
    }
