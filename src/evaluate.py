import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    f1_score,
)


def _default_metrics_path():
    return Path(__file__).parent / "results" / "metrics_summary.json"


def _to_python_type(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _to_python_type(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_python_type(item) for item in value]
    return value


def save_metrics(task_name, metrics, save_path=None):
    if save_path is None:
        save_path = _default_metrics_path()
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        with open(save_path, "r", encoding="utf-8") as file:
            try:
                all_metrics = json.load(file)
            except json.JSONDecodeError:
                all_metrics = {}
    else:
        all_metrics = {}

    all_metrics[task_name] = _to_python_type(metrics)

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=4, ensure_ascii=False)


def evaluate_classification(y_true, y_pred, save_path=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print("\n===== Classification Evaluation =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report_text)

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": report_dict,
    }
    save_metrics("classification", metrics, save_path)
    return metrics


def evaluate_regression(y_true, y_pred, save_path=None):
    """
    Đánh giá bài toán hồi quy.

    Chỉ file này dùng sklearn.metrics theo đúng yêu cầu đề bài.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n===== Regression Evaluation =====")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2-Score: {r2:.4f}")
    print(f"Độ khớp xấp xỉ: {r2 * 100:.2f}%")

    metrics = {
        "mse": mse,
        "mae": mae,
        "r2_score": r2,
        "fit_percent": r2 * 100,
    }
    save_metrics("regression", metrics, save_path)
    return metrics
