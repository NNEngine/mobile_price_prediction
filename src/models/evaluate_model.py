import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import yaml
from sklearn.metrics import (
    # classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,

    # regression
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "params.yaml") as f:
    params = yaml.safe_load(f)

REPORTS_DIR = ROOT / params["reports"]["output_path"]
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_classification(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Computes all classification metrics defined in params.yaml.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.

    Returns:
        dict of metric names and their computed values.
    """
    metrics = {}
    requested = params["metrics"]["classification"]

    metric_map = {
        "accuracy":           lambda: accuracy_score(y_true, y_pred),
        "precision_weighted": lambda: precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    lambda: recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted":        lambda: f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro":    lambda: precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":       lambda: recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":           lambda: f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa":        lambda: cohen_kappa_score(y_true, y_pred),
        "matthews_corrcoef":  lambda: matthews_corrcoef(y_true, y_pred),
        "roc_auc_ovr":        lambda: roc_auc_score(y_true, y_pred, multi_class="ovr", average="weighted"),
    }

    for metric in requested:
        if metric in metric_map:
            try:
                metrics[metric] = round(float(metric_map[metric]()), 4)
            except Exception as e:
                logger.warning(f"Could not compute '{metric}': {e}")
                metrics[metric] = None

    # always include confusion matrix and classification report
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True)

    return metrics


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Computes all regression metrics defined in params.yaml.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.

    Returns:
        dict of metric names and their computed values.
    """
    metrics = {}
    requested = params["metrics"]["regression"]

    mse_val = mean_squared_error(y_true, y_pred)

    metric_map = {
        "r2":               lambda: r2_score(y_true, y_pred),
        "mae":              lambda: mean_absolute_error(y_true, y_pred),
        "mse":              lambda: mse_val,
        "rmse":             lambda: np.sqrt(mse_val),
        "mape":             lambda: mean_absolute_percentage_error(y_true, y_pred),
        "explained_variance": lambda: explained_variance_score(y_true, y_pred),
    }

    for metric in requested:
        if metric in metric_map:
            try:
                metrics[metric] = round(float(metric_map[metric]()), 4)
            except Exception as e:
                logger.warning(f"Could not compute '{metric}': {e}")
                metrics[metric] = None

    return metrics


def save_report(report: dict) -> None:
    """
    Saves the evaluation report to reports/ as a timestamped JSON file.

    Args:
        report: Dictionary containing all evaluation results.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = params["selected_model"]
    filename = REPORTS_DIR / f"{model_name}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Report saved to: {filename}")


def evaluate_model(comparison: pd.DataFrame) -> dict:
    """
    Evaluates the model based on selected_task and saves results to reports/.

    Args:
        comparison: DataFrame with columns ['actual', 'predicted']
                    as returned by predict_model().

    Returns:
        dict containing all computed metrics.

    Raises:
        ValueError: If selected_task is not set in params.yaml.
    """
    task = params["selected_task"]
    selected_model = params["selected_model"]

    if not task:
        raise ValueError("No task found. Run select_model.py first.")

    y_true = comparison["actual"]
    y_pred = comparison["predicted"]

    logger.info(f"Evaluating '{selected_model}' for task '{task}'")

    if task == "classification":
        metrics = evaluate_classification(y_true, y_pred)
    else:
        metrics = evaluate_regression(y_true, y_pred)

    report = {
        "model": selected_model,
        "task": task,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }

    save_report(report)
    logger.info(f"Evaluation complete: {metrics}")

    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=params["logging"]["format"]
    )
    logger.info("evaluate_model.py is not meant to be run directly.")
    logger.info("Import evaluate_model() into your pipeline and pass the comparison DataFrame.")
