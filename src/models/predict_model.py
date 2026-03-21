import logging
import joblib
import pandas as pd
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "params.yaml") as f:
    params = yaml.safe_load(f)

MODELS_DIR = ROOT / "models"


def load_model(model_name: str):
    """
    Loads a trained model from the models directory using joblib.

    Args:
        model_name: Name of the model to load e.g. "RandomForestClassifier"

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model found at: {model_path}")
    logger.info(f"Loading model from: {model_path}")
    return joblib.load(model_path)


def predict_model(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Loads the selected model and returns a comparison of
    actual vs predicted values.

    Args:
        X_test: Test features.
        y_test: Actual target values.

    Returns:
        pd.DataFrame with columns ['actual', 'predicted'].

    Raises:
        ValueError: If no model is selected in params.yaml.
        FileNotFoundError: If the saved model file does not exist.
    """
    selected_model = params["selected_model"]

    if not selected_model:
        raise ValueError("No model selected. Run select_model.py first.")

    model = load_model(selected_model)

    logger.info(f"Running predictions with '{selected_model}'")
    predictions = model.predict(X_test)

    comparison = pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions,
    })

    logger.info(f"Prediction complete. Shape: {comparison.shape}")
    return comparison


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=params["logging"]["format"]
    )
    logger.info("predict_model.py is not meant to be run directly.")
    logger.info("Import predict_model() into your pipeline and pass X_test, y_test.")
