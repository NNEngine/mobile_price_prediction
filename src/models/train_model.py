import logging
import pickle
import joblib
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "params.yaml") as f:
    params = yaml.safe_load(f)

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_model(selected_model: str, task: str) -> Any:
    """
    Instantiates the selected model with parameters from params.yaml.

    Args:
        selected_model: Name of the model e.g. "RandomForestClassifier"
        task: "classification" or "regression"

    Returns:
        Instantiated model object.

    Raises:
        ValueError: If the model name is not found in params.yaml.
    """
    model_params = params["model_params"][task].get(selected_model)
    if model_params is None:
        raise ValueError(f"No params found for '{selected_model}' under task '{task}'")

    # dynamic import based on model name
    model_map = {
        # classification
        "LogisticRegression":           ("sklearn.linear_model",    "LogisticRegression"),
        "RidgeClassifier":              ("sklearn.linear_model",    "RidgeClassifier"),
        "SVC":                          ("sklearn.svm",             "SVC"),
        "KNeighborsClassifier":         ("sklearn.neighbors",       "KNeighborsClassifier"),
        "DecisionTreeClassifier":       ("sklearn.tree",            "DecisionTreeClassifier"),
        "RandomForestClassifier":       ("sklearn.ensemble",        "RandomForestClassifier"),
        "ExtraTreesClassifier":         ("sklearn.ensemble",        "ExtraTreesClassifier"),
        "GradientBoostingClassifier":   ("sklearn.ensemble",        "GradientBoostingClassifier"),
        "XGBoostClassifier":            ("xgboost",                 "XGBClassifier"),
        "LightGBMClassifier":           ("lightgbm",                "LGBMClassifier"),
        "CatBoostClassifier":           ("catboost",                "CatBoostClassifier"),
        "AdaBoostClassifier":           ("sklearn.ensemble",        "AdaBoostClassifier"),
        # regression
        "LinearRegression":             ("sklearn.linear_model",    "LinearRegression"),
        "Ridge":                        ("sklearn.linear_model",    "Ridge"),
        "Lasso":                        ("sklearn.linear_model",    "Lasso"),
        "ElasticNet":                   ("sklearn.linear_model",    "ElasticNet"),
        "KNeighborsRegressor":          ("sklearn.neighbors",       "KNeighborsRegressor"),
        "DecisionTreeRegressor":        ("sklearn.tree",            "DecisionTreeRegressor"),
        "RandomForestRegressor":        ("sklearn.ensemble",        "RandomForestRegressor"),
        "ExtraTreesRegressor":          ("sklearn.ensemble",        "ExtraTreesRegressor"),
        "GradientBoostingRegressor":    ("sklearn.ensemble",        "GradientBoostingRegressor"),
        "XGBoostRegressor":             ("xgboost",                 "XGBRegressor"),
        "LightGBMRegressor":            ("lightgbm",                "LGBMRegressor"),
        "CatBoostRegressor":            ("catboost",                "CatBoostRegressor"),
        "AdaBoostRegressor":            ("sklearn.ensemble",        "AdaBoostRegressor"),
        "SVR":                          ("sklearn.svm",             "SVR"),
    }

    if selected_model not in model_map:
        raise ValueError(f"Model '{selected_model}' is not supported.")

    module_path, class_name = model_map[selected_model]
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**model_params)


def save_model(model: Any, model_name: str) -> None:
    """
    Saves the trained model in both joblib and pickle formats.

    Args:
        model: Trained model object.
        model_name: Name used for the saved files.
    """
    joblib_path = MODELS_DIR / f"{model_name}.joblib"
    pickle_path = MODELS_DIR / f"{model_name}.pkl"

    joblib.dump(model, joblib_path)
    logger.info(f"Model saved (joblib): {joblib_path}")

    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved (pickle): {pickle_path}")


def train_model(X_train, y_train) -> Any:
    """
    Trains the selected model on the provided training data.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained model object.
    """
    selected_model = params["selected_model"]
    task = params["selected_task"]

    if not selected_model:
        raise ValueError("No model selected. Run select_model.py first.")

    logger.info(f"Training '{selected_model}' for task '{task}'")

    model = get_model(selected_model, task)
    model.fit(X_train, y_train)

    logger.info(f"Training complete.")
    save_model(model, selected_model)

    return model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=params["logging"]["format"]
    )
    logger.info("train_model.py is not meant to be run directly.")
    logger.info("Import train_model() into your pipeline and pass X_train, y_train.")
