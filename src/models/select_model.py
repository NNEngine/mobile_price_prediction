from typing import Literal
from pathlib import Path
from src.data.reg_vs_clf import task_type
from src.data.load_data import load_processed_data
import logging
import yaml

logger = logging.getLogger(__name__)

def model_list(task: Literal["classification", "regression"]) -> list:
    """
    Return a list of model names based on the specified machine learning task.

    This function provides a predefined set of commonly used machine learning
    algorithms for either classification or regression tasks.

    Args:
        task (Literal["classification", "regression"]): Type of ML task.

    Returns:
        list: A list of model names (str) corresponding to the given task.

            - For "classification": Returns classifiers such as LogisticRegression,
              RandomForestClassifier, XGBoostClassifier, etc.
            - For "regression": Returns regressors such as LinearRegression,
              RandomForestRegressor, SVR, etc.

    Raises:
        ValueError: If the task is not "classification" or "regression"
            (recommended to enforce externally if needed).

    Example:
        >>> model_list("classification")
        ['LogisticRegression', 'RidgeClassifier', 'SVC', ...]

        >>> model_list("regression")
        ['LinearRegression', 'Ridge', 'Lasso', ...]
    """

    if task == "classification":
        return [
            "LogisticRegression",
            "RidgeClassifier",
            "SVC",
            "KNeighborsClassifier",
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "GradientBoostingClassifier",
            "XGBoostClassifier",
            "LightGBMClassifier",
            "CatBoostClassifier",
            "AdaBoostClassifier",
        ]
    return [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "KNeighborsRegressor",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "GradientBoostingRegressor",
        "XGBoostRegressor",
        "LightGBMRegressor",
        "CatBoostRegressor",
        "AdaBoostRegressor",
        "SVR",
    ]

if __name__ == "__main__":
    # opening params.yaml
    ROOT = Path(__file__).resolve().parents[2]
    with open(ROOT / "params.yaml", "r") as f:
        params = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format=params["logging"]["format"]
    )

    processed_df = load_processed_data()
    result = task_type(processed_df[params["target"]["column"]], "processed")
    logger.info(model_list(result["type"]))
