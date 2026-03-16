from typing import Literal
from src.data.reg_vs_clf import task_type
from src.data.load_data import load_processed_data
import logging

logger = logging.getLogger(__name__)

TARGET = "price_range"

def model_list(task: Literal["classification", "regression"]) -> list:
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s\n"
    )

    processed_df = load_processed_data()
    result = task_type(processed_df[TARGET], "processed")
    logger.info(model_list(result["type"]))
