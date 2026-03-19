from typing import Literal
from pathlib import Path
from src.data.reg_vs_clf import task_type
from src.data.load_data import load_processed_data
import logging
import yaml

logger = logging.getLogger(__name__)

# opening params.yaml
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "params.yaml", "r") as f:
    params = yaml.safe_load(f)


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
        return params["models"]["classification"]
    return params["models"]["regression"]


def select_model(task: Literal["classification", "regression"]) -> str:
    """
    Displays available models with IDs and prompts user to select one.
    Saves the selected model to params.yaml under 'selected_model'.

    Args:
        task: "classification" or "regression"

    Returns:
        str: Name of the selected model.
    """
    models = model_list(task)
    prefix = "mc" if task == "classification" else "mr"
    model_dict = {f"{prefix}{i+1}": model for i, model in enumerate(models)}

    print(f"\nAvailable {task} models:")
    for model_id, model_name in model_dict.items():
        print(f"  [{model_id}] {model_name}")

    while True:
        choice = input("\nEnter model ID: ").strip()
        if choice in model_dict:
            selected = model_dict[choice]
            # save to params.yaml
            params["selected_model"] = selected
            with open(ROOT / "params.yaml", "w") as f:
                yaml.dump(params, f, default_flow_style=False)
            logger.info(f"Selected model: {selected} ({choice})")
            return selected
        print(f"  Invalid ID '{choice}', please choose from the list.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=params["logging"]["format"]
    )

    processed_df = load_processed_data()
    result = task_type(processed_df[params["target"]["column"]], "processed")
    models = model_list(result["type"])
    logger.info(models)
    selected = select_model(result["type"])
    logger.info(f"Model saved to params.yaml: {selected}")
