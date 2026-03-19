import logging
from pathlib import Path
from pprint import pformat

import pandas as pd
import yaml

from src.data.load_data import load_processed_data, load_raw_data

logger = logging.getLogger(__name__)

# opening params.yaml
ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "params.yaml") as f:
    params = yaml.safe_load(f)


def task_type(target_column: pd.Series, tag: str, ratio: int = params["task"]["ratio"]) -> dict:
    """
    Determine whether a machine learning task is regression or classification
    based on the uniqueness ratio of the target column.

    The function computes the ratio of total number of samples to the number
    of unique values in the target column. If this ratio is less than or equal
    to a specified threshold, the task is classified as "regression";
    otherwise, it is classified as "classification".

    Args:
        target_column (pd.Series): The target variable column from a dataset.
        tag (str): A label or identifier associated with the task.
        ratio (int, optional): Threshold value used to decide task type.
            Defaults to params["task"]["ratio"].

    Returns:
        dict: A dictionary containing:
            - "tag" (str): The provided task identifier.
            - "type" (str): Either "regression" or "classification".

    Example:
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> task_type(s, tag="example", ratio=2)
        {'tag': 'example', 'type': 'regression'}
    """

    unique_count = int(target_column.nunique())
    column_size = int(target_column.size)

    size_unique_ratio = column_size / unique_count

    return {
        "tag": tag,
        "type": "regression" if size_unique_ratio <= ratio else "classification"
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=params["logging"]["format"]
    )

    datasets = {"raw": load_raw_data(), "processed": load_processed_data()}

    for tag, df in datasets.items():
        logger.info("\n" + pformat(task_type(df[params["target"]["column"]], tag)))
