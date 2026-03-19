import pandas as pd
from pathlib import Path
from src.data.load_data import load_raw_data, load_processed_data
from pprint import pformat
import logging
import yaml

logger = logging.getLogger(__name__)

def data_summary(df: pd.DataFrame, tag: str) -> dict:
    """
    Generate a concise summary of a pandas DataFrame.

    This function computes key structural and data quality statistics,
    including shape, duplicate rows, missing values, data type distribution,
    and memory usage.

    Args:
        df (pd.DataFrame): The input DataFrame to summarize.
        tag (str): A label or identifier associated with the dataset.

    Returns:
        dict: A dictionary containing:
            - "tag" (str): The provided dataset identifier.
            - "shape" (tuple): Dimensions of the DataFrame (rows, columns).
            - "duplicates" (int): Number of duplicate rows.
            - "nulls" (dict): Mapping of column names to count of missing values
              (only columns with at least one missing value are included).
            - "dtypes" (dict): Distribution of data types in the DataFrame
              (dtype as key and count as value).
            - "memory" (str): Total memory usage of the DataFrame in MB.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "A": [1, 2, None],
        ...     "B": ["x", "y", "y"]
        ... })
        >>> data_summary(df, tag="sample")
        {
            'tag': 'sample',
            'shape': (3, 2),
            'duplicates': 0,
            'nulls': {'A': 1},
            'dtypes': {'float64': 1, 'object': 1},
            'memory': '0.00 MB'
        }
    """

    return {
        "tag": tag,
        "shape": df.shape,
        "duplicates": int(df.duplicated().sum()),
        "nulls": {col: int(n) for col, n in df.isnull().sum().items() if n > 0},
        "dtypes": {str(k): v for k, v in df.dtypes.value_counts().items()},
        "memory": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
    }

if __name__ == "__main__":
    # opening params.yaml
    ROOT = Path(__file__).resolve().parents[2]
    with open(ROOT / "params.yaml", "r") as f:
        params = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format=params["logging"]["format"]
    )

    datasets = {"raw": load_raw_data(), "processed": load_processed_data()}
    for tag, df in datasets.items():
        logger.info("\n" + pformat(data_summary(df, tag)))
