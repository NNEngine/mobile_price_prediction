import pandas as pd
from src.data.load_data import load_raw_data, load_processed_data
from pprint import pformat
import logging

logger = logging.getLogger(__name__)

def data_summary(df: pd.DataFrame, tag: str) -> dict:
    return {
        "tag": tag,
        "shape": df.shape,
        "duplicates": int(df.duplicated().sum()),
        "nulls": {col: int(n) for col, n in df.isnull().sum().items() if n > 0},
        "dtypes": {str(k): v for k, v in df.dtypes.value_counts().items()},
        "memory": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s\n"
    )
    
    datasets = {"raw": load_raw_data(), "processed": load_processed_data()}
    for tag, df in datasets.items():
        logger.info("\n" + pformat(data_summary(df, tag)))
