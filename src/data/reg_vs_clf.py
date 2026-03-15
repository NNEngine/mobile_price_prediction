import pandas as pd
import logging
from pprint import pformat
from src.data.load_data import load_raw_data, load_processed_data

logger = logging.getLogger(__name__)

def task_type(target_column: pd.Series, tag: str, ratio: int = 10) -> dict:
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
        format="%(asctime)s - %(levelname)s - %(message)s\n"
    )

	datasets = {"raw": load_raw_data(), "processed": load_processed_data()}
	TARGET = "price_range"

	for tag, df in datasets.items():
		logger.info("\n" + pformat(task_type(df[TARGET], tag)))
