import pandas as pd
from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parents[2]

# data path
RAW_DATA_PATH = ROOT / "data" / "raw" / "raw_data.csv"
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"


def load_raw_data(df_path: Path = RAW_DATA_PATH)->pd.DataFrame:
	"""
	- loads raw data.
	- returns a dataframe (pd.read_csv(df_path))
	"""

	return pd.read_csv(df_path)

def load_processed_data(df_path: Path=PROCESSED_DATA_PATH)->pd.DataFrame:
	"""
	- loads processed data
	- returns a dataframe (pd.read_csv(df_path))
	"""

	return pd.read_csv(df_path)
