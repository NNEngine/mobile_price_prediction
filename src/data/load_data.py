import pandas as pd
from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parents[2]

# data path
DATA_PATH = ROOT / "data" / "raw" / "mobile_price_prediction.csv"

def load_raw_data(df_path: Path = DATA_PATH)->pd.DataFrame:
	return pd.read_csv(df_path)
