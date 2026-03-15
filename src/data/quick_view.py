import pandas as pd
from src.data.load_data import load_raw_data,load_processed_data
from pprint import pprint

def data_summary(df:pd.DataFrame, tag:str)-> dict:
	summary = {
		"tag": tag,
		"shape": df.shape,
		"duplicates": int(df.duplicated().sum()),
		"nulls": {col: int(n) for col, n in df.isnull().sum().items() if n > 0},
		"dtypes": df.dtypes.value_counts().to_dict()
	}

	return summary

if __name__ == "__main__":
	datasets = {"raw": load_raw_data(), "processed": load_processed_data()}

	for tag, df in datasets.items():
		pprint(data_summary(df,tag))
