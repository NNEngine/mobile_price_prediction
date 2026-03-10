import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from src.data.load_data import load_raw_data



class NullChecker(BaseEstimator, TransformerMixin):
	"""Reports null values per column ; optionally drop or fills them."""

	def __init__(self, strategy="report", fill_value=None,threshold=0.5):
		"""
			strategy : 'report' | 'drop_rows' | 'drop_cols' | 'fill'
			fill_value : value used when strategy='fill' (scalar or dict)
			threshold : max null fraction before a column is dropped (drop_cols)
		"""
		self.strategy = strategy
		self.fill_value = fill_value
		self.threshold = threshold

	def fit(self, X, y = None):
		if not isinstance(X,pd.DataFrame):
			raise TypeError("Input must be a pandas DataFrame")

		self.null_counts_ = X.isnull().sum()
		self.null_pct_ = self.null_counts_ / len(X)
		self.cols_to_drop_ = self.null_pct_[self.null_pct_ > self.threshold].index.tolist()

		return self

	def transform(self, X):
		if not isinstance(X,pd.DataFrame):
				raise TypeError("Input must be a pandas DataFrame")

		X = X.copy()
		print("\n-----------------------NUll CHECK-----------------------------\n")

		null_cols = self.null_counts_[self.null_counts_ > 0]
		if len(null_cols) == 0:
				print("No NULL Values Found!")
		else:
				print(null_cols.to_string())

		if self.strategy == "drop_rows":
			before = len(X)
			X = X.dropna()
			print(f"Dropped {before - len(X)} rows with nulls.")
		elif self.strategy == "drop_cols":
			print(f"Dropping Columns > {self.threshold * 100:.0f}% null: {self.cols_to_drop_}")
			X = X.drop(columns = self.cols_to_drop_)
		elif self.strategy == "fill":
			X = X.fillna(self.fill_value)
			print(f"Filled nulls with: {self.fill_value}")

		return X



class OutlierChecker(BaseEstimator, TransformerMixin):
	"""Reports Outlier per column; optionally replace them."""

	def __init__(self, strategy="report", replace_method="IQR", threshold=None):
		"""
			strategy : 'report' | 'replace'
			replace_method : 'IQR' | 'Z-Score' |
			threshold : threshold value of replace_method (like k=1.5 in IQR and z=3 in Z-Score )
		"""
		self.strategy = strategy
		self.replace_method = replace_method

		if threshold is None:
			self.threshold=1.5 if replace_method=="IQR" else 3.0
		else:
			self.threshold = threshold

	def fit(self, X, y=None):
		if not isinstance(X, pd.DataFrame):
			raise TypeError("Input must be a pandas DataFrame")

		num = X.select_dtypes(include=np.number)

		#____________________________IQR METHOD___________________________#
		Q1 = num.quantile(0.25)
		Q3 = num.quantile(0.75)
		IQR = Q3 - Q1
		self.IQR_lower_ = Q1 - self.threshold * IQR
		self.IQR_upper_ = Q3 + self.threshold * IQR

	#____________________________Z-SCORE METHOD__________________________#
		self.mean_ = num.mean()
		self.std_ = num.std()

		self.Z_lower_ = self.mean_  - self.threshold * self.std_
		self.Z_upper_ = self.mean_ + self.threshold * self.std_

		return self

	def transform(self, X):
		if not isinstance(X, pd.DataFrame):
			raise TypeError("Input must be a pandas DataFrame")

		X = X.copy()
		num = X.select_dtypes(include=np.number)

		if self.replace_method == "IQR":
			lower, upper = self.IQR_lower_, self.IQR_upper_
			label = f"IQR (k={self.threshold})"
		else:
			lower, upper = self.Z_lower_, self.Z_upper_
			label = f"Z (k={self.threshold})"

		print(f"\n── Outlier Check ({label}) {'─'*20}")
		print(f"  {'Column':<20} {'Outliers':>8}    Bounds")
		print(f"  {'──────':<20} {'────────':>8}    ──────")

		found_any = False

		for col in num.columns:
			mask = (X[col] < lower[col]) | (X[col] > upper[col])
			n_out = mask.sum()
			if n_out:
				found_any = True
				print(f"  {col:<20} {n_out:>8}", f"[{lower[col]:.2f}, {upper[col]:.2f}]")
			if self.strategy == "replace":
				median = X[col].median()
				X.loc[mask, col] = median

		if not found_any:
			print("No Outliers Found!")
		elif self.strategy == "replace":
			print("\n Outliers  replaced  with  column median")

		return X
