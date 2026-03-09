import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin


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
