import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from src.data.load_data import load_raw_data
import tqdm


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
				print(f"{col:<20} {n_out:>8}", f"[{lower[col]:.2f}, {upper[col]:.2f}]")
			if self.strategy == "replace" and n_out > 0:
				median = X[col].median()
				X.loc[mask, col] = median

		if not found_any:
			print("No Outliers Found!")
		elif self.strategy == "replace":
			print("\n Outliers  replaced  with  column median")

		return X


class DuplicateChecker(BaseEstimator, TransformerMixin):
	"""Reports Duplicate Per Column, Optionally drop them"""

	def __init__(self, strategy="report", subset=None, keep="first"):
		"""
		strategy : 'report' | 'drop'
		subset : Columns to Consider (None = all)
		keep : 'first' | 'last' | False
		"""
		self.strategy = strategy
		self.subset = subset
		self.keep = keep

	def fit(self, X, y=None):
		if not isinstance(X, pd.DataFrame):
			raise TypeError("Input must be pandas DataFrame")

		return self

	def transform(self,X):
		if not isinstance(X, pd.DataFrame):
			raise TypeError("Input must be pandas DataFrame")

		X = X.copy()
		self.n_duplicates_ = X.duplicated(subset=self.subset).sum()

		print("\n----------------------CHECKING DUPLICATES-----------------------")
		print(f"Total Rows: {len(X)}")
		print(f"Number of Duplicated Rows Found: {self.n_duplicates_}")

		if self.strategy == "drop" and self.n_duplicates_ > 0:
			X = X.drop_duplicates(subset=self.subset, keep=self.keep)
			print(f"Removed Duplicated. Remaining rows: {len(X)}")

		return X


class DataTypeChecker(BaseEstimator, TransformerMixin):
	"""Reports column data types and flags suspicious mixed-type columns."""

	def __init__(self, expected_dtypes=None):
		"""
		expected_dtypes : dict {col: dtype_str}, e.g. {'age': 'int64', 'price': 'float64'}
		"""
		self.expected_dtypes = expected_dtypes or {}

	def fit(self, X, y=None):
		if not isinstance(X, pd.DataFrame):
			raise TypeError("Input must be pandas DataFrame")

		return self

	def transform(self, X):
		if not isinstance(X, pd.DataFrame):
				raise TypeError("Input must be pandas DataFrame")

		print("\n-------------------------------DATA TYPE CHECK----------------------------------\n")
		print(X.dtypes.to_string())

		# Warn about object columns that might be numeric
		for col in X.select_dtypes(include="object").columns:
				converted = pd.to_numeric(X[col], errors="coerce")
				if converted.notna().mean() > 0.9:
						print(f"Column '{col}' looks numeric but is stored as object.")

		# Check expected dtypes
		for col, expected in self.expected_dtypes.items():
				if col in X.columns and str(X[col].dtype) != expected:
						print(f"'{col}': expected {expected}, got {X[col].dtype}")

		return X


class CardinalityChecker(BaseEstimator, TransformerMixin):
	"""Reports unique value counts; flags high-cardinality categoricals and constant columns."""

	def __init__(self, high_cardinality_threshold=50):
			self.high_cardinality_threshold = high_cardinality_threshold

	def fit(self, X, y=None):
			return self

	def transform(self, X):
			print("\n----------------------------CARDINALITY CHECK--------------------------\n")
			for col in X.columns:
					n_unique = X[col].nunique()
					dtype = X[col].dtype
					if n_unique == 1:
							print(f"'{col}': constant column (only 1 unique value).")
					elif n_unique == len(X):
							print(f"'{col}': all values unique — possible ID column.")
					elif dtype == "object" and n_unique > self.high_cardinality_threshold:
							print(f"'{col}': high cardinality categorical ({n_unique} unique).")
			print("  Done.")
			return X


#-----------------------------------BUILD THE PIPELINE----------------------------------

def run_pipeline_with_progress(pipeline, X):

    print("\n================ PIPELINE PROGRESS ================\n")

    steps = pipeline.steps
    data = X

    for name, transformer in tqdm.tqdm(steps, desc="Pipeline Progress", unit="step"):

        tqdm.tqdm.write(f"\n▶ Running Step: {name}")

        transformer.fit(data)
        data = transformer.transform(data)

    print("\Pipeline Completed\n")

    return data


def main():

    df = load_raw_data()

    pipeline = Pipeline([
        ("null_check", NullChecker()),
        ("duplicate_check", DuplicateChecker(strategy="drop")),
        ("datatype_check", DataTypeChecker()),
        ("cardinality_check", CardinalityChecker()),
        ("outlier_check", OutlierChecker(strategy="replace"))
    ])

    df_clean = run_pipeline_with_progress(pipeline, df)

    # save processed dataset
    ROOT = Path(__file__).resolve().parents[2]
    processed_path = ROOT / "data" / "processed" / "mobile_price_clean_processed.csv"

    df_clean.to_csv(processed_path, index=False)

    print(f"\nProcessed dataset saved to:\n{processed_path}")


if __name__ == "__main__":
    main()
