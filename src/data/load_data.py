from pathlib import Path

import pandas as pd
import yaml

# project root
ROOT = Path(__file__).resolve().parents[2]

# opening params.yaml
with open(ROOT / "params.yaml") as f:
    params = yaml.safe_load(f)

# data path
RAW_DATA_PATH = ROOT / params["data"]["raw_data_path"]
PROCESSED_DATA_PATH = ROOT / params["data"]["processed_data_path"]


def load_data(df_path: Path) -> pd.DataFrame:
    """
    Load a CSV file from the specified file path.

    This function verifies that the file exists before attempting to read it
    using pandas. If the file is not found, a FileNotFoundError is raised.

    Args:
        df_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the specified file path does not exist.

    Example:
        >>> from pathlib import Path
        >>> df = load_data(Path("data.csv"))
    """

    if not df_path.exists():
        raise FileNotFoundError(f"File not found at: {df_path}")
    return pd.read_csv(df_path)


def load_raw_data(df_path: Path = RAW_DATA_PATH)->pd.DataFrame:
	"""
    Load raw dataset from the default or specified path.

    This is a wrapper around `load_data` for loading raw data.

    Args:
        df_path (Path, optional): Path to the raw dataset.
            Defaults to RAW_DATA_PATH.

    Returns:
        pd.DataFrame: DataFrame containing the raw data.

    Example:
        >>> df = load_raw_data()
    """

	return load_data(df_path)


def load_processed_data(df_path: Path=PROCESSED_DATA_PATH)->pd.DataFrame:
	"""
    Load processed dataset from the default or specified path.

    This is a wrapper around `load_data` for loading processed data.

    Args:
        df_path (Path, optional): Path to the processed dataset.
            Defaults to PROCESSED_DATA_PATH.

    Returns:
        pd.DataFrame: DataFrame containing the processed data.

    Example:
        >>> df = load_processed_data()
    """

	return load_data(df_path)
