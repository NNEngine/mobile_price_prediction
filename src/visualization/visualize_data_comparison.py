import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_feature_distribution(raw_df, processed_df, column):
    """
    Useful for verifying:
        - scaling
        - normalization
        - transformation effects
    """

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    sns.histplot(raw_df[column], kde=True, ax=axes[0])
    axes[0].set_title(f"Raw {column}")
    sns.histplot(processed_df[column], kde=True, ax=axes[1])
    axes[1].set_title(f"Processed {column}")
    plt.suptitle(f"Distribution Comparison: {column}")
    plt.show()


def compare_boxplots(raw_df, processed_df, column):
    """
    Shows:
    - removed outliers
    - scaled ranges
    """
    combined = pd.DataFrame({
        "value": list(raw_df[column]) + list(processed_df[column]),
        "dataset": ["Raw"]*len(raw_df) + ["Processed"]*len(processed_df)
    })
    plt.figure(figsize=(6,5))
    sns.boxplot(x="dataset", y="value", data=combined)
    plt.title(f"Boxplot Comparison: {column}")
    plt.show()

def compare_boxplots(raw_df, processed_df, column):
    """
    Useful to detect:
    - feature leakage
    - unexpected relationships
    """

    combined = pd.DataFrame({
        "value": list(raw_df[column]) + list(processed_df[column]),
        "dataset": ["Raw"]*len(raw_df) + ["Processed"]*len(processed_df)
    })
    plt.figure(figsize=(6,5))
    sns.boxplot(x="dataset", y="value", data=combined)
    plt.title(f"Boxplot Comparison: {column}")
    plt.show()

def compare_statistics(raw_df, processed_df):
    """
    Compare mean and variance.
    Useful for:
        - verifying scaling
        - checking normalization
    """

    raw_stats = raw_df.describe().T
    processed_stats = processed_df.describe().T
    comparison = pd.DataFrame({
        "raw_mean": raw_stats["mean"],
        "processed_mean": processed_stats["mean"],
        "raw_std": raw_stats["std"],
        "processed_std": processed_stats["std"]
    })
    return comparison

def compare_missing_values(raw_df, processed_df):

    raw_missing = raw_df.isnull().sum()
    processed_missing = processed_df.isnull().sum()
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    raw_missing.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Missing Values (Raw)")
    processed_missing.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Missing Values (Processed)")
    plt.show()

def compare_feature_count(raw_df, processed_df):

    print("Raw Features:", len(raw_df.columns))
    print("Processed Features:", len(processed_df.columns))

def compare_scatter(raw_df, processed_df, column):
    """Useful for verifying scaling / log transforms."""
    
    plt.figure(figsize=(6,5))
    plt.scatter(raw_df[column], processed_df[column])
    plt.xlabel("Raw")
    plt.ylabel("Processed")
    plt.title(f"Raw vs Processed: {column}")
    plt.show()
