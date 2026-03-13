import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_values(df):
    """Shows if your dataset has missing values."""
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Values Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.show()

def plot_feature_distribution(df, column):
    """Use it to detect:
        - skewed data
        - extreme values
        - abnormal distributions
        Example usage:
            plot_feature_distribution(df, "ram")
            plot_feature_distribution(df, "battery_power")
    """

    plt.figure(figsize=(8,5))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def plot_boxplot(df, column):
    """
    Helps detect:
        - outliers
        - extreme values
        - abnormal distributions
    """

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

def plot_target_distribution(df, target_column):
    """
    Detects class imbalance.
    Example:
        plot_target_distribution(df, "price_range")
    """

    plt.figure(figsize=(6,4))
    sns.countplot(x=df[target_column])
    plt.title("Target Class Distribution")
    plt.show()

def plot_correlation_matrix(df):
    """
    Shows relationships between features.
    Helps detect:
        - redundant features
        - multicollinearity
    """

    plt.figure(figsize=(12,8))
    sns.heatmap(
        df.corr(),
        annot=False,
        cmap="coolwarm"
    )
    plt.title("Feature Correlation Matrix")
    plt.show()

def plot_pairwise_relationship(df, columns):
    """
    Shows relationships between multiple features.
    Example:
        plot_pairwise_relationship(df, ["ram", "battery_power", "px_height", "price_range"])
    """

    sns.pairplot(df[columns])
    plt.show()

def plot_categorical_distribution(df, column):
    """
    For binary or categorical columns.
    Example:
        plot_categorical_distribution(df, "blue")
        plot_categorical_distribution(df, "dual_sim")
    """

    plt.figure(figsize=(6,4))
    sns.countplot(x=df[column])
    plt.title(f"Distribution of {column}")
    plt.show()

def plot_feature_vs_target(df, feature, target):
    """
    Useful for feature importance intuition.
    Example:
        plot_feature_vs_target(df, "ram", "price_range")

    You will often see ram strongly correlated with price_range.
    """

    plt.figure(figsize=(8,5))
    sns.boxplot(x=df[target], y=df[feature])
    plt.title(f"{feature} vs {target}")
    plt.show()

def plot_all_numeric_distributions(df):
    """Plot distributions for all numeric columns."""

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    df[numeric_cols].hist(figsize=(14,10), bins=20)
    plt.suptitle("Numeric Feature Distributions")
    plt.show()
