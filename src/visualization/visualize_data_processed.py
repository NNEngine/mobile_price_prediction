import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scaled_distribution(df, column):
    """
    If you apply StandardScaler / MinMaxScaler, you should verify the transformation.
    Example: histogram of scaled features.
    """

    plt.figure(figsize=(7,4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Scaled Distribution of {column}")
    plt.xlabel(column)
    plt.show()

def plot_processed_correlation(df):
    """Sometimes preprocessing introduces new correlations."""

    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Correlation Matrix (Processed Data)")
    plt.show()

def plot_feature_importance(importances, feature_names):
    """
    After training models like:
        - RandomForest
        - XGBoost
        - GradientBoosting
    You visualize importance.
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10,6))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    plt.gca().invert_yaxis()
    plt.title("Feature Importance")
    plt.show()

def plot_pca_projection(X_pca, y):
    """If you reduce features using PCA, visualize the space."""

    plt.figure(figsize=(7,6))
    plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=y
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection of Processed Data")
    plt.show()

def plot_processed_distributions(df):
    """After preprocessing, features should look cleaner and normalized."""
    df.hist(
        figsize=(14,10),
        bins=20
    )
    plt.suptitle("Processed Feature Distributions")
    plt.show()

def plot_feature_separation(df, feature, target):
    """
    Very useful for classification datasets.
    Example: feature vs target.
    """

    plt.figure(figsize=(7,4))
    sns.boxplot(
        x=df[target],
        y=df[feature]
    )
    plt.title(f"{feature} vs {target} (Processed)")
    plt.show()

def plot_feature_variance(df):
    """Low variance features are useless for ML."""

    variances = df.var().sort_values()
    plt.figure(figsize=(10,5))
    variances.plot(kind="bar")
    plt.title("Feature Variance")
    plt.show()

def plot_processed_pairplot(df, columns):
    """Useful for checking cleaned feature relationships."""
    sns.pairplot(df[columns])
