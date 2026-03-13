import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curve(train_sizes, train_scores, val_scores):
    """
    Shows train score vs validation score as dataset size increases.
    Helps detect:
        - overfitting
        - underfitting
    """
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, val_mean, label="Validation Score")
    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

def plot_cross_validation_scores(cv_scores):
    """
    Useful when using k-fold cross validation.
    This helps check:
        - model stability
        - variance between folds
    """

    plt.figure(figsize=(6,4))
    plt.plot(cv_scores, marker="o")
    plt.title("Cross Validation Scores")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.show()

def plot_hyperparameter_results(results, param_name):

    params = results["param_" + param_name]
    scores = results["mean_test_score"]
    plt.figure(figsize=(7,5))
    plt.plot(params, scores, marker="o")
    plt.xlabel(param_name)
    plt.ylabel("Validation Score")
    plt.title(f"Hyperparameter Tuning: {param_name}")
    plt.show()

def plot_feature_importance(importances, feature_names):
    """This helps understand which features affect prediction most."""

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

def plot_training_loss(loss_values):
    """if your model provides epoch-wise loss (e.g. neural networks)."""

    plt.figure(figsize=(7,4))
    plt.plot(loss_values)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_residuals(y_true, y_pred):
    """If you're predicting continuous values."""

    residuals = y_true - y_pred
    plt.figure(figsize=(6,5))
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.show()

def plot_model_comparison(model_names, scores):
    """If testing multiple models."""

    plt.figure(figsize=(8,5))
    plt.bar(model_names, scores)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.show()
