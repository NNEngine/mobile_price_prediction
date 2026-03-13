import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrix(y_true, y_pred):
    """Shows where the model is making mistakes."""

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_prediction_distribution(y_true, y_pred):
    """Helps detect bias in predictions."""
    df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    df["Actual"].value_counts().sort_index().plot(
        kind="bar",
        ax=axes[0]
    )
    axes[0].set_title("Actual Distribution")
    df["Predicted"].value_counts().sort_index().plot(
        kind="bar",
        ax=axes[1]
    )
    axes[1].set_title("Predicted Distribution")
    plt.show()

def plot_roc_curve(y_true, y_prob):
    """Used to measure classification performance."""

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_precision_recall(y_true, y_prob):
    """Better when classes are imbalanced."""

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

def plot_classification_metrics(report):
    """Instead of printing the report, visualize it."""

    df = pd.DataFrame(report).T
    df[["precision","recall","f1-score"]].plot(
        kind="bar",
        figsize=(8,5)
    )
    plt.title("Classification Metrics")
    plt.show()

def plot_prediction_confidence(probabilities):
    """Shows how confident the model is."""

    confidence = np.max(probabilities, axis=1)
    plt.figure(figsize=(6,4))
    plt.hist(confidence, bins=20)
    plt.title("Prediction Confidence")
    plt.xlabel("Confidence")
    plt.show()

def plot_prediction_errors(y_true, y_pred):
    """Focus on incorrect predictions."""
    
    errors = y_true != y_pred
    plt.figure(figsize=(6,4))
    plt.hist(errors.astype(int), bins=2)
    plt.title("Prediction Errors")
    plt.show()
