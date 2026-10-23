"""Figure generation for the Iris study."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iris_analysis.data import FEATURES, TARGET

COLORS = ["#2E86AB", "#E67E22", "#27AE60"]


def _save(fig: plt.Figure, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pca_3d(
    transformed: np.ndarray,
    labels: pd.Series,
    class_names: list[str],
    explained_variance_ratio: np.ndarray,
    output_path: Path | str,
) -> Path:
    """3D scatter of the first three principal components, coloured by class."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    labels = pd.Series(labels).to_numpy()
    for index, name in enumerate(class_names):
        mask = labels == name
        ax.scatter(
            transformed[mask, 0],
            transformed[mask, 1],
            transformed[mask, 2],
            s=45,
            alpha=0.8,
            color=COLORS[index % len(COLORS)],
            label=name,
            edgecolors="white",
            linewidths=0.5,
        )
    ratios = explained_variance_ratio * 100
    ax.set_xlabel(f"PC1 ({ratios[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratios[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({ratios[2]:.1f}%)")
    ax.set_title("Iris PCA 3D Feature Distribution")
    ax.legend(title="Species", loc="upper right")
    ax.view_init(elev=22, azim=135)
    return _save(fig, output_path)


def plot_feature_distributions(
    df: pd.DataFrame, features: list[str] | None = None, output_path: Path | str = "feature_distributions.png"
) -> Path:
    """Histogram of each feature split by species."""
    features = features or FEATURES
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feature in zip(axes.ravel(), features):
        for index, name in enumerate(sorted(df[TARGET].unique())):
            subset = df.loc[df[TARGET] == name, feature]
            ax.hist(
                subset,
                bins=12,
                alpha=0.65,
                label=name,
                color=COLORS[index % len(COLORS)],
                edgecolor="white",
            )
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("cm")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle("Feature Distributions by Species")
    fig.tight_layout()
    return _save(fig, output_path)


def plot_correlation_heatmap(
    df: pd.DataFrame, features: list[str] | None = None, output_path: Path | str = "correlation_heatmap.png"
) -> Path:
    """Pearson correlation heatmap of the numeric features."""
    features = features or FEATURES
    correlations = df[features].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), [f.replace("_", "\n") for f in features])
    ax.set_yticks(range(len(features)), [f.replace("_", " ") for f in features])
    for row in range(len(features)):
        for column in range(len(features)):
            ax.text(
                column,
                row,
                f"{correlations[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(correlations[row, column]) > 0.55 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return _save(fig, output_path)


def plot_learning_curve(loss_curve: list[float], output_path: Path | str) -> Path:
    """Training loss per epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_curve) + 1), loss_curve, color=COLORS[0], linewidth=2)
    ax.set_title("Neural Network Training Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_confusion_matrix(
    matrix: np.ndarray, class_names: list[str], output_path: Path | str
) -> Path:
    """Annotated confusion matrix."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=25)
    ax.set_yticks(range(len(class_names)), class_names)
    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=12,
            )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (test set)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_class_metrics(
    report: dict, class_names: list[str], output_path: Path | str
) -> Path:
    """Grouped bar chart of precision/recall/F1 per class."""
    metric_names = ["precision", "recall", "f1-score"]
    values = np.array(
        [[report[name][metric] for metric in metric_names] for name in class_names]
    )
    positions = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, metric in enumerate(metric_names):
        ax.bar(
            positions + (index - 1) * width,
            values[:, index],
            width,
            label=metric,
            color=COLORS[index % len(COLORS)],
        )
    ax.set_xticks(positions, class_names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("Per-class Classification Metrics")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, output_path)
