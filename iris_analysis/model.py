"""Neural-network classifier for Iris."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from iris_analysis.data import CLASS_NAMES, FEATURES, TARGET


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split returning ``X_train, X_test, y_train, y_test``."""
    X = df[FEATURES].to_numpy()
    y = df[TARGET].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy()
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_model(
    hidden_layers: tuple[int, ...] = (16, 8),
    random_state: int = 42,
    max_iter: int = 2000,
) -> MLPClassifier:
    """Create the MLP classifier used in the experiment."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.01,
        max_iter=max_iter,
        random_state=random_state,
    )


def train_model(
    model: MLPClassifier, X_train: np.ndarray, y_train: np.ndarray
) -> tuple[MLPClassifier, StandardScaler]:
    """Standardize features on the training split, then fit the network."""
    scaler = StandardScaler()
    model.fit(scaler.fit_transform(X_train), y_train)
    return model, scaler


def evaluate_model(
    model: MLPClassifier,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Compute accuracy, macro precision/recall/F1, confusion matrix and report."""
    y_pred = model.predict(scaler.transform(X_test))
    labels = list(range(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "y_pred": y_pred,
        "y_test": y_test,
    }
