"""Iris dataset acquisition and persistence."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_iris_dataframe() -> pd.DataFrame:
    """Load the bundled scikit-learn Iris dataset as a labelled DataFrame."""
    bunch = load_iris()
    df = pd.DataFrame(bunch.data, columns=FEATURES)
    df[TARGET] = [CLASS_NAMES[index] for index in bunch.target]
    return df


def save_dataset(df: pd.DataFrame, path: Path | str) -> Path:
    """Persist the dataset as CSV and return the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load a previously saved Iris CSV and validate its schema."""
    path = Path(path)
    df = pd.read_csv(path)
    expected = FEATURES + [TARGET]
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset {path} is missing columns: {missing}")
    return df[expected]
