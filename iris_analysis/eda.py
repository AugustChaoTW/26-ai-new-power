"""Exploratory data analysis and PCA for the Iris dataset."""

from typing import Any

import pandas as pd
from sklearn.decomposition import PCA

from iris_analysis.data import CLASS_NAMES, FEATURES, TARGET


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return headline facts used in the report."""
    counts = df[TARGET].value_counts().to_dict()
    return {
        "n_samples": int(len(df)),
        "n_features": len(FEATURES),
        "n_classes": int(df[TARGET].nunique()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {name: int(counts.get(name, 0)) for name in CLASS_NAMES},
        "dtypes": {column: str(df[column].dtype) for column in FEATURES + [TARGET]},
    }


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics (count/mean/std/min/quartiles/max) per feature."""
    return df[FEATURES].describe().T


def detect_outliers(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Flag rows with at least one value outside 1.5 * IQR of a feature."""
    features = features or FEATURES
    counts = pd.Series(0, index=df.index, dtype=int)
    for column in features:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        counts += ((df[column] < lower) | (df[column] > upper)).astype(int)
    return pd.DataFrame(
        {"outlier_count": counts, "is_outlier": counts > 0}, index=df.index
    )


def class_balance(df: pd.DataFrame) -> pd.Series:
    """Sample count per class, ordered by the canonical class names."""
    return df[TARGET].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)


def perform_pca(
    df: pd.DataFrame, n_components: int = 3, features: list[str] | None = None
) -> dict[str, Any]:
    """Project the features onto ``n_components`` principal components."""
    features = features or FEATURES
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(df[features].to_numpy())
    return {
        "pca": pca,
        "transformed": transformed,
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "feature_names": features,
    }
