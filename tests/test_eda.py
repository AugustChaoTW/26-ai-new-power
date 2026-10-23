"""Tests for exploratory data analysis and PCA (written before implementation)."""

import numpy as np
import pandas as pd

from iris_analysis.data import load_iris_dataframe
from iris_analysis.eda import (
    class_balance,
    dataset_overview,
    describe_features,
    detect_outliers,
    perform_pca,
)


def test_dataset_overview_contains_key_facts():
    overview = dataset_overview(load_iris_dataframe())

    assert overview["n_samples"] == 150
    assert overview["n_features"] == 4
    assert overview["n_classes"] == 3
    assert overview["missing_values"] == 0
    assert overview["duplicate_rows"] >= 0
    assert overview["class_counts"] == {"setosa": 50, "versicolor": 50, "virginica": 50}


def test_describe_features_has_expected_statistics():
    stats = describe_features(load_iris_dataframe())

    assert stats.shape[0] == 4
    for column in ("mean", "std", "min", "50%", "max"):
        assert column in stats.columns
    assert stats.loc["petal_length", "mean"] > stats.loc["sepal_width", "mean"]


def test_detect_outliers_marks_rows_and_keeps_shape():
    df = load_iris_dataframe()
    outliers = detect_outliers(df)

    assert isinstance(outliers, pd.DataFrame)
    assert outliers.shape[0] == len(df)
    assert set(outliers.columns) == {"outlier_count", "is_outlier"}
    assert outliers["is_outlier"].dtype == bool
    # Iris contains a few known univariate outliers; must not flag everything.
    assert 0 < outliers["is_outlier"].sum() < 30


def test_class_balance_sums_to_sample_count():
    balance = class_balance(load_iris_dataframe())

    assert isinstance(balance, pd.Series)
    assert balance.sum() == 150
    assert (balance == 50).all()


def test_perform_pca_returns_three_components():
    result = perform_pca(load_iris_dataframe(), n_components=3)

    assert result["transformed"].shape == (150, 3)
    assert result["components"].shape == (3, 4)
    ratios = result["explained_variance_ratio"]
    assert ratios.shape == (3,)
    assert np.all(np.diff(ratios) <= 0)
    assert 0.9 < ratios.sum() <= 1.0
    assert ratios[0] > 0.7
