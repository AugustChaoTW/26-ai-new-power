"""Tests for dataset acquisition and persistence (TDD: written before implementation)."""

from pathlib import Path

import pandas as pd
import pytest

from iris_analysis.data import (
    CLASS_NAMES,
    FEATURES,
    TARGET,
    load_dataset,
    load_iris_dataframe,
    save_dataset,
)


def test_load_iris_dataframe_shape_and_columns():
    df = load_iris_dataframe()

    assert df.shape == (150, 5)
    assert list(df.columns) == FEATURES + [TARGET]
    assert df[FEATURES].shape == (150, 4)
    assert df[TARGET].nunique() == 3


def test_class_names_and_balance():
    df = load_iris_dataframe()

    assert sorted(df[TARGET].unique().tolist()) == sorted(CLASS_NAMES)
    assert df[TARGET].value_counts().tolist() == [50, 50, 50]


def test_all_features_are_numeric_and_non_null():
    df = load_iris_dataframe()

    for column in FEATURES:
        assert pd.api.types.is_numeric_dtype(df[column])
    assert df.isna().sum().sum() == 0


def test_save_and_load_roundtrip(tmp_path: Path):
    df = load_iris_dataframe()
    target = tmp_path / "iris.csv"

    returned = save_dataset(df, target)

    assert returned == target
    assert target.exists()
    loaded = load_dataset(target)
    pd.testing.assert_frame_equal(loaded, df)


def test_load_dataset_rejects_missing_columns(tmp_path: Path):
    broken = tmp_path / "broken.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(broken, index=False)

    with pytest.raises(ValueError):
        load_dataset(broken)
