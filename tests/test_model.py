"""Tests for the neural-network classifier (written before implementation)."""

import numpy as np
from sklearn.neural_network import MLPClassifier

from iris_analysis.data import load_iris_dataframe
from iris_analysis.model import (
    build_model,
    evaluate_model,
    split_data,
    train_model,
)


def test_split_is_stratified_and_sized():
    X_train, X_test, y_train, y_test = split_data(
        load_iris_dataframe(), test_size=0.2, random_state=42
    )

    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)
    assert y_train.shape == (120,)
    assert y_test.shape == (30,)
    assert np.bincount(y_test).tolist() == [10, 10, 10]


def test_build_model_returns_configured_mlp():
    model = build_model(hidden_layers=(16, 8), random_state=42)

    assert isinstance(model, MLPClassifier)
    assert model.hidden_layer_sizes == (16, 8)
    assert model.random_state == 42


def test_training_learns_iris_and_evaluation_metrics_are_sane():
    df = load_iris_dataframe()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    model, scaler = train_model(build_model(random_state=42), X_train, y_train)
    evaluation = evaluate_model(model, scaler, X_test, y_test)

    assert len(model.loss_curve_) > 1
    assert model.loss_curve_[-1] < model.loss_curve_[0]
    assert evaluation["accuracy"] >= 0.85
    assert 0.0 <= evaluation["precision_macro"] <= 1.0
    assert 0.0 <= evaluation["recall_macro"] <= 1.0
    assert 0.0 <= evaluation["f1_macro"] <= 1.0
    assert evaluation["confusion_matrix"].shape == (3, 3)
    assert evaluation["confusion_matrix"].sum() == 30
    assert len(evaluation["y_pred"]) == 30
