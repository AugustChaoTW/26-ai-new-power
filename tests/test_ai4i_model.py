"""TDD tests for ai4i_analysis.model — written before implementation (issue #8 / work item 5 of #3).

The module (and every symbol imported below) does not exist yet, so this file
must fail at collection time with ``ModuleNotFoundError``. That is the
intended red step of the TDD plan.

Locked contract (from issue #8):
    - MODEL_CONFIGS      : dict[str, dict] with hyperparameters for the paper
    - CLASS_WEIGHTS      : MLP class weighting {0: 1.0, 1: 28.0}
    - FEATURE_COLUMNS    : the 8 training feature names
    - prepare_training_data : minimal preprocessing (leakage-safe, scaler fits
      on train only), returns (X_train, X_test, y_train, y_test, feature_names)
    - build_models       : dict with dummy / logreg / rf / mlp estimators
    - train_model        : fits a model; for MLP performs class-weight
      resampling internally (MLPClassifier.fit has no class_weight argument)
    - apply_class_weights : replicate minority class rows, equivalent to
      class_weight effects
"""

import time

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from ai4i_analysis.model import (
    CLASS_WEIGHTS,
    FEATURE_COLUMNS as _FEATURE_COLUMNS,
    MODEL_CONFIGS as _MODEL_CONFIGS,
    apply_class_weights,
    build_models,
    prepare_training_data,
    train_model,
)

# ---------------------------------------------------------------------------
# Expected feature layout (locked down from issue #8 spec)
# ---------------------------------------------------------------------------

EXPECTED_SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# one-hot payload columns ordered alphabetically (sklearn / pandas convention)
EXPECTED_ONEHOT_COLUMNS = ["Type_H", "Type_L", "Type_M"]

EXPECTED_FEATURE_NAMES = EXPECTED_SENSOR_COLUMNS + EXPECTED_ONEHOT_COLUMNS

LEAKAGE_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


@pytest.fixture(scope="module")
def prepared():
    """Split + scaled training data, computed once for the module."""
    return prepare_training_data()


@pytest.fixture(scope="module")
def models():
    return build_models()


# ---------------------------------------------------------------------------
# 1. build_models contract
# ---------------------------------------------------------------------------


def test_build_models_returns_four_models(models):
    assert dict(models) == {"dummy": models["dummy"], "logreg": models["logreg"], "rf": models["rf"], "mlp": models["mlp"]}
    assert set(models) == {"dummy", "logreg", "rf", "mlp"}

    assert isinstance(models["dummy"], DummyClassifier)
    assert isinstance(models["logreg"], LogisticRegression)
    assert isinstance(models["rf"], RandomForestClassifier)
    assert isinstance(models["mlp"], MLPClassifier)

    # accuracy-trap baseline must be deterministic
    assert models["dummy"].strategy == "most_frequent"
    assert models["dummy"].random_state is None

    assert models["logreg"].class_weight == "balanced"
    assert models["logreg"].max_iter == 1000
    assert models["logreg"].random_state == 42

    assert models["rf"].class_weight == "balanced"
    assert models["rf"].n_estimators == 300
    assert models["rf"].random_state == 42
    assert models["rf"].n_jobs == -1

    # MLP cannot take class_weight in fit(); weighting happens in train_model
    assert models["mlp"].activation in {"logistic", "tanh"}
    assert models["mlp"].solver == "adam"
    assert models["mlp"].hidden_layer_sizes == (16, 8)
    assert models["mlp"].random_state == 42
    assert models["mlp"].max_iter == 500
    assert models["mlp"].early_stopping is False


def test_model_configurations_documented():
    # constants exposed for the paper
    assert set(_MODEL_CONFIGS) == {"dummy", "logreg", "rf", "mlp"}
    for name, config in _MODEL_CONFIGS.items():
        assert isinstance(config, dict), name
        assert len(config) > 0, name

    mlp_config = _MODEL_CONFIGS["mlp"]
    for key in ["activation", "solver", "class_weight", "random_state"]:
        assert key in mlp_config, key

    logreg_config = _MODEL_CONFIGS["logreg"]
    assert logreg_config.get("class_weight") == "balanced"

    assert CLASS_WEIGHTS == {0: 1.0, 1: 28.0}
    assert list(_FEATURE_COLUMNS) == EXPECTED_FEATURE_NAMES
    assert len(_FEATURE_COLUMNS) == 8


# ---------------------------------------------------------------------------
# 2. prepare_training_data contract
# ---------------------------------------------------------------------------


def test_prepare_training_data_shapes_and_features(prepared):
    X_train, X_test, y_train, y_test, feature_names = prepared

    assert X_train.shape == (8000, 8)
    assert X_test.shape == (2000, 8)
    assert list(feature_names) == EXPECTED_FEATURE_NAMES
    # feature order is locked: sensors first, then one-hot alphabetically
    assert list(feature_names) == EXPECTED_SENSOR_COLUMNS + sorted(EXPECTED_ONEHOT_COLUMNS)

    # leakage lock: none of the excluded columns may reappear
    for column in LEAKAGE_COLUMNS:
        assert column not in feature_names, column
    assert feature_names.count("Type") == 0  # raw categorical replaced by one-hot


def test_split_is_stratified(prepared):
    X_train, X_test, y_train, y_test, feature_names = prepared

    # whole dataset has 339 positives -> stratified 80/20 gives 271 / 68
    assert int(np.asarray(y_test).sum()) == 68
    assert int(np.asarray(y_train).sum()) == 271

    y_train_arr, y_test_arr = np.asarray(y_train), np.asarray(y_test)
    assert set(np.unique(y_train_arr)) <= {0, 1}
    assert set(np.unique(y_test_arr)) <= {0, 1}
    for arr in (y_train_arr, y_test_arr):
        assert np.issubdtype(arr.dtype, np.integer), arr.dtype


def test_scaler_uses_train_only(prepared):
    X_train, X_test, y_train, y_test, feature_names = prepared

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # scaler fit on the training split only -> train means ~0
    train_abs_means = np.abs(X_train.mean(axis=0))
    assert (train_abs_means < 1e-9).all()

    # ...and train stds close to 1 (guard against an over-loose scaler)
    train_stds = X_train.std(axis=0)
    assert (train_stds < 1.01).all()
    assert (train_stds > 0.9).all()

    # the test split was only transformed, so its column means are NOT 0
    test_abs_means = np.abs(X_test.mean(axis=0))
    assert (test_abs_means > 1e-9).any()


# ---------------------------------------------------------------------------
# 3. Baseline behaviour on the real data
# ---------------------------------------------------------------------------


def test_dummy_predicts_no_positives(prepared, models):
    X_train, X_test, y_train, y_test, feature_names = prepared

    dummy = build_models()["dummy"]
    fitted = train_model(dummy, X_train, y_train)
    predictions = fitted.predict(X_test)

    assert predictions.sum() == 0


def test_train_model_fits_all_four(prepared):
    X_train, X_test, y_train, y_test, feature_names = prepared

    for name, model in build_models().items():
        start = time.perf_counter()
        fitted = train_model(model, X_train, y_train)
        elapsed = time.perf_counter() - start
        print(f"[timing] {name}: {elapsed:.2f}s")

        assert hasattr(fitted, "predict"), name
        assert fitted.predict(X_test[:5]).shape == (5,)
        assert elapsed < 180.0, name


def test_mlp_loss_curve_converges(prepared):
    X_train, X_test, y_train, y_test, feature_names = prepared

    mlp = build_models()["mlp"]
    fitted = train_model(mlp, X_train, y_train)

    loss_curve = fitted.loss_curve_
    assert loss_curve is not None
    assert len(loss_curve) >= 10

    curve = list(loss_curve)
    assert curve[-1] < curve[0]

    quarter = max(len(curve) // 4, 1)
    first_quarter = float(np.mean(curve[:quarter]))
    last_quarter = float(np.mean(curve[-quarter:]))
    assert last_quarter < first_quarter


def test_models_reproducible(prepared):
    X_train, X_test, y_train, y_test, feature_names = prepared

    for name in ["dummy", "logreg", "rf"]:
        first = train_model(build_models()[name], X_train, y_train)
        second = train_model(build_models()[name], X_train, y_train)
        assert np.array_equal(first.predict(X_test), second.predict(X_test)), name


# ---------------------------------------------------------------------------
# 4. MLP class-weighting fallback
# ---------------------------------------------------------------------------


def test_apply_class_weights_mlp_fallback():
    rng = np.random.RandomState(42)
    X = rng.rand(100, 8)
    y = np.array([1] * 5 + [0] * 95)

    X_weighted, y_weighted = apply_class_weights(X, y, CLASS_WEIGHTS)
    y_weighted = np.asarray(y_weighted)

    # 5 positives x 28 -> 140 positives, 95 negatives untouched
    assert X_weighted.shape[0] == 95 + 140
    assert int((y_weighted == 1).sum()) == 140
    assert int((y_weighted == 0).sum()) == 95
    assert set(np.unique(y_weighted)) == {0, 1}
    assert X_weighted.shape[1] == 8

    # identity weights must not duplicate anything
    X_same, y_same = apply_class_weights(X, y, {0: 1, 1: 1})
    assert np.asarray(X_same).shape == (100, 8)
    assert np.array_equal(np.asarray(y_same), y)

    # any non-trivial weighting grows the output
    X_any, y_any = apply_class_weights(X, y, {0: 1, 1: 2})
    assert np.asarray(X_any).shape[0] > 100


def test_train_model_applies_class_weights_to_mlp(monkeypatch, prepared):
    import importlib

    ai4i_model = importlib.import_module("ai4i_analysis.model")
    X_train, X_test, y_train, y_test, feature_names = prepared

    original_apply = ai4i_model.apply_class_weights
    calls = {"count": 0, "kwargs": None}

    def spy(X, y, weights):
        calls["count"] += 1
        calls["kwargs"] = weights
        return original_apply(X, y, weights)

    monkeypatch.setattr(ai4i_model, "apply_class_weights", spy)

    mlp = build_models()["mlp"]
    fitted = train_model(mlp, X_train, y_train)

    # train_model must route the MLP through the weighting resample exactly once
    assert calls["count"] == 1
    assert calls["kwargs"] == CLASS_WEIGHTS
    # sanity: the returned estimator is fitted
    assert hasattr(fitted, "predict")
