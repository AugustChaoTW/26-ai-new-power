"""Failure-prediction models for the AI4I 2020 predictive maintenance dataset.

Purpose
-------
Train and compare four classifiers that predict ``Machine failure`` from the
five process-sensor readings plus the one-hot encoded machine ``Type``:

    1. ``dummy``   : DummyClassifier (most_frequent) — the accuracy-trap
       control. Because only 339 of 10,000 rows are failures, "never predict
       failure" already reaches ~96.6% accuracy while catching zero failures,
       illustrating why accuracy alone is misleading under severe imbalance.
    2. ``logreg``  : LogisticRegression with class_weight="balanced".
    3. ``rf``      : RandomForestClassifier with class_weight="balanced".
    4. ``mlp``     : MLPClassifier (activation="logistic", solver="adam"),
       whose loss_curve_ is recorded for convergence discussion.

Design note — MLP class weighting via equivalent oversampling
-------------------------------------------------------------
sklearn's ``MLPClassifier`` (unlike ``LogisticRegression`` /
``RandomForestClassifier``) does not support a ``class_weight`` parameter, so
the weighting {0: 1, 1: 28} is implemented inside :func:`train_model` as an
equivalent weighted resampling step (:func:`apply_class_weights`): every
minority-class row is replicated 28 times while majority rows are kept once.
Because both weights are integers the replication is exact (no approximate
fractional weighting needed), yielding ~19,352 training rows for the MLP.

Class imbalance reference: 339 failures / 10,000 rows ≈ 1 : 28.5.

Issue tracking: implements issue #8 (work item 5 of #3). Tests:
``tests/test_ai4i_model.py``.

Persistence (conclusion 4 of "morning experiment conclusions")
-------------------------------------------------------------
A saved model without its preprocessing state cannot reproduce predictions
on new data, so :func:`prepare_training_data_with_state` exposes the fitted
``StandardScaler`` that :func:`prepare_training_data` consumes internally,
and :func:`save_preprocessing` / :func:`load_preprocessing` / 
:func:`rebuild_scaler` round-trip its parameters (mean / std / feature
layout) through JSON.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow running this file as a script (`python ai4i_analysis/model.py`).
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ai4i_analysis.data import load_ai4i_dataframe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "Machine failure"

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ONE_HOT_COLUMNS = ["Type_H", "Type_L", "Type_M"]

# Locked feature layout: sensors first, then one-hot columns alphabetically.
FEATURE_COLUMNS = SENSOR_COLUMNS + ONE_HOT_COLUMNS

# Failure-mode flags / identifiers / the target itself are label leakage if
# used as features. "Machine failure" is the strongest leakage of all — the
# five flags are its components — so it is included here (conclusion 1).
LEAKAGE_COLUMNS = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

# MLP equivalent weighting: 339 / 10000 ≈ 1 : 28.5 -> minority class x 28.
CLASS_WEIGHTS = {0: 1.0, 1: 28.0}

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hyperparameters / training settings per model, kept for the paper.
# MLP has no class_weight parameter: weighting is applied via weighted
# resampling inside train_model (see design note in the module docstring).
MODEL_CONFIGS = {
    "dummy": {
        "strategy": "most_frequent",
        "purpose": "accuracy-trap control (always predicts the majority class)",
    },
    "logreg": {
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "scaler": "StandardScaler (fit on train split only)",
    },
    "rf": {
        "n_estimators": 300,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "scaler": "none (tree model, raw features)",
    },
    "mlp": {
        "hidden_layer_sizes": (16, 8),
        "activation": "logistic",
        "solver": "adam",
        "max_iter": 500,
        "random_state": RANDOM_STATE,
        "early_stopping": False,
        "class_weight": {"0": 1, "1": 28, "note": "applied via weighted resampling"},
        "class_balance": "339/10000",
        "scaler": "StandardScaler (fit on train split only)",
        "loss_curve_": "recorded after fit for convergence analysis",
    },
}


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------


def check_no_leakage(feature_columns) -> None:
    """Programmatic leakage assertion.

    Raises ``ValueError`` if any exclusion/identifier column (failure-mode
    flags, ``UDI``, ``Product ID``) leaks into the feature list.
    """
    leaked = [
        column
        for column in LEAKAGE_COLUMNS
        if column in set(feature_columns)
    ]
    if leaked:
        raise ValueError(
            f"Leakage detected: target-derived or identifier columns must "
            f"never be features: {leaked}"
        )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _split_and_scale(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    """Shared split+scale core of :func:`prepare_training_data`.

    Steps: one-hot encode ``Type`` (``Type_H`` / ``Type_L`` / ``Type_M``),
    assemble features in :data:`FEATURE_COLUMNS` order, stratified 80/20
    split, then ``StandardScaler`` fit on the train split only.

    Returns ``(X_train, X_test, y_train, y_test, feature_names, scaler)``
    where the ``X`` matrices are numpy arrays, ``y`` arrays are integer, and
    ``scaler`` is the fitted ``StandardScaler`` (needed by conclusion 4:
    its mean/std are what gets persisted).
    """
    if df is None:
        df = load_ai4i_dataframe()

    one_hot = pd.get_dummies(df["Type"], prefix="Type", dtype=int)
    X = pd.concat([df[SENSOR_COLUMNS], one_hot], axis=1)
    X = X[FEATURE_COLUMNS]

    check_no_leakage(list(X.columns))

    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        np.asarray(X_train, dtype=float),
        np.asarray(X_test, dtype=float),
        np.asarray(y_train, dtype=int),
        np.asarray(y_test, dtype=int),
        list(X.columns),
        scaler,
    )


def prepare_training_data(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build leakage-safe, scaled training data.

    Thin wrapper over :func:`_split_and_scale` that drops the scaler, keeping
    the locked 5-tuple interface ``(X_train, X_test, y_train, y_test,
    feature_names)`` of issue #8 unchanged. Use
    :func:`prepare_training_data_with_state` when the fitted scaler is also
    needed (conclusion 4 persistence).
    """
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        _scaler,
    ) = _split_and_scale(df, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, feature_names


def prepare_training_data_with_state(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    """Same data as :func:`prepare_training_data`, plus the fitted scaler.

    Conclusion 4 of the "morning experiment conclusions" requires the
    preprocessing parameters (mean / std) to be persisted; this is the entry
    point that exposes them so tests / scripts can call
    :func:`save_preprocessing` on the returned ``scaler``.
    """
    return _split_and_scale(df, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Preprocessing persistence (conclusion 4: save / reload the scaler)
# ---------------------------------------------------------------------------


def save_preprocessing(
    path: Path | str,
    scaler: StandardScaler,
    feature_names,
) -> Path:
    """Persist the fitted scaler's parameters and feature layout as JSON.

    Writes ``{"scaler_mean": [...], "scaler_scale": [...],
    "feature_names": [...]}`` with full-precision floats (Python's JSON float
    repr round-trips exactly, so reloaded parameters are bit-identical) and
    returns the written path. A model saved without these parameters cannot
    reproduce its predictions on new data — that is the invariant locked by
    conclusion 4.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "feature_names": [str(name) for name in feature_names],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_preprocessing(path: Path | str) -> dict:
    """Load the JSON payload written by :func:`save_preprocessing`.

    Returns a dict with keys ``{"scaler_mean", "scaler_scale",
    "feature_names"}``; pass it to :func:`rebuild_scaler` to get a usable
    ``StandardScaler`` again.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rebuild_scaler(params: dict) -> StandardScaler:
    """Rebuild a fitted ``StandardScaler`` from :func:`load_preprocessing`.

    Restores ``mean_`` / ``scale_`` (and the derived ``var_``) so the scaler
    can be used directly for ``transform`` — reproducing bit-identical
    transformed matrices and therefore identical model predictions.
    """
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(params["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(params["scaler_scale"], dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


# ---------------------------------------------------------------------------
# Class weighting for MLP (equivalent oversampling)
# ---------------------------------------------------------------------------


def apply_class_weights(X, y, weights: dict = CLASS_WEIGHTS):
    """Replicate rows so the effective class weights match ``weights``.

    Both weights in ``weights`` must be (roundable to) positive integers;
    each row of class ``k`` is repeated ``weights[k]`` times, e.g.
    ``{0: 1, 1: 28}`` gives 28 exact copies of every minority row.
    Supports both numpy arrays and pandas DataFrames for ``X``.
    """
    y_array = np.asarray(y)
    counts = {int(k): max(1, int(round(v))) for k, v in weights.items()}
    repeats = np.array([counts[int(label)] for label in y_array], dtype=int)

    if isinstance(X, pd.DataFrame):
        X_out = X.iloc[np.repeat(np.arange(len(X)), repeats)].reset_index(drop=True)
    else:
        X_array = np.asarray(X)
        X_out = np.repeat(X_array, repeats, axis=0)
    y_out = np.repeat(y_array, repeats)
    return X_out, y_out


# ---------------------------------------------------------------------------
# Model construction and training
# ---------------------------------------------------------------------------


def build_models() -> dict:
    """Build the four estimators of issue #8 with reproducible settings."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="logistic",
            solver="adam",
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=False,
        ),
    }


def train_model(model, X_train, y_train):
    """Fit ``model`` on ``(X_train, y_train)`` and return the fitted estimator.

    ``MLPClassifier`` has no ``class_weight`` parameter, so it is trained on
    the weighted resample produced by :func:`apply_class_weights`
    (:data:`CLASS_WEIGHTS`); all other estimators are fitted directly.
    """
    if isinstance(model, MLPClassifier):
        X_weighted, y_weighted = apply_class_weights(X_train, y_train, CLASS_WEIGHTS)
        model.fit(np.asarray(X_weighted, dtype=float), y_weighted)
    else:
        model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - manual smoke test
    X_train, X_test, y_train, y_test, feature_names = prepare_training_data()
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Positives: train={int(np.asarray(y_train).sum())}, test={int(np.asarray(y_test).sum())}")

    for name, model in build_models().items():
        start = time.perf_counter()
        fitted = train_model(model, X_train, y_train)
        elapsed = time.perf_counter() - start

        note = f"{name}: trained in {elapsed:.2f}s"
        if isinstance(fitted, MLPClassifier) and fitted.loss_curve_ is not None:
            curve = fitted.loss_curve_
            note += (
                f" | loss_curve_ length={len(curve)}"
                f", first loss={curve[0]:.4f}, last loss={curve[-1]:.4f}"
            )
        note += f" | test accuracy={fitted.score(X_test, y_test):.4f}"
        print(note)
