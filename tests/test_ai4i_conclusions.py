"""TDD tests for the four "morning experiment conclusions" invariants.

Unlike the other ai4i test modules — which lock function contracts — these
tests assert the EXPERIMENT CONCLUSIONS themselves, so that any future change
that silently violates a reported conclusion turns red:

  1. The AI4I feature list never contains label leakage (TWF / HDF / PWF /
     OSF / RNF) nor the target ``Machine failure``.
  2. The three real models are not majority-class-only: every one of
     logreg / rf / mlp achieves recall > 0 and emits BOTH classes.
  3. Every number in ``docs/results.json`` that is marked as sourced from
     ``reports/paper.md`` appears verbatim in ``reports/paper.md`` (and the
     four Iris metrics match the paper's table exactly); every ai4i number is
     marked with its source file.
  4. The preprocessing parameters (scaler mean/std) can be persisted and
     reloaded, and reloading them reproduces IDENTICAL model predictions.

Testing-strategy note for conclusion 2 — the ``dummy`` estimator is
deliberately EXCLUDED: it is the accuracy-trap control group of issue #8 /
MODEL_CONFIGS["dummy"] ("always predicts the majority class"), i.e. it fails
this invariant BY DESIGN, which is exactly the point it demonstrates. It is
not a real model and therefore not part of this conclusion's scope; see issue
#8's baseline discussion.

Fixture style mirrors tests/test_ai4i_evaluate.py: one module-scope fixture
prepares the split and trains the three real models exactly once (rf ~6s,
mlp ~30-50s) so all conclusion tests reuse the same fitted estimators.

Written before implementation (conclusion-4 persistence APIs
``prepare_training_data_with_state`` / ``save_preprocessing`` /
``load_preprocessing`` / ``rebuild_scaler`` do not exist yet), so the red
step fails at collection time with ImportError — the intended TDD signal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from ai4i_analysis.data import load_ai4i_dataframe
from ai4i_analysis.model import (
    FEATURE_COLUMNS as _FEATURE_COLUMNS,
    SENSOR_COLUMNS as _SENSOR_COLUMNS,
    TARGET as _TARGET,
    build_models,
    check_no_leakage,
    load_preprocessing,
    prepare_training_data,
    prepare_training_data_with_state,
    rebuild_scaler,
    save_preprocessing,
    train_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "docs" / "results.json"
PAPER_PATH = REPO_ROOT / "reports" / "paper.md"

# Columns that must NEVER be features: the five failure-mode flags are
# components of the target, and "Machine failure" is the target itself.
FORBIDDEN_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"]

REAL_MODEL_NAMES = ["logreg", "rf", "mlp"]  # dummy excluded by design


# ---------------------------------------------------------------------------
# Module-scope fixtures: prepare + train the three real models exactly once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_real_models():
    """Split + the three REAL fitted models, computed once for the module.

    Returns ``(X_test, y_test, fitted_models)``. The ``dummy`` accuracy-trap
    control (issue #8) is deliberately excluded here: it always predicts the
    majority class by design and is therefore out of scope for conclusion 2,
    which only covers models that actually learn the signal.
    """
    X_train, X_test, y_train, y_test, _feature_names = prepare_training_data()
    fitted_models = {
        name: train_model(build_models()[name], X_train, y_train)
        for name in REAL_MODEL_NAMES
    }
    return X_test, y_test, fitted_models


@pytest.fixture(scope="module")
def prepared_with_state():
    """``prepare_training_data_with_state()`` payload computed once."""
    return prepare_training_data_with_state()


def _fake_ai4i_dataframe(rows: int = 21) -> pd.DataFrame:
    """A tiny synthetic df containing the leakage columns AND the target.

    Needed to prove ``prepare_training_data`` drops them from the feature
    layout even when they are present in the input data.
    """
    rng = np.random.RandomState(42)
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "UDI": np.arange(1, rows + 1),
            "Product ID": [f"L{i}" for i in index],
            "Type": rng.choice(["H", "L", "M"], size=rows),
            "Air temperature [K]": rng.normal(298.0, 2.0, rows),
            "Process temperature [K]": rng.normal(308.0, 2.0, rows),
            "Rotational speed [rpm]": rng.normal(1500.0, 100.0, rows),
            "Torque [Nm]": rng.normal(45.0, 10.0, rows),
            "Tool wear [min]": rng.randint(0, 250, rows),
            "Machine failure": (index % 7 == 0).astype(int),
            "TWF": (index % 11 == 0).astype(int),
            "HDF": (index % 13 == 0).astype(int),
            "PWF": (index % 17 == 0).astype(int),
            "OSF": (index % 19 == 0).astype(int),
            "RNF": (index % 5 == 4).astype(int),
        }
    )


# ---------------------------------------------------------------------------
# Conclusion 1 — feature list excludes leakage columns and the target
# ---------------------------------------------------------------------------


def test_conclusion_1_feature_list_excludes_leakage_and_target():
    """Conclusion: TWF/HDF/PWF/OSF/RNF and Machine failure are never features.

    Locked three ways so each line of defence is checked independently:
      * the module-level FEATURE_COLUMNS constant,
      * prepare_training_data() output on a synthetic df that CONTAINS the
        forbidden columns (they must still be dropped),
      * the programmatic check_no_leakage() guard raising ValueError.
    """
    # 1) the locked constant
    features = list(_FEATURE_COLUMNS)
    for column in FORBIDDEN_COLUMNS:
        assert column not in features, column

    # 2) the pipeline drops them even from adversarial input
    result = prepare_training_data(df=_fake_ai4i_dataframe())
    pipeline_features = result[4]
    for column in FORBIDDEN_COLUMNS:
        assert column not in pipeline_features, column

    # 3) the leakage guard refuses each forbidden column outright
    for column in FORBIDDEN_COLUMNS:
        with pytest.raises(ValueError):
            check_no_leakage([column])


# ---------------------------------------------------------------------------
# Conclusion 2 — the real models are not majority-class-only
# ---------------------------------------------------------------------------


def test_conclusion_2_models_are_not_majority_class_only(trained_real_models):
    """Conclusion: logreg / rf / mlp each have recall > 0 and predict BOTH classes.

    A majority-class-only predictor ("always 0") has recall == 0 and emits a
    single class; that is the dummy's behaviour and is a design choice for the
    accuracy-trap baseline only. Every real model must catch at least one
    failure (recall > 0) and must not collapse onto one prediction class
    (``len(np.unique(pred)) >= 2`` covers both "all 0" and "all 1" cases).
    Expected on the real data: logreg recall ~0.82, rf ~0.46, mlp ~0.88 —
    the threshold is deliberately the loose ``> 0`` floor.
    """
    X_test, y_test, models = trained_real_models

    for name, model in models.items():
        predictions = model.predict(X_test)

        recall = recall_score(y_test, predictions, zero_division=0)
        assert recall > 0.0, f"{name}: recall={recall} — majority-class-only!"

        # one assert covers "all 0" and "all 1" (they are the same failure mode)
        assert len(np.unique(predictions)) >= 2, (
            f"{name}: predictions collapsed to a single class"
        )


# ---------------------------------------------------------------------------
# Conclusion 3 — docs/results.json numbers match reports/paper.md
# ---------------------------------------------------------------------------


def _collect_numeric_strings(node, out: list[str]) -> list[str]:
    """Recursively collect number-bearing string values from a JSON subtree.

    Provenance markers (``source`` / ``*_source`` keys) are skipped: they cite
    section numbers, they are not measured values. Every remaining string that
    contains a digit is treated as a reported number and must appear verbatim
    in the paper text.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "source" or key.endswith("_source"):
                continue
            _collect_numeric_strings(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_numeric_strings(item, out)
    elif isinstance(node, str):
        if any(character.isdigit() for character in node):
            out.append(node)
    return out


def test_conclusion_3_results_json_numbers_match_paper():
    """Conclusion: every docs/results.json number matches reports/paper.md.

    Scope note: the ``iris`` subtree marks EVERY number with
    ``"source": "reports/paper.md §…"``; the ``ai4i`` subtree is sourced from
    ``outputs/ai4i_metrics.json`` (the AI4I paper section has not been written
    yet), so this test locks (a) each paper-sourced number appears verbatim in
    ``reports/paper.md``, (b) the four Iris metrics match the paper's table
    exactly (regex-extracted values, strict equality), and (c) every ai4i
    model metric block carries its ``outputs/ai4i_metrics.json`` source mark.
    """
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    paper_text = PAPER_PATH.read_text(encoding="utf-8")

    # (a) every numeric string of the paper-sourced iris subtree appears in the paper
    numbers = _collect_numeric_strings(results["iris"], [])
    assert len(numbers) >= 40, f"collector broke: only {len(numbers)} numbers found"
    for number in numbers:
        assert number in paper_text, (
            f"results.json value {number!r} not found verbatim in reports/paper.md"
        )

    # (b) strict: the 4 headline metrics must equal the paper table exactly
    metric_patterns = {
        "Accuracy": r"\|\s*Accuracy\s*\|\s*([\d.]+)\s*\|",
        "Precision (macro)": r"\|\s*Precision \(macro\)\s*\|\s*([\d.]+)\s*\|",
        "Recall (macro)": r"\|\s*Recall \(macro\)\s*\|\s*([\d.]+)\s*\|",
        "F1-score (macro)": r"\|\s*F1-score \(macro\)\s*\|\s*([\d.]+)\s*\|",
    }
    for metric_name, pattern in metric_patterns.items():
        match = re.search(pattern, paper_text)
        assert match, f"metric row {metric_name!r} missing from paper table"
        reported = results["iris"]["metrics"][metric_name]["value"]
        assert reported == match.group(1), (
            f"{metric_name}: results.json={reported} vs paper={match.group(1)}"
        )

    # (c) the ai4i numbers are likewise marked with their (data) source
    ai4i_metrics = results["ai4i"]["metrics"]
    assert "source" in ai4i_metrics
    assert "outputs/ai4i_metrics.json" in ai4i_metrics["source"]
    assert set(ai4i_metrics["models"]) == {"dummy", "logreg", "rf", "mlp"}


# ---------------------------------------------------------------------------
# Conclusion 4 — preprocessing params persisted and predictions reproduced
# ---------------------------------------------------------------------------


def test_conclusion_4_preprocessing_params_persisted_and_reproduce_predictions(
    prepared_with_state, tmp_path
):
    """Conclusion: the scaler (mean/std) is saved, and reloading it reproduces
    IDENTICAL predictions.

    Red-step note: ``model.py`` originally swallowed the scaler inside
    ``prepare_training_data``; the persistence API (``save_preprocessing``,
    ``load_preprocessing`` / ``rebuild_scaler``, plus
    ``prepare_training_data_with_state`` exposing the state) is the minimal
    implementation added to satisfy this conclusion.
    """
    X_train, X_test, y_train, y_test, feature_names, scaler1 = prepared_with_state

    # 1) persist to tmp_path so the repo is never touched
    prep_path = tmp_path / "prep.json"
    written = save_preprocessing(prep_path, scaler1, feature_names)
    assert isinstance(written, Path)
    assert written == prep_path
    assert prep_path.exists()

    payload = json.loads(prep_path.read_text(encoding="utf-8"))
    assert set(payload) == {"scaler_mean", "scaler_scale", "feature_names"}
    assert list(payload["feature_names"]) == list(feature_names)

    # 2) reload and rebuild an equivalent scaler
    params = load_preprocessing(prep_path)
    scaler2 = rebuild_scaler(params)

    assert np.array_equal(np.asarray(scaler2.mean_), np.asarray(scaler1.mean_))
    assert np.array_equal(np.asarray(scaler2.scale_), np.asarray(scaler1.scale_))

    # 3) rebuild the RAW features + the same split (identical random_state),
    #    then apply both scalers to the identical raw input: the persisted
    #    parameters must transform it into exactly the same matrices, so a
    #    freshly trained model produces IDENTICAL predictions
    df = load_ai4i_dataframe()
    one_hot = pd.get_dummies(df["Type"], prefix="Type", dtype=int)
    raw_features = pd.concat([df[_SENSOR_COLUMNS], one_hot], axis=1)
    target = df[_TARGET].astype(int)
    raw_train, raw_test = train_test_split(
        raw_features,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    X_train2 = scaler2.transform(raw_train)
    X_test2 = scaler2.transform(raw_test)
    assert np.array_equal(np.asarray(X_train2, dtype=float), np.asarray(X_train, dtype=float))
    assert np.array_equal(np.asarray(X_test2, dtype=float), np.asarray(X_test, dtype=float))

    model1 = train_model(build_models()["logreg"], X_train, y_train)
    model2 = train_model(build_models()["logreg"], X_train2, y_train)
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test2)

    assert np.array_equal(predictions1, predictions2)
    # sanity: the shared prediction is not itself majority-class-only
    assert len(np.unique(predictions1)) >= 2
