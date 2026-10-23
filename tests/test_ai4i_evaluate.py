"""TDD tests for ai4i_analysis.evaluate — written before implementation (issue #9 / work item 6 of #3).

The module (and every symbol imported below) does not exist yet, so this file
must fail at collection time with ``ModuleNotFoundError``. That is the
intended red step of the TDD plan.

Locked contract (from issue #9, PR-AUC as the primary metric):

    - evaluate_model(model, X_test, y_test) -> dict with keys
      {"pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy",
       "confusion_matrix", "y_pred", "y_score"}
      * pr_auc = average_precision_score(y_test, y_score)  # primary metric
      * precision / recall / f1 on the positive class (pos label=1,
        zero_division=0); accuracy = accuracy_score
      * confusion_matrix has shape (2, 2)
      * y_score comes from predict_proba (positive-class column)
    - compare_models(models, X_test, y_test) -> pd.DataFrame whose FIRST
      column is "pr_auc" and rows are sorted by pr_auc descending;
      columns {"model", "pr_auc", "roc_auc", "precision", "recall", "f1",
      "accuracy"}
    - plot_pr_curves(models, X_test, y_test, out_dir)       -> list[Path]
      (file "ai4i_pr_curves.png", all 4 models in one figure)
    - plot_roc_curves(models, X_test, y_test, out_dir)      -> list[Path]
      (file "ai4i_roc_curves.png")
    - plot_confusion_matrix(model, X_test, y_test, out_dir, model_name) -> Path
      (file f"confusion_matrix_{model_name}.png", one per model)
    - plot_learning_curve(model, out_dir, model_name="mlp") -> Path
      (file f"learning_curve_{model_name}.png", from MLP loss_curve_;
      a model without loss_curve_ raises ValueError)
    - plot_model_comparison(df, out_dir)                    -> Path
      (file "model_comparison.png", PR-AUC / F1 bar comparison)

Fixture strategy mirrors tests/test_ai4i_model.py: a module-scope ``trained``
fixture prepares the split and trains all four models exactly once
(rf ~6s, mlp ~30-50s) so evaluation and every plot reuse the same fitted
estimators, keeping the whole module well under 200s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai4i_analysis.evaluate import (
    compare_models,
    evaluate_model,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
)
from ai4i_analysis.model import build_models, prepare_training_data, train_model

REQUIRED_EVAL_KEYS = {
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "confusion_matrix",
    "y_pred",
    "y_score",
}

METRIC_KEYS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]

COMPARE_COLUMNS = {"model", "pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"}

EXPECTED_OUT_DIRECTORY = Path("outputs") / "ai4i_figures"


# ---------------------------------------------------------------------------
# Module-scope fixtures: prepare + train all 4 models exactly once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    """Split + the four fitted models, computed once for the module.

    Returns ``(X_test, y_test, fitted_models)`` where ``fitted_models`` maps
    {"dummy", "logreg", "rf", "mlp"} to fitted estimators.
    """
    X_train, X_test, y_train, y_test, _feature_names = prepare_training_data()
    fitted_models = {name: train_model(model, X_train, y_train) for name, model in build_models().items()}
    return X_test, y_test, fitted_models


@pytest.fixture(scope="module")
def tmp_fig_dir(tmp_path_factory):
    """Module-scope directory for every plot_* output (keeps repo clean)."""
    return tmp_path_factory.mktemp("ai4i_figures")


@pytest.fixture(scope="module")
def comparison(trained):
    """compare_models() DataFrame computed once on the fitted models."""
    X_test, y_test, models = trained
    return compare_models(models, X_test, y_test)


# ---------------------------------------------------------------------------
# 1. evaluate_model contract
# ---------------------------------------------------------------------------


def test_evaluate_model_returns_required_keys(trained):
    X_test, y_test, models = trained

    for name in ["dummy", "logreg"]:
        result = evaluate_model(models[name], X_test, y_test)

        # the 9 keys are locked for the issue-#9 acceptance check
        assert set(result) == REQUIRED_EVAL_KEYS, name
        assert isinstance(result, dict)

        # array outputs have test-set length / shape
        assert len(result["y_pred"]) == len(y_test)
        assert len(result["y_score"]) == len(y_test)
        cm = np.asarray(result["confusion_matrix"])
        assert cm.shape == (2, 2)


def test_metrics_in_unit_range(trained):
    X_test, y_test, models = trained

    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test)
        for key in METRIC_KEYS:
            value = float(result[key])
            assert 0.0 <= value <= 1.0, f"{name}.{key}={value} outside [0, 1]"


def test_dummy_accuracy_trap(trained):
    """The accuracy trap: near-perfect accuracy, PR-AUC at chance level.

    Two reference points are locked so the test stays reproducible without
    retraining anything:
      * dummy pr_auc is close to the positive-class prevalence of y_test
        (|pr_auc - prevalence| <= 0.02),
      * dummy roc_auc is close to chance (|roc_auc - 0.5| <= 0.05).
    """
    X_test, y_test, models = trained

    result = evaluate_model(models["dummy"], X_test, y_test)
    prevalence = float(np.asarray(y_test).mean())

    assert result["accuracy"] >= 0.95  # the trap: looks excellent …
    assert abs(float(result["pr_auc"]) - prevalence) <= 0.02  # … but PR-AUC is baseline
    assert abs(float(result["roc_auc"]) - 0.5) <= 0.05  # ...and ROC-AUC is chance

    # kept on purpose for the issue-#9 write-up: show the concrete numbers
    print(
        f"\n[accuracy-trap] dummy accuracy={result['accuracy']:.4f} "
        f"vs prevalence={prevalence:.4f}; pr_auc={result['pr_auc']:.4f}; "
        f"roc_auc={result['roc_auc']:.4f}"
    )


def test_confusion_matrix_shape_and_sums(trained):
    X_test, y_test, models = trained

    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test)
        cm = np.asarray(result["confusion_matrix"])
        assert cm.shape == (2, 2), name
        assert int(cm.sum()) == len(y_test) == 2000, name

    # dummy (most_frequent) predicts zero positives: no predicted-positive
    # cells — (0,1) and (1,1) must be 0
    dummy_cm = np.asarray(evaluate_model(models["dummy"], X_test, y_test)["confusion_matrix"])
    assert dummy_cm[0, 1] == 0  # FP = 0
    assert dummy_cm[1, 1] == 0  # TP = 0
    assert dummy_cm[1, 0] == int(np.asarray(y_test).sum())  # all 68 misses


# ---------------------------------------------------------------------------
# 2. compare_models contract
# ---------------------------------------------------------------------------


def test_compare_models_sorted_by_pr_auc_first_column(comparison):
    assert isinstance(comparison, pd.DataFrame)
    assert list(comparison.columns)[0] == "pr_auc"  # primary metric first
    assert set(comparison.columns) == COMPARE_COLUMNS

    # all four model names present in the "model" column
    assert set(comparison["model"]) == {"dummy", "logreg", "rf", "mlp"}

    # sorted by pr_auc descending
    pr_auc_values = comparison["pr_auc"].to_numpy()
    assert np.all(np.diff(pr_auc_values) <= 0)


def test_compare_models_mlp_vs_dummy_positioning(trained, comparison):
    """A real model must clearly beat the accuracy-trap baseline on PR-AUC."""
    X_test, y_test, models = trained

    dummy_pr_auc = float(comparison.loc[comparison["model"] == "dummy", "pr_auc"].iloc[0])
    best_pr_auc = float(comparison["pr_auc"].max())

    # lenient floor first for diagnosis, then the locked margin
    assert best_pr_auc > dummy_pr_auc + 0.1
    # rf / logreg / mlp genuinely learn the signal -> large gap expected
    assert best_pr_auc > 0.3


# ---------------------------------------------------------------------------
# 3. evaluate_model edge behaviour
# ---------------------------------------------------------------------------


def test_evaluate_model_positives_count_broad(trained):
    """Dummy predicts no positives -> zero_division=0 yields 0.0 metrics."""
    X_test, y_test, models = trained

    result = evaluate_model(models["dummy"], X_test, y_test)
    assert result["precision"] == 0.0  # zero_division=0 convention
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


# ---------------------------------------------------------------------------
# 4. Visualisation contract
# ---------------------------------------------------------------------------


def test_curves_report_consistent_points(trained, tmp_fig_dir):
    """The curve functions run on evaluate_model's y_score and return Paths.

    The score vector of evaluate_model is the same one a precision_recall /
    ROC curve is built from, so axis-length consistency is an internal
    guarantee of sklearn; here we lock existence, file names, and the
    documented return types (list[Path] / Path).
    """
    X_test, y_test, models = trained

    pr_paths = plot_pr_curves(models, X_test, y_test, tmp_fig_dir)
    assert isinstance(pr_paths, list)
    assert len(pr_paths) >= 1
    assert all(isinstance(p, Path) for p in pr_paths)
    assert pr_paths[0].name == "ai4i_pr_curves.png"
    assert pr_paths[0].exists()

    roc_paths = plot_roc_curves(models, X_test, y_test, tmp_fig_dir)
    assert isinstance(roc_paths, list)
    assert len(roc_paths) >= 1
    assert all(isinstance(p, Path) for p in roc_paths)
    assert roc_paths[0].name == "ai4i_roc_curves.png"
    assert roc_paths[0].exists()


def test_figures_outputs(trained, tmp_fig_dir):
    """All five plot_* functions together produce the full 8-figure set."""
    X_test, y_test, models = trained

    produced: list[Path] = []
    produced += plot_pr_curves(models, X_test, y_test, tmp_fig_dir)           # 1
    produced += plot_roc_curves(models, X_test, y_test, tmp_fig_dir)          # 2
    for name, model in models.items():                                        # 3-6
        cm_path = plot_confusion_matrix(model, X_test, y_test, tmp_fig_dir, name)
        assert isinstance(cm_path, Path), name
        assert cm_path.name == f"confusion_matrix_{name}.png"
        produced.append(cm_path)
    mlp_curve = plot_learning_curve(models["mlp"], tmp_fig_dir, model_name="mlp")  # 7
    assert isinstance(mlp_curve, Path)
    assert mlp_curve.name == "learning_curve_mlp.png"
    produced.append(mlp_curve)
    comparison_df = compare_models(models, X_test, y_test)
    comparison_png = plot_model_comparison(comparison_df, tmp_fig_dir)        # 8
    assert isinstance(comparison_png, Path)
    assert comparison_png.name == "model_comparison.png"
    produced.append(comparison_png)

    # every announced file exists …
    pngs = sorted(tmp_fig_dir.glob("*.png"))
    expected_names = {
        "ai4i_pr_curves.png",
        "ai4i_roc_curves.png",
        "model_comparison.png",
        "learning_curve_mlp.png",
    } | {f"confusion_matrix_{name}.png" for name in models}
    assert {p.name for p in pngs} == expected_names

    # … the directory holds the full set of at least 8 .png figures …
    assert len(pngs) >= 8, [p.name for p in pngs]

    # … and none of them is an empty/broken file
    for path in produced:
        assert path.stat().st_size > 1024, f"{path.name} too small"
    # (mirrors the intended real-project location outputs/ai4i_figures)
    assert expected_names == {
        "ai4i_pr_curves.png",
        "ai4i_roc_curves.png",
        "confusion_matrix_dummy.png",
        "confusion_matrix_logreg.png",
        "confusion_matrix_rf.png",
        "confusion_matrix_mlp.png",
        "learning_curve_mlp.png",
        "model_comparison.png",
    }


def test_learning_curve_from_mlp_loss(trained, tmp_fig_dir):
    """Learning curve comes from the MLP's loss_curve_; others raise."""
    X_test, y_test, models = trained

    mlp = models["mlp"]
    assert mlp.loss_curve_ is not None and len(mlp.loss_curve_) >= 10

    path = plot_learning_curve(mlp, tmp_fig_dir, model_name="mlp")
    assert isinstance(path, Path)
    assert path.name == "learning_curve_mlp.png"
    assert path.exists()
    assert path.stat().st_size > 1024

    # a model without loss_curve_ must raise ValueError (locked behaviour)
    for name in ["dummy", "logreg", "rf"]:
        non_mlp = models[name]
        assert not hasattr(non_mlp, "loss_curve_"), name
        with pytest.raises(ValueError):
            plot_learning_curve(non_mlp, tmp_fig_dir, model_name=name)
