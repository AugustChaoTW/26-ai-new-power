"""Evaluation and visualisation for the AI4I 2020 failure-prediction models.

Purpose
-------
Evaluate the four classifiers trained in :mod:`ai4i_analysis.model` on the
held-out test split, compare them in a single metrics table, and produce the
figure set of issue #9 into ``outputs/ai4i_figures/``.

Why PR-AUC (average precision) is the primary metric
----------------------------------------------------
The dataset is severely imbalanced: only 339 of 10,000 rows are machine
failures (positive-class prevalence ≈ 3.4%), so the test split contains
roughly 68 failures out of 2,000 rows.

The accuracy trap (issue #9, with the issue-text correction)
------------------------------------------------------------
``DummyClassifier(strategy="most_frequent")`` predicts only the majority
class "no failure". Its accuracy is therefore ≈ 1 - prevalence ≈ 96.6%, which
*looks* excellent, while it catches zero failures (recall = 0). Issue #9's
original text said "PR-AUC ≈ 0.5" for the dummy; the mathematically correct
baseline numbers under severe imbalance are:

    * ROC-AUC ≈ 0.5  (chance level — random ranking of positives/negatives)
    * PR-AUC ≈ positive-class prevalence ≈ 0.034
      (the "no-skill" baseline of the precision-recall curve equals the
      fraction of positives, NOT 0.5)

This module records that correction explicitly (see also
``tests/test_ai4i_evaluate.py::test_dummy_accuracy_trap``, which locks the
two reference points). Accuracy is still reported in
:func:`evaluate_model` / :func:`compare_models` — but for comparison /
reproduction of the trap only; model selection must use ``pr_auc``.

Contract (locked by tests/test_ai4i_evaluate.py)
------------------------------------------------
    - :func:`evaluate_model`(model, X_test, y_test) -> dict with keys
      {"pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy",
       "confusion_matrix", "y_pred", "y_score"}
    - :func:`compare_models`(models, X_test, y_test) -> pd.DataFrame whose
      FIRST column is "pr_auc", rows sorted by pr_auc descending, columns
      {"model", "pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"}
    - plots: plot_pr_curves / plot_roc_curves (one shared figure with all
      models), plot_confusion_matrix (per model), plot_learning_curve (MLP
      loss_curve_), plot_model_comparison (PR-AUC / F1 bars) — each saved
      with dpi=150 to ``out_dir`` (default ``outputs/ai4i_figures``).

Issue tracking: implements issue #9 (work item 6 of #3).
Tests: ``tests/test_ai4i_evaluate.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file as a script (`python ai4i_analysis/evaluate.py`).
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ai4i_analysis.data import load_ai4i_dataframe
from ai4i_analysis.model import build_models, prepare_training_data, train_model

DEFAULT_OUT_DIR = Path("outputs") / "ai4i_figures"

COLORS = ["#2E86AB", "#E67E22", "#27AE60", "#8E44AD"]

MODEL_ORDER = ["dummy", "logreg", "rf", "mlp"]

METRIC_COLUMNS = ["pr_auc", "model", "roc_auc", "precision", "recall", "f1", "accuracy"]

FIGURE_DPI = 150


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _obtain_y_score(model, X_test) -> np.ndarray:
    """Return the positive-class score vector ``y_score`` for ``model``.

    Prefers ``predict_proba`` (positive-class column, ``[:, 1]``), falls back
    to ``decision_function``. Raises ``ValueError`` when the estimator offers
    neither — a locked contract of issue #9.
    """
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_test))
        # robust to estimators predicting a subset of classes at inference
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_test), dtype=float)
    raise ValueError(
        "Model exposes neither predict_proba nor decision_function; "
        "cannot compute a y_score for PR / ROC evaluation."
    )


def _savefig(fig: plt.Figure, output_path: Path | str) -> Path:
    """Save a figure with the locked dpi and close it (no plt.show())."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _resolve_out_dir(out_dir: Path | str | None) -> Path:
    return Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR


# ---------------------------------------------------------------------------
# 1. Metric evaluation
# ---------------------------------------------------------------------------


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute the issue-#9 metric bundle for one fitted model.

    Returns a dict with exactly:
        pr_auc          : average_precision_score (PRIMARY metric; under the
                          3.4% imbalance the no-skill baseline ≈ prevalence)
        roc_auc         : roc_auc_score (dummy/no-skill ≈ 0.5 — the corrected
                          statement of the issue-#9 accuracy trap)
        precision/recall/f1 : positive class (pos_label=1, zero_division=0)
        accuracy        : accuracy_score — for the trap comparison only
        confusion_matrix: sklearn confusion_matrix, a (2, 2) numpy array
        y_pred / y_score: the underlying prediction / score vectors
    """
    y_pred = np.asarray(model.predict(X_test))
    y_score = _obtain_y_score(model, X_test)
    y_test_array = np.asarray(y_test)

    return {
        "pr_auc": float(average_precision_score(y_test_array, y_score)),
        "roc_auc": float(roc_auc_score(y_test_array, y_score)),
        "precision": float(
            precision_score(y_test_array, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test_array, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y_test_array, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_test_array, y_pred)),
        "confusion_matrix": confusion_matrix(y_test_array, y_pred),
        "y_pred": y_pred,
        "y_score": y_score,
    }


def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every model and return the issue-#9 comparison table.

    One row per model with columns
    {"model", "pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"},
    sorted by ``pr_auc`` (the primary metric) descending, with ``pr_auc``
    moved to the first column and the index reset.
    """
    metric_keys = ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]
    rows = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test)
        rows.append({"model": name, **{key: result[key] for key in metric_keys}})

    df = pd.DataFrame(rows)
    df = df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    # pr_auc first per the locked contract
    return df[METRIC_COLUMNS]


# ---------------------------------------------------------------------------
# 2. Figures (all saved with dpi=150, never shown, always closed)
# ---------------------------------------------------------------------------


def plot_pr_curves(models: dict, X_test, y_test, out_dir=DEFAULT_OUT_DIR) -> list[Path]:
    """One shared figure with the PR curve of every model.

    Each line is labelled with its average precision (AP) value; the
    prevalence baseline is drawn as a horizontal reference. Returns
    ``[path_to_ai4i_pr_curves.png]`` — a single-figure promise, hence a
    one-element list per the locked contract.
    """
    out_dir = _resolve_out_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 6))

    for index, (name, model) in enumerate(models.items()):
        y_score = _obtain_y_score(model, X_test)
        precision, recall, _ = precision_recall_curve(np.asarray(y_test), y_score)
        ap = average_precision_score(np.asarray(y_test), y_score)
        ax.plot(
            recall,
            precision,
            color=COLORS[index % len(COLORS)],
            linewidth=2,
            label=f"{name} (AP={ap:.3f})",
        )

    prevalence = float(np.asarray(y_test).mean())
    ax.axhline(
        prevalence,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"no-skill baseline (prevalence={prevalence:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — AI4I 2020 (primary metric: PR-AUC)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    path = _savefig(fig, out_dir / "ai4i_pr_curves.png")
    return [path]


def plot_roc_curves(models: dict, X_test, y_test, out_dir=DEFAULT_OUT_DIR) -> list[Path]:
    """One shared figure with the ROC curve of every model.

    A constant ``y_score`` (the dummy predicts all-equal probabilities)
    collapses to the chance diagonal — expected and kept on the figure.
    Returns ``[path_to_ai4i_roc_curves.png]``.
    """
    out_dir = _resolve_out_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 6))

    for index, (name, model) in enumerate(models.items()):
        y_score = _obtain_y_score(model, X_test)
        fpr, tpr, _ = roc_curve(np.asarray(y_test), y_score)
        auc = roc_auc_score(np.asarray(y_test), y_score)
        ax.plot(
            fpr,
            tpr,
            color=COLORS[index % len(COLORS)],
            linewidth=2,
            label=f"{name} (AUC={auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="chance (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — AI4I 2020 (secondary metric)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = _savefig(fig, out_dir / "ai4i_roc_curves.png")
    return [path]


def plot_confusion_matrix(model, X_test, y_test, out_dir=DEFAULT_OUT_DIR, model_name="mlp") -> Path:
    """Confusion-matrix heatmap for one model (one file per model).

    Saves ``confusion_matrix_{model_name}.png`` in ``out_dir``. Cell-type
    annotations (TN / FP / FN / TP for array indices [0, 0] / [0, 1] /
    [1, 0] / [1, 1]) are drawn inside each cell along with the raw counts.
    """
    out_dir = _resolve_out_dir(out_dir)
    cm = np.asarray(evaluate_model(model, X_test, y_test)["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5.5, 4.75))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for row in range(2):
        for column in range(2):
            value = cm[row, column]
            ax.text(
                column,
                row - 0.08,
                cell_labels[row][column],
                ha="center",
                va="center",
                fontsize=9,
                color="black" if value / max(cm.max(), 1) > 0.55 else "#555555",
            )
            ax.text(
                column,
                row + 0.15,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="white" if value / max(cm.max(), 1) > 0.55 else "black",
            )

    ax.set_xticks([0, 1], ["predicted 0", "predicted 1"])
    ax.set_yticks([0, 1], ["actual 0", "actual 1"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    return _savefig(fig, out_dir / f"confusion_matrix_{model_name}.png")


def plot_learning_curve(model, out_dir=DEFAULT_OUT_DIR, model_name="mlp") -> Path:
    """Plot the MLP training loss (``loss_curve_``) per iteration.

    Raises ``ValueError`` when the estimator has no ``loss_curve_`` — the
    locked behaviour that must trigger for dummy / logreg / rf.
    """
    out_dir = _resolve_out_dir(out_dir)
    loss_curve = getattr(model, "loss_curve_", None)
    if loss_curve is None:
        raise ValueError(
            f"Model '{model_name}' has no loss_curve_; a learning curve is "
            "only available for MLPClassifier."
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        range(1, len(loss_curve) + 1),
        loss_curve,
        color=COLORS[0],
        linewidth=2,
    )
    ax.set_title(f"MLP Training Loss — {model_name} (loss_curve_)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, out_dir / f"learning_curve_{model_name}.png")


def plot_model_comparison(df: pd.DataFrame, out_dir=DEFAULT_OUT_DIR) -> Path:
    """Side-by-side PR-AUC / F1 bars from a :func:`compare_models` DataFrame.

    Saves ``model_comparison.png`` with one bar pair per model, keeping the
    DataFrame's pr_auc-descending row order.
    """
    out_dir = _resolve_out_dir(out_dir)
    names = df["model"].tolist()
    positions = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        positions - width / 2,
        df["pr_auc"].to_numpy(dtype=float),
        width,
        color=COLORS[0],
        label="PR-AUC (primary)",
    )
    ax.bar(
        positions + width / 2,
        df["f1"].to_numpy(dtype=float),
        width,
        color=COLORS[1],
        label="F1 (positive class)",
    )

    # annotate the primary metric on top of each bar pair
    for position, value in zip(positions, df["pr_auc"].to_numpy(dtype=float)):
        ax.text(position - width / 2, value + 0.008, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8, color=COLORS[0])

    ax.set_xticks(positions, names)
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Model Comparison — PR-AUC vs F1 (sorted by PR-AUC)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return _savefig(fig, out_dir / "model_comparison.png")


# ---------------------------------------------------------------------------
# 3. Manual end-to-end smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - manual smoke test
    X_train, X_test, y_train, y_test, feature_names = prepare_training_data()
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Test shape: {X_test.shape}, Test positives: {int(np.asarray(y_test).sum())}")

    fitted = {name: train_model(model, X_train, y_train) for name, model in build_models().items()}

    results = {}
    for name, model in fitted.items():
        results[name] = evaluate_model(model, X_test, y_test)
        r = results[name]
        print(
            f"\n{name}: pr_auc={r['pr_auc']:.4f} roc_auc={r['roc_auc']:.4f} "
            f"precision={r['precision']:.4f} recall={r['recall']:.4f} "
            f"f1={r['f1']:.4f} accuracy={r['accuracy']:.4f}\n"
            f"confusion_matrix=\n{np.asarray(r['confusion_matrix'])}"
        )

    print("\ncompare_models():")
    comparison = compare_models(fitted, X_test, y_test)
    print(comparison.to_string())

    out_dir = DEFAULT_OUT_DIR
    pr_paths = plot_pr_curves(fitted, X_test, y_test, out_dir)
    roc_paths = plot_roc_curves(fitted, X_test, y_test, out_dir)
    cm_paths = [plot_confusion_matrix(model, X_test, y_test, out_dir, name) for name, model in fitted.items()]
    curve_path = plot_learning_curve(fitted["mlp"], out_dir, model_name="mlp")
    comparison_path = plot_model_comparison(comparison, out_dir)
    written = pr_paths + roc_paths + cm_paths + [curve_path, comparison_path]
    print(f"\nFigures written to {out_dir}/:")
    for path in written:
        print(f"  - {path}")

    # accuracy-trap verdict: the dummy's numbers vs the corrected baselines
    dummy = results["dummy"]
    prevalence = float(np.asarray(y_test).mean())
    print(
        f"\n[accuracy-trap] dummy accuracy={dummy['accuracy']:.4f} (looks great) "
        f"vs recall={dummy['recall']:.4f} (catches nothing)\n"
        f"  PR-AUC={dummy['pr_auc']:.4f} ≈ prevalence={prevalence:.4f} "
        f"(|diff|={abs(dummy['pr_auc'] - prevalence):.4f})\n"
        f"  ROC-AUC={dummy['roc_auc']:.4f} ≈ 0.5 "
        f"(|diff|={abs(dummy['roc_auc'] - 0.5):.4f})\n"
        "(correction to issue #9: the dummy baseline is ROC-AUC≈0.5 and "
        "PR-AUC≈positive-class prevalence, not PR-AUC≈0.5)"
    )

    from sklearn.neural_network import MLPClassifier

    mlp = fitted["mlp"]
    if isinstance(mlp, MLPClassifier) and mlp.loss_curve_ is not None:
        curve = mlp.loss_curve_
        print(
            f"\n[mlp] loss_curve_ length={len(curve)}, "
            f"first={curve[0]:.4f}, last={curve[-1]:.4f}"
        )
