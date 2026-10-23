# -*- coding: utf-8 -*-
"""Export AI4I evaluation metrics into a persisted JSON file.

Calls the repo's own modules (ai4i_analysis.model / ai4i_analysis.evaluate)
— no reimplemented math, no invented numbers. The output
``outputs/ai4i_metrics.json`` is picked up by ``scripts/build_docs.py`` and
shown on the docs homepage.

Run:  python scripts/export_ai4i_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai4i_analysis.model import build_models, prepare_training_data, train_model
from ai4i_analysis.evaluate import evaluate_model

OUT = ROOT / "outputs" / "ai4i_metrics.json"


def main() -> None:
    X_train, X_test, y_test_arr, names = None, None, None, None
    X_train, X_test, y_train, y_test, feature_names = prepare_training_data()

    models = build_models()
    fitted = {name: train_model(model, X_train, y_train) for name, model in models.items()}

    per_model: dict = {}
    for name, model in fitted.items():
        r = evaluate_model(model, X_test, y_test) if False else evaluate_model(model, np.asarray(X_test), y_test)
        cm = np.asarray(r["confusion_matrix"]).tolist()
        entry = {
            "pr_auc": round(r["pr_auc"], 6),
            "roc_auc": round(r["roc_auc"], 6),
            "precision": round(r["precision"], 6),
            "recall": round(r["recall"], 6),
            "f1": round(r["f1"], 6),
            "accuracy": round(r["accuracy"], 6),
            "confusion_matrix": cm,
        }
        loss_curve = getattr(model, "loss_curve_", None)
        if loss_curve is not None:
            entry["loss_curve"] = {
                "length": len(loss_curve),
                "first": round(float(loss_curve[0]), 6),
                "last": round(float(loss_curve[-1]), 6),
                "series": [round(float(v), 6) for v in loss_curve],
            }
        per_model[name] = entry

    comparison = sorted(per_model.items(), key=lambda kv: -kv[1]["pr_auc"])

    payload = {
        "generated_by": "scripts/export_ai4i_metrics.py (呼叫 ai4i_analysis.model / evaluate 既有函式)",
        "feature_names": feature_names,
        "test_rows": int(len(X_test)),
        "test_positive_rate": round(float(np.mean(np.asarray(y_test))), 6),
        "primary_metric": "pr_auc (average precision)",
        "models": per_model,
        "ranking_by_pr_auc": [name for name, _ in comparison],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"written {OUT}")
    for name, m in per_model.items():
        print(f"  {name}: pr_auc={m['pr_auc']:.4f} roc_auc={m['roc_auc']:.4f} f1={m['f1']:.4f} acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
