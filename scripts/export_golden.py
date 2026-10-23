# -*- coding: utf-8 -*-
"""Export golden predictions to tests/golden.json.

流程：
  1. 以既有訓練流程重訓 Iris（iris_analysis.model）與 AI4I（ai4i_analysis.model）
     MLPClassifier（random_state=42，與 docs/trained_params.js 相同設定）。
  2. 先做「參數同步檢查」：重訓出的 scaler/coefs/intercepts 必須與
     trained_params.js 逐位一致（atol 1e-9）；不一致則明確印出差異——
     這對應三種失因中的「係數貼錯或貼漏」，需先修參數檔再產 golden。
  3. 對每個模型固定 3 筆輸入，以 sklearn predict_proba / predict 取得
     類別與各類機率（全精度），寫入 tests/golden.json。

不再訓練新模型、不改演算法；golden 基準 = 部署參數來源模型本人的輸出。

Run:  PYTHONPATH=. python scripts/export_golden.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd  # noqa: F401  (AI4I one-hot 與訓練路徑同構)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ai4i_analysis.data import load_ai4i_dataframe
from ai4i_analysis.model import (
    CLASS_WEIGHTS,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS as AI4I_FEATURES,
    RANDOM_STATE,
    SENSOR_COLUMNS,
    TARGET,
    apply_class_weights,
    build_models,
    check_no_leakage,
)
from iris_analysis.data import CLASS_NAMES, FEATURES, load_iris_dataframe
from iris_analysis.model import build_model, split_data, train_model

PARAMS_JS = ROOT / "docs" / "trained_params.js"
GOLDEN_PATH = ROOT / "tests" / "golden.json"
SYNC_ATOL = 1e-9

# ---------------------------------------------------------------- 3 筆固定輸入
IRIS_CASES = [
    # 與訓練資料同一 Alex 分佈的代表性樣本（setosa / 邊界 / virginica）
    [5.1, 3.5, 1.4, 0.2],
    [5.9, 2.7, 4.2, 1.3],
    [6.8, 3.0, 5.5, 2.1],
]

AI4I_RAW_CASES = [  # 取自真實資料列（固定列號，不改資料）
    {"row": 0},        # 一般運轉
    {"row": 5000},     # 稍後期的正常列
    {"row": -1},       # 最後一列（涵蓋型別差異）
]


# ---------------------------------------------------------------- 參數檔解析
def grab_params(section: str, name: str):
    """從 trained_params.js 區塊中擷取 `name:` 後的 JSON 陣列（略過註解）。"""
    m = re.search(name + r"\s*:\s*", section)
    part = section[m.end():]
    start = part.index("[")
    depth = 0
    for k in range(start, len(part)):
        if part[k] == "[":
            depth += 1
        elif part[k] == "]":
            depth -= 1
            if depth == 0:
                return json.loads(re.sub(r"//[^\n]*", "", part[start:k + 1]))
    raise ValueError(name)


def max_diff(a, b) -> float:
    return max(abs(x - y) for x, y in zip(np.asarray(a).ravel(), np.asarray(b).ravel()))


def check_param_sync() -> dict:
    """重訓參數 vs trained_params.js 的逐位檢查；不一致時 raise。"""
    src = PARAMS_JS.read_text(encoding="utf-8")
    iris_block = src[src.index("var IRIS_PARAMS"):]
    ai4i_block = src[src.index("var AI4I_PARAMS"):]

    # Iris
    df_iris = load_iris_dataframe()
    Xi_tr, Xi_te, yi_tr, yi_te = split_data(df_iris, test_size=0.2, random_state=42)
    iris_model, iris_scaler = train_model(build_model(random_state=42), Xi_tr, yi_tr)

    results = {"iris": {}, "ai4i": {}}
    for name in ("scalerMean", "scalerScale"):
        results["iris"][name] = max_diff(grab_params(iris_block, name), iris_scaler.mean_ if name == "scalerMean" else iris_scaler.scale_)
    results["iris"]["coefs"] = max(
        max_diff(a, b) for a, b in zip(iris_model.coefs_, grab_params(iris_block, "coefs"))
    )
    results["iris"]["intercepts"] = max(
        max_diff(a, b) for a, b in zip(iris_model.intercepts_, grab_params(iris_block, "intercepts"))
    )

    # AI4I（重現 prepare_training_data 的 split + scaler 步驟以取得 scaler 本體）
    df = load_ai4i_dataframe()
    one_hot = pd.get_dummies(df["Type"], prefix="Type", dtype=int)
    X = pd.concat([df[SENSOR_COLUMNS], one_hot], axis=1)[FEATURE_COLUMNS]
    check_no_leakage(list(X.columns))
    y = df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    mlp = build_models()["mlp"]
    Xw, yw = apply_class_weights(X_tr_scaled, np.asarray(y_tr), CLASS_WEIGHTS)
    mlp.fit(np.asarray(Xw, dtype=float), yw)

    results["ai4i"]["scalerMean"] = max_diff(grab_params(ai4i_block, "scalerMean"), scaler.mean_)
    results["ai4i"]["scalerScale"] = max_diff(grab_params(ai4i_block, "scalerScale"), scaler.scale_)
    results["ai4i"]["coefs"] = max(
        max_diff(a, b) for a, b in zip(mlp.coefs_, grab_params(ai4i_block, "coefs"))
    )
    results["ai4i"]["intercepts"] = max(
        max_diff(a, b) for a, b in zip(mlp.intercepts_, grab_params(ai4i_block, "intercepts"))
    )

    worst = max(v for d in results.values() for v in d.values())
    if worst > SYNC_ATOL:
        raise RuntimeError(
            f"參數同步檢查失敗（max diff {worst}）：重訓參數與 trained_params.js 不一致，"
            "可能係數貼錯/貼漏或 sklearn 版本漂移——請先修 docs/trained_params.js。"
        )
    return results, iris_model, iris_scaler, mlp, scaler, list(AI4I_FEATURES)


# ---------------------------------------------------------------- golden 產出
def export() -> None:
    sync, iris_model, iris_scaler, mlp, ai4i_scaler, ai4i_features = check_param_sync()
    print("參數同步檢查 OK（max diff =",
          max(v for d in sync.values() for v in d.values()), "）")

    iris_cases = []
    for x in IRIS_CASES:
        probs = iris_model.predict_proba(iris_scaler.transform([x]))[0].tolist()
        idx = int(np.argmax(probs))
        iris_cases.append({
            "input": x,
            "class_index": idx,
            "class_name": CLASS_NAMES[idx],
            "probs": probs,
        })
        print("iris ", x, "->", iris_cases[-1]["class_name"], probs)

    df = load_ai4i_dataframe()
    ai4i_cases = []
    for spec in AI4I_RAW_CASES:
        row = df.iloc[spec["row"]]
        raw = {
            "Type": row["Type"],
            "Air temperature [K]": float(row[SENSOR_COLUMNS[0]]),
            "Process temperature [K]": float(row[SENSOR_COLUMNS[1]]),
            "Rotational speed [rpm]": float(row[SENSOR_COLUMNS[2]]),
            "Torque [Nm]": float(row[SENSOR_COLUMNS[3]]),
            "Tool wear [min]": float(row[SENSOR_COLUMNS[4]]),
        }
        onehot = pd.get_dummies(pd.Series([raw["Type"]]), prefix="Type", dtype=int)
        flat = [raw[SENSOR_COLUMNS[0]], raw[SENSOR_COLUMNS[1]], raw[SENSOR_COLUMNS[2]],
                raw[SENSOR_COLUMNS[3]], raw[SENSOR_COLUMNS[4]]]
        flat += [int(onehot.get(c, pd.Series([0])).iloc[0])
                 for c in ("Type_H", "Type_L", "Type_M")]
        p = float(mlp.predict_proba(ai4i_scaler.transform([flat]))[0][1])
        ai4i_cases.append({
            "raw": raw,
            "input": flat,
            "true_failure": int(row[TARGET]),
            "p_failure": p,
        })
        print("ai4i", raw["Type"], flat, "-> p_failure =", p,
              "(true =", ai4i_cases[-1]["true_failure"], ")")

    golden = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "generator": "scripts/export_golden.py（sklearn predict_proba，random_state=42）",
        "models": {
            "iris": "iris_analysis.model: MLPClassifier(16,8) relu adam alpha=1e-4 lr=0.01 seed=42",
            "ai4i": "ai4i_analysis.model: MLPClassifier(16,8) logistic adam seed=42, 28x 類別權重",
        },
        "tolerance": 1e-6,
        "param_sync_max_diff": {k: v for k, v in sync.items()},
        "iris": {
            "feature_names": FEATURES,
            "classes": CLASS_NAMES,
            "cases": iris_cases,
        },
        "ai4i": {
            "feature_names": ai4i_features,
            "cases": ai4i_cases,
        },
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print("written:", GOLDEN_PATH)


if __name__ == "__main__":
    export()
