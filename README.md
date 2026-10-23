# Iris 資料集分析與類神經網路分類

以經典 Iris 資料集完成資料取得、探索分析、PCA 三維視覺化、MLP 類神經網路分類，並產出小論文。

## 專案結構

```
├── data/iris.csv                  # 保存的 Iris 資料集（150 x 5）
├── iris_analysis/                 # 分析套件
│   ├── data.py                    # 資料取得與存取
│   ├── eda.py                     # 探索分析、離群值、PCA
│   ├── model.py                   # 切分、MLP 訓練與評估
│   ├── visualize.py               # 所有圖表
│   ├── report.py                  # 小論文（Markdown）產生器
│   └── pipeline.py                # 端到端流程
├── scripts/run_analysis.py        # 命令列入口
├── outputs/figures/               # 產出的圖表
├── reports/paper.md               # 小論文
└── tests/                         # TDD 測試（17 個）
```

## 環境需求

- Python 3.10+
- 套件：`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`tabulate`

```bash
pip install -r requirements.txt
```

## 使用方式

執行完整分析（會重新產生資料、圖表與小論文）：

```bash
python scripts/run_analysis.py
```

自訂輸出路徑與實驗參數：

```bash
python scripts/run_analysis.py \
  --data data/iris.csv \
  --figures outputs/figures \
  --report reports/paper.md \
  --random-state 42 \
  --test-size 0.2
```

執行測試：

```bash
python -m pytest -q
```

## 開發方式（TDD）

本專案採測試導向開發：

1. 先於 `tests/` 撰寫資料、EDA、模型、圖表與端到端流程的測試。
2. 確認測試失敗（紅燈）。
3. 實作 `iris_analysis/` 對應模組直至測試通過（綠燈）。
4. 重構並執行完整流程驗證產出。

目前測試結果：**17 passed**。

## 實驗結果

| 指標 | 數值 |
| --- | --- |
| Accuracy | 0.9333 |
| Precision (macro) | 0.9333 |
| Recall (macro) | 0.9333 |
| F1-score (macro) | 0.9333 |

- 資料切分：120 筆訓練 / 30 筆測試（分層抽樣，random_state = 42）
- 模型：MLP（4 → 16 → 8 → 3），ReLU + Adam
- PCA 前三主成分累積解釋變異量：99.48%
- 詳細內容請見 [`reports/paper.md`](reports/paper.md)

## 圖表

| 圖表 | 檔案 |
| --- | --- |
| PCA 三維分布 | `outputs/figures/pca_3d.png` |
| 特徵分布 | `outputs/figures/feature_distributions.png` |
| 相關性熱圖 | `outputs/figures/correlation_heatmap.png` |
| 訓練損失曲線 | `outputs/figures/learning_curve.png` |
| 混淆矩陣 | `outputs/figures/confusion_matrix.png` |
| 各類別指標 | `outputs/figures/class_metrics.png` |

## 資料來源

- scikit-learn `load_iris`（原始資料：UCI Machine Learning Repository）
- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.

## AI4I 2020（進行中）

事件單：見 Issue #3（子工作單 #4–#10）。已有成果：

- `data/ai4i2020.csv`：UCI AI4I 2020 Predictive Maintenance Dataset（10,000 筆 × 14 欄位）
  - 來源：https://archive.ics.uci.edu/dataset/601
  - DOI：10.24432/C5HS5C；授權：CC BY 4.0
  - 取得方式：`python ai4i_analysis/data.py`
- `ai4i_analysis/data.py`：下載、保存、讀取與欄位驗證模組（TDD，`tests/test_ai4i_data.py` 8 個測試）
- `ai4i_analysis/model.py`：模型訓練模組（TDD，`tests/test_ai4i_model.py` 11 個測試）
  - `prepare_training_data()`：最小前處理 — 防洩漏特徵 5 感測器＋Type one-hot 共 8 欄、分層切分 0.2、StandardScaler 僅以訓練集 fit
  - `build_models()`：DummyClassifier 對照組 / LogisticRegression / RandomForest / MLP，皆可重現（random_state=42）
  - `train_model()`：MLP 因 sklearn 不支援 class_weight，以等效過採樣實作 {0:1, 1:28}
- `ai4i_analysis/evaluate.py`：評估與視覺化模組（TDD，`tests/test_ai4i_evaluate.py` 10 個測試）
  - 主指標 PR-AUC；`evaluate_model()` 回傳 PR-AUC／ROC-AUC／正類 precision/recall/F1／accuracy（對照用）／混淆矩陣／y_pred／y_score
  - `compare_models()`：4 模型指標彙整表，以 PR-AUC 為首欄降冪排序
  - 5 個繪圖函式輸出至 `outputs/ai4i_figures/`：PR/ROC 曲線 4 模型同圖、per-model 混淆矩陣×4、MLP loss 曲線、模型比較長條，共 8 張
  - Dummy accuracy 陷阱示範：96.6% acc 但 PR-AUC≈0.034（=正類比例）、ROC-AUC≈0.5
  - 全套件測試：46 = iris 17 + ai4i data 8 + model 11 + evaluate 10
