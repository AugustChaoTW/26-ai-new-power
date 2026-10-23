# Delphi Panel：AI4I 不平衡分析建議（關聯 issue #5）

- **主題**：AI4I 2020 預測性維護資料集之 EDA 基本探索與不平衡分析
- **單位**：AugustChaoTW/26-ai-new-power#5（[AI4I] 2. EDA 基本探索與不平衡分析）
- **輪次**：Delphi 小組 **第 1 輪 → 第 2 輪 → 第 3 輪**（第三輪為最終輪）
- **主席**：林建國 教授（AI 實驗室指導教授／專案 PI，負責最終匯整）

---

## 0. 專家背景（5 位成員 + 主席）

### 主席（Chair）
**林建國 教授** — AI 實驗室指導教授／專案 PI。負責把會議控管、壓合共識、最終匯整。

### E1 — Dr. 陳映雪（統計學家／不平衡學）
- **背景**：統計學系副教授，專精 imbalanced binary classification 的統計推論（ROC-AUC、PR-AUC、prevalence 作為 no-skill baseline、bootstrap 信賴區間）。
- **與本單相關**：最能評估「3.4% 不平衡」的量化與統計意義。

### E2 — Dr. 張兆明（領域專家／維護顧問）
- **背景**：工業設備故障預測領域顧問，具 10 年以上機器維護失敗預測經驗；熟悉 failure 長尾、severity 與 threshold/cost 敏感性。
- **與本單相關**：評估 EDA imbalance 分析在領域語境下的合理性與決策影響。

### E3 — Dr. 蔡明軒（ML 工程師／AI4I 技術負責）
- **背景**：資料科學／機器學習工程師，AI4I 專案技術負責；熟悉 `ai4i_analysis`（data/model/evaluate）、sklearn MLP、StandardScaler、one-hot 特徵順序。
- **與本單相關**：評估 issue #5 EDA 實作（`ai4i_analysis/eda.py`）與工程面可行性。

### E4 — Prof. 王慧玲（教學者／審稿）
- **背景**：資料科學教學與技術審稿；熟悉 TDD 流程與測試合約（acceptance criteria）。
- **與本單相關**：審核 issue #5 的 TDD 要求（先紅燈後綠燈）與驗收條件（imbalance ratio 339/10000 ≈ 3.4%、3 張圖表 > 1KB、balance 總和 10000）。

### E5 — Dr. 趙建軍（方法學家／imbalanced learning）
- **背景**：機器學習方法學研究者；專精 imbalanced learning（SMOTE、加權 loss、threshold tuning、evaluation under imbalance）。
- **與本單相關**：評估 EDA imbalance 資訊如何餵應後續建模（resampling、threshold）與評估指標（PR-AUC primary）。

---

## 1) 第 1 輪討論（獨立初評）

**主席開場**：issue #5 工作內容 — EDA：`dataset_overview`、`describe_features`、`class_balance`、`correlation_matrix`、圖表 `plot_class_imbalance`／`plot_feature_distributions`／`plot_correlation_heatmap`；驗收：不平衡比例具體呈現（339/10,000 ≈ 3.4%）、3 張圖表、對應測試全數通過。

### E1（統計學家）初評
- **結論**：3.4% 的呈現（339/10000）屬正確且必要量化；建議在 EDA `class_balance` 中補 **prevalence（≈0.034）** 與 **no-skill baseline**（ROC-AUC ≈ 0.5、PR-AUC ≈ 0.034）。
- **建議**：`plot_class_imbalance` 應畫 no-skill 參考線，讓「模型表現 > baseline」可量化。

### E2（領域專家）初評
- **結論**：3.4% 長尾 imbalance 合理；建議 EDA 納入 **failure severity／type 分布**（Type_H/L/M）與 threshold/cost 敏感性視角。
- **建議**：domain 層面支持 imbalance 分析的價值；threshold decision 留待建模後。

### E3（ML 工程師）初評
- **結論**：工程上 `ai4i_analysis/eda.py` 尚未落地（現有 imbalance logic 多見於 `iris_analysis`），TDD 前置（`tests/test_ai4i_eda.py`）尚未建立。
- **建議**：實作 eda.py 時納入 one-hot Type 的相關性檢查；describe_features 僅涵蓋 5 感測器，EDA 應完整。

### E4（教學者／審稿）初評
- **結論**：TDD 順序必須：先寫 `tests/test_ai4i_eda.py`（紅燈）→ 實作 eda.py（綠燈）。
- **驗收對齊**：balance 總和 10000；describe 指定統計欄位；corr matrix 形狀；3 plots > 1KB。

### E5（方法學家）初評
- **結論**：EDA 是 imbalance 分析的正確起點；建議 EDA 輸出 **no-skill baseline** 作為建模前 baseline；之後 resampling/threshold 有依據。

### Round 1 小結
- 全員同意「3.4% 不平衡必須被具體呈現」。
- 分歧/待補：no-skill baseline 是否納入 EDA（E1/E5 支持，E3 偏工程取徑）；TDD 落地是否同步（E4/E3 支持）。

---

## 2) 第 2 輪討論（意見回饋與修正）

- **E1 → E5**：讚同 EDA 應含 no-skill baseline；補充 bootstrap CI 建議。
- **E5 → E1／E3**：EDA 不只「描述」，還應「預告」resampling/threshold 需求；兩位意見互補。
- **E3 → E4**：TDD 執行順序同意；工程實作上 one-hot 相關性納入 checklist。
- **E2 → E1／E3**：領域確認 baseline 數值（ROC≈0.5, PR≈0.034）合理；threshold/cost 留待後段。
- **E4 → 全員**：審核共識——驗收條件與 TDD 是排期前提。

### Round 2 小結
- 達成 consensus 基礎：EDA 應補 no-skill baseline；TDD 先行；engineer 建議納入 one-hot/corr。

---

## 3) 第 3 輪討論（最終輪）＋ 主席匯整

### Round 3 各位專家最終立場
- **E1**：最終站：EDA imbalance 呈現需含 prevalence + no-skill baseline；可量化模型改善空間。
- **E2**：最終站：domain 支持 imbalance 分析；threshold 後續決策。
- **E3**：最終站：工程面 TDD + eda.py 實作順序；one-hot 納入。
- **E4**：最終站：驗收/測試先行，紅燈後綠燈。
- **E5**：最終站：EDA → resampling/threshold pipeline 的 method 依據。

### 主席匯整（Final Synthesis，林建國 教授）
**最終結論（Consolidated Result）**：

1. **不平衡量化**：339/10000 ≈ 3.4%（prevalence ≈ 0.034）是 EDA 的核心產出，應完整呈現於報告（accept）。
2. **No-skill baseline**：EDA 應納入 ROC-AUC ≈ 0.5、PR-AUC ≈ 0.034 之 baseline（與 `evaluate.py` 既有 PR-AUC primary 一致）；`plot_class_imbalance` 應畫 baseline 線。
3. **實作順序**：`tests/test_ai4i_eda.py` 先紅燈 → `ai4i_analysis/eda.py` 實作 → 3 圖輸出至 `outputs/ai4i_figures/`。
4. **建模介入**：EDA 完成 imbalance 統計後，進入 #8（MLP 訓練）時以 PR-AUC 為 primary；threshold/cost-sensitive 決策留待 #8 後段。

**匯整後結果（Final Consolidated Output）**：
- 共識事項：3.4% imbalance 必須被量化呈現；no-skill baseline 建議納入 EDA；TDD 落地為前置。
- 待決事項：EDA 是否納入 one-hot Type 相關性／missing values（工程建議列入 checklist，本輪未決）。
- 對 issue #5 的行動建議：依序完成 EDA 實作與測試，後續以 PR-AUC / no-skill baseline 框架評估模型。

---

## 4. 匯整後的結果（Final Consolidated Result）

**Final outcome for issue #5（Delphi 第三輪匯整）**：

- **通過／建議通過的內容**：imbalance 比例（3.4% / 339/10000）具體呈現、3 圖表產出、測試通過。
- **建議補強**：EDA 加入 no-skill baseline 量化（PR-AUC ≈ 0.034 / ROC-AUC ≈ 0.5）並視覺化；TDD（tests/test_ai4i_eda.py）前置落地。
- **最終結論**：小組一致（主席認可）認為 issue #5 之不平衡分析應以「prevalence + no-skill baseline + 未來 resampling/threshold 依據」為框架；主席將此建議交由後續工作單（#7/#8）沿用。

- **結論摘要（Chair）**：本次 Delphi 共識 — **EDA 的 imbalance 分析應從「比例數字」進階為「數量+統計 baseline+視覺化」的完整不平衡分析**，並以 TDD 保證交付品質；後續模型（#8）應以 PR-AUC primary。
