<!-- 2026-10-23，彙整當日 5 項已完成的修復／強化工作；數字取自 git 歷史與 outputs/static-checks/；勿手改數字 -->

# 修正前後報告（before / after）

範圍：2026-10-23 完成的 5 項工作（A–E），期間 commit：`22943da`（static checks 基線）、`bb690f7`（typo）、`f200058`（golden parity）、`1f5703a`（結論測試）、`fe56e52`（README 數字）。
專案起點：`04db5f3`（Iris 分析完成，pytest 17 passed，無任何靜態檢查）。

## 1. 指標修正前後對照

| 指標 | 修正前 | 修正後 | 說明／時點 |
| --- | ---: | ---: | --- |
| 警告數（eslint） | 未量測 | **0** | eslint 配置於 `22943da` 才誕生，之前無量測值；基線後三次掃描（05:30／05:47／最新）皆 0 |
| 複雜度（sonarjs complexity ≤10、max-depth ≤3、max-lines-per-function ≤50、cognitive ≤15） | 未量測 | **0** 違規 | 基線首掃即全數低於門檻，無初始高峰可引用 |
| 重複率（jscpd，min-tokens 50） | **2.32%**（11 片段） | **3.91%**（12 片段；token 計 8.12%） | **上升**：05:47 的 `app.js` 擴充與新頁面帶入 `report.html ⇄ smell.html` 1167-token／50 行的新片段（=#11 追蹤中，未修）。另：進行中的未提交重排掃描已降至 **1.29%**（僅 1 片段），屬平行工作，本報告不計入 |
| 死碼（knip） | **1** 項 | **0** 項 | jscpd devDependency 被誤判未使用（npx 呼叫 knip 看不見）→ `package.json` 補 npm scripts 後歸零（#12，CLOSED） |
| 測試數 | **17**（pytest） | **50**（pytest）＋**6**（前端 golden）＝**56** | 時間線：17（`04db5f3`）→ 25（`04fe357`）→ 46（`9daa716`）→ 50（`1f5703a`）；golden 6 起於 `f200058`。全套件與 golden 全綠 |

> 口徑註記：`outputs/static-checks/issues_summary.json` 中的 `jscpd_percentage: 2.31` 是 runner 首輪（含 `tests/`、分母較大）的實時摘要，其後重掃未覆寫；時間線權威值取 `jscpd-report.json`（2.32→3.11% 同 11 片段 → 3.91% 12 片段）。

## 2. 五項 diff 摘要

### A. knip 死碼誤報修復（#12 → CLOSED）— commit `22943da`
- 檔案：`package.json`（+36，新檔，scripts 區塊與 devDependencies 同時建立；`f200058` 再補 `"test:golden"` 1 行）
- 內容：新增 `checks` / `checks:eslint` / `checks:jscpd` / `checks:knip` npm scripts，使 jscpd 以 package.json scripts 呼叫、knip 得以看見使用記錄，死碼誤報 1 → 0

```json
"checks": "bash scripts/run_static_checks.sh",
"checks:jscpd": "jscpd --min-tokens 50 --reporters json,consoleFull --output outputs/static-checks/jscpd docs scripts tests",
"checks:knip": "knip --config knip.json --reporter json --no-exit-code"
```

### B. eslint＋sonarjs 基線建立 — commit `22943da`
- 檔案：`eslint.config.mjs`（+84，新檔）、`scripts/run_static_checks.sh`（+61，新檔）、`knip.json`（+6）、`scripts/static_check_issues.py`（+213）、`package.json`（+36）、`.gitignore`（+1）；合計 7 files，+2722（含 package-lock）
- 內容：browser／Node 端 globals 分組宣告、四種複雜度門檻、單指令 runner（eslint → jscpd → knip → 摘要/開 issue）

```js
const commonRules = { "no-undef": "error", complexity: ["error", 10], "max-depth": ["error", 3], "max-lines-per-function": ["error", 50] };
// docs/**/*.js → globals.browser；scripts/tests → commonjs + globals.node；sonarjs/cognitive-complexity 15
```

### C. 輸出欄位 typo 修復 — commit `bb690f7`
- 檔案：`scripts/export_ai4i_metrics.py`（1 insertion / 1 deletion）
- 內容：console 輸出 `poc_auc` → `roc_auc`（值域鍵本來就正確，純 print 文字 typo）

```diff
-  print(f"  {name}: pr_auc={m['pr_auc']:.4f} poc_auc={m['roc_auc']:.4f} f1={m['f1']:.4f} acc={m['accuracy']:.4f}")
+  print(f"  {name}: pr_auc={m['pr_auc']:.4f} roc_auc={m['roc_auc']:.4f} f1={m['f1']:.4f} acc={m['accuracy']:.4f}")
```

### D. 前端 ↔ Python golden parity 測試（#14）— commit `f200058`
- 檔案：`scripts/export_golden.py`（+214，新檔）、`tests/golden.json`（+163，新檔）、`tests/golden.test.js`（+119，新檔）、`docs/app.js`（+5）、`package.json`（+1）；合計 9 files，+689/−142
- 內容：seed=42 重訓 Iris／AI4I MLP → 8 項參數與 `docs/trained_params.js` 逐位比對（max diff **0.0**）→ 每模型 3 筆固定輸入寫 golden；Node 端以 document stub 載入 `trained_params.js`＋`app.js`（經新暴露的 `window.PREDICT_API`）比對同一組輸入

```js
if (typeof window !== 'undefined') {
  window.PREDICT_API = { predict_iris: predict_iris, predict_ai4i: predict_ai4i };
}
```
- 實測：**6/6 PASS**（tol 1e-6），前端與 sklearn 機率最大差 **1.30e-18**

### E. 結論測試＋scaler 持久化＋README 數字 — commits `1f5703a`、`fe56e52`
- 檔案：`tests/test_ai4i_conclusions.py`（+325，新檔，4 個結論測試）、`ai4i_analysis/model.py`（+132 / −7）、`README.md`（3 / 3 行）、`tests/golden.json`（1 行時間戳）
- 內容：4 條實驗結論固化为不變量測試（不含洩漏欄／模型非全猜多數類／results.json ↔ 報告逐字 50 個數字／scaler 存檔重現），每條經「改壞 → 紅燈 → 還原」突變驗證

```diff
-LEAKAGE_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
+LEAKAGE_COLUMNS = [..., "Machine failure", ...]        # 目標本身加入洩漏清單
+def prepare_training_data_with_state(...)  # 暴露已擬合 scaler
+def save_preprocessing(path, scaler, feature_names) / load_preprocessing / rebuild_scaler
```

## 3. 哪一項最難修，為什麼

**最難：D（golden parity）。**

1. **跨語言的逐位一致要求**：要讓瀏覽器端手抄的參數（`trained_params.js`）＋JavaScript 前向傳播，與 Python 端 sklearn 重訓結果逐位吻合——特徵順序、one-hot 排序、StandardScaler 參數、激活函數行為任何一處漂移都會 silent 失配（這正是 #11 留言指出的 UI 無警示風險），所以必須先做「重訓參數 vs 參數檔 8 項逐位比對」的守門，失配直接 RuntimeError。
2. **測試運行環境差異**：`golden.test.js` 要在 Node（CommonJS）中模擬瀏覽器載入——需 document stub、嚴格按 `trained_params.js` → `app.js` 的載入順序、並為此微調 `app.js`（+5 行 `window.PREDICT_API`）同時保持瀏覽器行為不變、且 19 個既有函式不破壞 eslint 基線。
3. **浮點精度的容差設計**：二進位往返造成 ~1e-18 的微小差異，容差定 1e-6 是「寬到穩定、緊到抓得住義務漂移」的取捨；差異若真漂移（特徵錯位）是數量級的，這個測試必須可靠地翻紅。

第二難是 E（結論測試）：「測結論不測函式」的斷言設計＋bit-identical 重現（第一版用 `inverse_transform` 反推原始資料在浮點往返下不一致，改為同參數重建原始特徵才全等），以及結論 3 的「報告範圍」界定（ai4i 數字尚無報告可比，只能鎖來源標記）。

相對地，C 是 1 行 typo，A/B 是配置性修復（一次到位、判斷點少）：難度排序 **D > E ≫ B > A > C**。

---
*本檔由 agent 依 git 歷史與 `outputs/static-checks/` 實測數字彙整；進行中的未提交重排（共用 CSS 方案）不在統計內。*
