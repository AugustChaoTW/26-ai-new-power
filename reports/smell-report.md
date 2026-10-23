<!-- 生成：scripts/generate_smell_report.py，來源 outputs/static-checks；勿手改數字 -->

# 程式碼氣味報告（smell report）

針對 `docs/`（瀏覽器端 JS/HTML）與 `scripts/`（部署建置腳本）執行之靜態檢查彙整：
ESLint + eslint-plugin-sonarjs（門檻 complexity 10 / max-depth 3 / max-lines-per-function 50 / cognitive-complexity 15）、jscpd（重複率，min-tokens 50）、knip（死碼）。
生成時間：**2026-10-23 06:12 UTC**；產出水準：只搬真實數字，未重算未修改任何程式。

## 1. 警告總數（依規則分組）

eslint 違規總數：**0**。

| 規則 | 數量 | 位置（最多列 5） |
| --- | ---: | --- |
| （無任何違規） | 0 | — |

門檻設定位置：`eslint.config.mjs`（browser/node 兩組範圍分開宣告全域）。

## 2. 重複率與重複片段位置

掃描範圍：`docs/` + `scripts/`。複製率 **3.91%**（以 token 計 **8.12%**），共 12 個重複片段。

| 格式 | 重複片段 | 重複行數 / 總行數 |
| --- | ---: | --- |
| bash | 0 | 0 / 61 |
| javascript | 0 | 0 / 341 |
| json | 0 | 0 / 839 |
| markup | 7 | 112 / 1124 |
| python | 5 | 39 / 1498 |

**重複片段位置**

| # | 位置 A | 位置 B | tokens / 行數 |
| ---: | --- | --- | --- |
| 1 | `index.html:6-11` | `inference.html:6-11` | 92 / 6 |
| 2 | `index.html:11-19` | `inference.html:12-20` | 100 / 9 |
| 3 | `index.html:22-31` | `inference.html:22-30` | 232 / 10 |
| 4 | `index.html:106-115` | `inference.html:92-101` | 61 / 10 |
| 5 | `inference.html:20-29` | `report.html:14-21` | 208 / 10 |
| 6 | `inference.html:84-100` | `report.html:48-55` | 108 / 17 |
| 7 | `report.html:6-55` | `smell.html:6-55` | 1167 / 50 |
| 8 | `build_docs.py:204-211` | `build_docs.py:537-546` | 208 / 8 |
| 9 | `build_docs.py:205-211` | `build_docs.py:285-292` | 207 / 7 |
| 10 | `build_docs.py:275-282` | `build_docs.py:530-537` | 99 / 8 |
| 11 | `build_docs.py:291-296` | `build_docs.py:545-549` | 74 / 6 |
| 12 | `build_docs.py:369-378` | `build_docs.py:609-618` | 61 / 10 |


## 3. 未使用的檔案與函式

掃描範圍：`knip.json` entry = `docs/app.js`、`docs/trained_params.js`、`scripts/run_static_checks.sh`。

**未使用項目：0**（無未使用檔案、未使用匯出/函式、未使用相依）

> 註：此前曾偵測 `jscpd` devDependency「未使用」，原因是工具以 `npx` 呼叫未被 knip 看見；補上 `package.json` npm scripts（`checks:eslint` / `checks:jscpd` / `checks:knip`）後歸零，過程見 [#12]。(https://github.com/AugustChaoTW/26-ai-new-power/issues/12)


## 4. 前端推論相關程式

「前端推論」＝使用者在瀏覽器輸入參數、由 `app.js` 以 Python 訓練好的參數做前向傳播的整組程式，組成與統計如下（行數含 HTML 版面；註：`index.html` 只列出連結、不含推論程式）。

| 檔案 | 角色 | 總行數 | 非空白行 | 函式（具名 / 匿名） |
| --- | --- | ---: | ---: | --- |
| `docs/inference.html` | 表單與版面（JS 皆在外部 app.js） | 155 | 144 | 0 / 0 |
| `docs/app.js` | 標準化＋前向傳播＋渲染（推論核心） | 207 | 189 | 17 / 2 |
| `docs/trained_params.js` | 模型參數資料（Python 逐字複製） | 134 | 133 | 0 / 0 |
| `docs/index.html` | 推論結果分頁（僅連結，不含推論程式） | 250 | 229 | 0 / 0 |

**小計：前端推論相關檔案 4 個，共 746 行；其中 JavaScript 推論核心 `app.js` 含 19 個函式。**


## 5. 觸發的追蹤單

- [#11][11] jscpd 重複片段追蹤單——OPEN（重跑指令：`bash scripts/run_static_checks.sh`）
- [#12][12] knip 死碼 1 項 ——已解決 CLOSED
- [#13][13] 基線摘要——OPEN（持續更新）

[11]: https://github.com/AugustChaoTW/26-ai-new-power/issues/11
[12]: https://github.com/AugustChaoTW/26-ai-new-power/issues/12
[13]: https://github.com/AugustChaoTW/26-ai-new-power/issues/13

## 6. 修復紀錄（Lab4：#11 / #13 / #14，一次一項）

<!-- 修復紀錄：新增條目請加在最下方（生成器會全文附到 smell-report.md 的 §6） -->

### 基線（修復前）

- 日期：2026-10-23（commit `1f5703a` 之後）
- eslint：**0** 違規（complexity 10 / max-depth 3 / max-lines-per-function 50 / cognitive 15）
- jscpd：**3.91%**（token 計 8.12%），**12** 個重複片段（markup 7 + python 5）＝ #11
- knip：**0** 項（#13 舊文寫 1 項為過期數字，最新為 0）
- 測試數：Python pytest **50**（50 passed）+ 前端 golden **6**（6/6，tol 1e-6）＝ 合計 **56**，全綠 ✅
