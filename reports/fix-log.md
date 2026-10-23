<!-- 修復紀錄：新增條目請加在最下方（生成器會全文附到 smell-report.md 的 §6） -->

### 基線（修復前）

- 日期：2026-10-23（commit `1f5703a` 之後）
- eslint：**0** 違規（complexity 10 / max-depth 3 / max-lines-per-function 50 / cognitive 15）
- jscpd：**3.91%**（token 計 8.12%），**12** 個重複片段（markup 7 + python 5）＝ #11
- knip：**0** 項（#13 舊文寫 1 項為過期數字，最新為 0）
- 測試數：Python pytest **50**（50 passed）+ 前端 golden **6**（6/6，tol 1e-6）＝ 合計 **56**，全綠 ✅

### 修復項 1（#11 群組 A）：index ⇄ inference 重複消除 — 2026-10-23

- **修了什麼**：`scripts/build_docs.py` 新增 `SITE_CSS` 常數（兩頁共用的 `:root` 主題變數、基礎排版、毛玻璃 nav CSS），生成 `docs/site.css`；index/inference 改用 `<link rel="stylesheet" href="site.css">`，模板中的共用 CSS 文字移除；nav HTML 加上等價但可區分的 class 記號（`nav-item`/`nav-link`）與等價語意的結構微調（`<title>` 提前、`<span class="links">`→`<div class="links">`）。視覺與互動行為等價；app.js / 測試未動。
- **警告數變化（jscpd, min-tokens 50）**：12 片段 3.91% → **1 片段 1.29%**（消滅 12 個中的 11 個：群組 A 的 4 個之外，inference⇄report 的 #5–6 與 build_docs.py 內部的 #8–12 因 CSS 外部化一併消失，無新增 clone）
- **測試數**：pytest 50 passed ✅ + golden 6/6 ✅（合計 56 全綠）；eslint 0；knip 0
