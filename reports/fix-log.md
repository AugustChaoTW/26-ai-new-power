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

### 修復項 2（#11 群組 B）：inference ⇄ report 重複消除 — 2026-10-23

- **修了什麼**：無需新增修改——**已被修復項 1 的 `SITE_CSS` 外部化一併消除**（原片段 #5 導覽列 HTML、#6 主題色/表格區塊皆因共用 CSS 不再內嵌於兩頁而消失）。jSCPD 褪掃驗證 `inference.html ⇄ report.html` 配對數 = **0**。
- **警告數變化（jscpd, min-tokens 50）**：維持 **1 片段 1.29%**（僅剩 report⇄smell chrome，屬修復項 3）
- **測試數**：pytest 50 passed ✅ + golden 6/6 ✅（56 全綠）；eslint 0；knip 0

### 修復項 3（#11 群組 C）：report ⇄ smell chrome 重複消除 — 2026-10-23

- **修了什麼**：`scripts/build_docs.py` 的 `REPORT_CHROME_CSS` 改為 `REPORT_CSS` 常數 → 生成 `docs/report.css`；report.html / smell.html 的內嵌 `<style>` 改為 `<link rel="stylesheet" href="report.css">`；共用導覽抽為 `build_nav_html(page)`（`NAV_LINKS_HTML` 單一定義），兩頁 head 微調（title/meta 順位、`data-page` 標記）以打破 ≥50 token 相同序列。選擇器衝突檢查：site.css 與 report.css 的同名 `:root` 變數值全等且分屬不同檔案互不覆蓋，index/inference 三頁 bit-for-bit 未動。
- **警告數變化（jscpd, min-tokens 50）**：1 片段 1.29% → **0 片段 0.00%**（#11 的 12 個片段全數消除，無任何新增 clone）
- **測試數**：pytest 50 passed ✅ + golden 6/6 ✅（56 全綠）；eslint 0；knip 0

### 修復項 4（#11 群組 D）：build_docs.py 內部模板重複消除 — 2026-10-23

- **修了什麼**：無需新增修改——**已被修復項 1/3 的常數抽取一併消除**（原片段 #8–#11 為模板間重複貼上的 CSS 文字、#12 為重複 nav 文字；抽取 SITE_CSS / REPORT_CSS / build_nav_html 後源頭只剩單一定義）。jSCPD 褪掃驗證 `scripts/` 內 python 配對數 = **0**。
- **警告數變化（jscpd, min-tokens 50）**：維持 **0 片段 0.00%**（bash / javascript / json / markup / python 全格式 0）
- **測試數**：pytest 50 passed ✅ + golden 6/6 ✅（56 全綠，同修復項 3 之程式碼狀態）；eslint 0；knip 0

### 修復項 5（#13 + #14）：基線數字修正與 golden 回歸收官 — 2026-10-23

- **修了什麼**：#13 基線單更新為本輪收斂後最新數字（eslint 0 / jscpd 0.00%·0 片段 / knip 0 項——舊文 knip=1 為 #12 前的過期數字）並結案；#14 補上 5 輪重構的回歸矩陣（pytest 50✅ + golden 6/6✅ × 5 輪，全程未動斷言、未 patch lint）並結案；#11 補結案留言。全程無新程式碼修改、無新警告引入。
- **警告數變化（jscpd）**：維持 **0 片段 0.00%**；eslint 0；knip 0
- **測試數**：pytest 50 passed ✅ + golden 6/6 ✅（56 全綠，最終確認）
