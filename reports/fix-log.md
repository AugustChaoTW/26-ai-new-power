<!-- 修復紀錄：新增條目請加在最下方（生成器會全文附到 smell-report.md 的 §6） -->

### 基線（修復前）

- 日期：2026-10-23（commit `1f5703a` 之後）
- eslint：**0** 違規（complexity 10 / max-depth 3 / max-lines-per-function 50 / cognitive 15）
- jscpd：**3.91%**（token 計 8.12%），**12** 個重複片段（markup 7 + python 5）＝ #11
- knip：**0** 項（#13 舊文寫 1 項為過期數字，最新為 0）
- 測試數：Python pytest **50**（50 passed）+ 前端 golden **6**（6/6，tol 1e-6）＝ 合計 **56**，全綠 ✅
