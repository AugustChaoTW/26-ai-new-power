# -*- coding: utf-8 -*-
"""Generate reports/smell-report.md from existing static-check outputs.

Inputs (must already exist, produced by scripts/run_static_checks.sh or fresh
scoped runs):
  outputs/static-checks/eslint.json
  outputs/static-checks/jscpd/jscpd-report.json
  outputs/static-checks/knip.json

Counts frontend-inference code metrics by reading the files themselves.
Writes reports/smell-report.md. Never modifies any program.

Run:  python scripts/generate_smell_report.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "static-checks"
FRONTEND_FILES = ["docs/index.html", "docs/inference.html", "docs/app.js", "docs/trained_params.js"]
INFERENCE_SCOPE_FILES = [  # 「前端推論」程式的組成
    ("docs/inference.html", "表單與版面（JS 皆在外部 app.js）"),
    ("docs/app.js", "標準化＋前向傳播＋渲染（推論核心）"),
    ("docs/trained_params.js", "模型參數資料（Python 逐字複製）"),
    ("docs/index.html", "推論結果分頁（僅連結，不含推論程式）"),
]


def load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def eslint_section() -> tuple[str, list, int]:
    entries = load(OUT / "eslint.json", [])
    rows, total = [], 0
    by_rule: dict = {}
    for f in entries:
        rel = f.get("filePath", "").split("/26-ai-new-power/")[-1]
        for m in f.get("messages", []):
            by_rule.setdefault(m.get("ruleId") or "(internal)", []).append(f"{rel}:{m.get('line')}:{m.get('column')}")
            total += 1
    for rule, locs in sorted(by_rule.items()):
        rows.append(f"| {rule} | {len(locs)} | {', '.join(locs[:5])} |")
    md = (
        "| 規則 | 數量 | 位置（最多列 5） |\n| --- | ---: | --- |\n"
        + ("\n".join(rows) + "\n" if rows else "| （無任何違規） | 0 | — |\n")
    )
    return md, by_rule, total


def jscpd_section() -> str:
    jscpd = load(OUT / "jscpd" / "jscpd-report.json", {})
    total = jscpd.get("statistics", {}).get("total", {})
    pct = round(total.get("percentage", 0), 2)
    pct_tok = round(total.get("percentageTokens", 0), 2)
    by_format: dict = {}
    for fmt, st in (jscpd.get("statistics", {}).get("formats", {}) or {}).items():
        by_format[fmt] = (st.get("clones", 0), st.get("duplicatedLines", 0), st.get("lines", 0))
    md = [
        f"掃描範圍：`docs/` + `scripts/`。複製率 **{pct}%**（以 token 計 **{pct_tok}%**），"
        f"共 {len(jscpd.get('duplicates', []))} 個重複片段。\n",
        "| 格式 | 重複片段 | 重複行數 / 總行數 |",
        "| --- | ---: | --- |",
        *(f"| {fmt} | {c} | {dup} / {lines} |" for fmt, (c, dup, lines) in sorted(by_format.items())),
        "",
        "**重複片段位置**\n",
        "| # | 位置 A | 位置 B | tokens / 行數 |",
        "| ---: | --- | --- | --- |",
    ]
    for i, c in enumerate(jscpd.get("duplicates", []), 1):
        a = f"{c['firstFile']['name']}:{c['firstFile']['start']}-{c['firstFile']['end']}"
        b = f"{c['secondFile']['name']}:{c['secondFile']['start']}-{c['secondFile']['end']}"
        md.append(f"| {i} | `{a}` | `{b}` | {c.get('tokens')} / {c.get('lines')} |")
    return "\n".join(md) + "\n"


def knip_section() -> str:
    payload = load(OUT / "knip.json", {})
    issues = payload.get("issues", [])
    if isinstance(issues, dict):
        workspaces = [issues]
        pass
    workspaces = issues if isinstance(issues, list) else [issues]
    rows = []
    for ws in workspaces:
        for cat in ("files", "dependencies", "devDependencies", "unlisted", "binaries",
                    "unresolved", "exports", "types"):
            for e in ws.get(cat, []):
                detail = (e.get("name") or e.get("file") if isinstance(e, dict) else str(e))
                rows.append((cat, ws.get("file", "(workspace)"), detail))
    if not rows:
        return ("掃描範圍：`knip.json` entry = `docs/app.js`、`docs/trained_params.js`、`scripts/run_static_checks.sh`。\n\n"
                "**未使用項目：0**（無未使用檔案、未使用匯出/函式、未使用相依）\n\n"
                "> 註：此前曾偵測 `jscpd` devDependency「未使用」，原因是工具以 `npx` 呼叫未被 knip 看見；"
                "補上 `package.json` npm scripts（`checks:eslint` / `checks:jscpd` / `checks:knip`）後歸零，"
                "過程見 [#12]。(https://github.com/AugustChaoTW/26-ai-new-power/issues/12)\n")
    lines = ["| 類別 | 檔案 | 項目 |", "| --- | --- | --- |",
             *(f"| {c} | {f} | {d} |" for c, f, d in rows)]
    return "\n".join(lines) + "\n"


def func_count(path: Path) -> tuple[int, int, int]:
    """(具名函式, 全部 function 關鍵字出現次數, 暫時未用) — 註解與字串中的 'function' 一併檢查過出現在 app.js 純碼。"""
    src = path.read_text(encoding="utf-8")
    decls = len(re.findall(r"\bfunction\s+\w+\s*\(", src))
    allf = len(re.findall(r"\bfunction\b", src))  # 全部關鍵字（含匿名 / IIFE / 回呼）
    return decls, allf, allf - decls


def frontend_metrics() -> tuple[str, int, int]:
    md = [
        "「前端推論」＝使用者在瀏覽器輸入參數、由 `app.js` 以 Python 訓練好的參數做前向傳播的整組程式，"
        "組成與統計如下（行數含 HTML 版面；註：`index.html` 只列出連結、不含推論程式）。\n",
        "| 檔案 | 角色 | 總行數 | 非空白行 | 函式（具名 / 匿名） |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    total_funcs = 0
    grand_lines = 0
    for rel, role in INFERENCE_SCOPE_FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        src_lines = p.read_text(encoding="utf-8").splitlines()
        nonblank = sum(1 for x in src_lines if x.strip())
        if p.suffix == ".js":
            named, allf, _ = func_count(p)
            fcount = f"{named} / {allf - named}"
        else:
            named = 0
            fcount = "0 / 0"
        md.append(f"| `{rel}` | {role} | {len(src_lines)} | {nonblank} | {fcount} |")
        total_funcs += allf if p.suffix == ".js" else 0
        grand_lines += len(src_lines)
    md.append(f"\n**小計：前端推論相關檔案 {len(INFERENCE_SCOPE_FILES)} 個，"
              f"共 {grand_lines} 行；其中 JavaScript 推論核心 `app.js` "
              f"含 {total_funcs} 個函式。**\n")
    return "\n".join(md), total_funcs, grand_lines


def main() -> None:
    eslint_md, _, eslint_total = eslint_section()
    jscpd_md = jscpd_section()
    knip_md = knip_section()
    fe_md, fe_funcs, fe_loc = frontend_metrics()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = f"""<!-- 生成：scripts/generate_smell_report.py，來源 outputs/static-checks；勿手改數字 -->

# 程式碼氣味報告（smell report）

針對 `docs/`（瀏覽器端 JS/HTML）與 `scripts/`（部署建置腳本）執行之靜態檢查彙整：
ESLint + eslint-plugin-sonarjs（門檻 complexity 10 / max-depth 3 / max-lines-per-function 50 / cognitive-complexity 15）、jscpd（重複率，min-tokens 50）、knip（死碼）。
生成時間：**{now}**；產出水準：只搬真實數字，未重算未修改任何程式。

## 1. 警告總數（依規則分組）

eslint 違規總數：**{eslint_total}**。

{eslint_md}
門檻設定位置：`eslint.config.mjs`（browser/node 兩組範圍分開宣告全域）。

## 2. 重複率與重複片段位置

{jscpd_md}

## 3. 未使用的檔案與函式

{knip_md}

## 4. 前端推論相關程式

{fe_md}

## 5. 觸發的追蹤單

- [#11][11] jscpd 重複片段追蹤單——OPEN（重跑指令：`bash scripts/run_static_checks.sh`）
- [#12][12] knip 死碼 1 項 ——已解決 CLOSED
- [#13][13] 基線摘要——OPEN（持續更新）

[11]: https://github.com/AugustChaoTW/26-ai-new-power/issues/11
[12]: https://github.com/AugustChaoTW/26-ai-new-power/issues/12
[13]: https://github.com/AugustChaoTW/26-ai-new-power/issues/13
"""
    report_path = ROOT / "reports" / "smell-report.md"
    fixlog_path = ROOT / "reports" / "fix-log.md"
    if fixlog_path.exists():
        md += "\n## 6. 修復紀錄（Lab4：#11 / #13 / #14，一次一項）\n\n" + fixlog_path.read_text(encoding="utf-8")
    report_path.write_text(md, encoding="utf-8")
    print(f"written {report_path}（前端推論 {fe_funcs} 函式 / {fe_loc} 行）")


if __name__ == "__main__":
    main()
