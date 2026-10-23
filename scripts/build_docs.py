# -*- coding: utf-8 -*-
"""Build the static docs/ website from existing project artifacts.

Instinct: no retraining, no computations, no invented numbers.
 - Figures are copied verbatim (no re-plotting) from outputs/figures/ (and
   outputs/ai4i_figures/ when that directory exists).
 - Every metric in docs/results.json is parsed from reports/paper.md (Iris
   study). Each entry carries its source section.
 - reports/paper.md is converted to docs/report.html (markdown -> HTML).
 - Homepage docs/index.html is pure HTML + JS, reading results.json with
   fetch() (no build tooling, no framework).

Existing layout note: the repo has no src/ and no reports/figures/ — code
lives in iris_analysis/ / ai4i_analysis/ / scripts/, figures in
outputs/figures/.

Usage: python scripts/build_docs.py
"""

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGURES_OUT = ROOT / "outputs" / "figures"
AI4I_FIGURES_OUT = ROOT / "outputs" / "ai4i_figures"
PAPER = ROOT / "reports" / "paper.md"

# ---------------------------------------------------------------------------
# 1. Figures — verbatim copies only
# ---------------------------------------------------------------------------


def copy_figures() -> list[dict]:
    dest = DOCS / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for src_dir in (FIGURES_OUT, AI4I_FIGURES_OUT):
        if not src_dir.is_dir():
            continue
        for png in sorted(src_dir.glob("*.png")):
            shutil.copy2(png, dest / png.name)
            copied.append({
                "file": f"figures/{png.name}",
                "from": str(png.relative_to(ROOT)),
                "bytes": png.stat().st_size,
            })
            print(f"figure copied: {png.name}  <-  {png.relative_to(ROOT)}")
    return copied


# ---------------------------------------------------------------------------
# 2. Metrics — parsed from reports/paper.md (source of truth), never computed
# ---------------------------------------------------------------------------

SECTION_HEADER = re.compile(r"^## (\d+\.)\s")
SUBSECTION_HEADER = re.compile(r"^### (\d+\.\d+)\s")


def split_sections(text: str) -> dict:
    """Return {'5.2': {'title': ..., 'text': ..., 'tables': [...]}, ...}."""
    sections: dict = {}
    current_key, current_lines, current_title = None, [], ""

    def flush():
        nonlocal current_lines
        if current_key is None:
            return
        tables = []
        rows = []
        for line in current_lines:
            if line.lstrip().startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                rows.append(cells)
            elif rows:
                tables.append(rows)
                rows = []
        if rows:
            tables.append(rows)
        sections[current_key] = {
            "title": current_title,
            "text": "\n".join(current_lines),
            "tables": [t for t in tables if len(t) >= 2],
        }

    for line in text.splitlines():
        sub = SUBSECTION_HEADER.match(line)
        top = SECTION_HEADER.match(line)
        if sub or top:
            flush()
            nonlocal_key = (sub or top)
            key = (sub.group(1) if sub else top.group(1)).rstrip(".")
            current_key = key
            current_title = re.sub(r"^#+\s*\d+(?:\.\d+)*\.?\s*", "", line).strip()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)
    flush()
    return sections


def parse_rows(table: list) -> list:
    """Drop markdown separator rows; return list of cell lists."""
    out = []
    for row in table:
        if all(re.fullmatch(r":?-{2,}:?", c) for c in row if c != ""):
            continue
        out.append(row)
    return out


def parse_paper(paper_path: Path) -> dict:
    text = paper_path.read_text(encoding="utf-8")
    sec = split_sections(text)

    def table_rows(section_key: str) -> list:
        return parse_rows(sec[section_key]["tables"][0])

    iris: dict = {"source": "reports/paper.md"}

    # §5.2 整體分類表現：指標 -> 數值
    iris["metrics"] = {
        row[0]: {"value": row[1], "source": f"reports/paper.md §5.2 表格「{row[0]}」"}
        for row in table_rows("5.2")[1:] if len(row) > 1
    }

    # §5.4 各類別指標（保留整列原文）
    iris["per_class"] = {
        "header": table_rows("5.4")[0],
        "rows": table_rows("5.4")[1:],
        "source": "reports/paper.md §5.4 表格",
    }

    # §2.3 分類分布
    iris["class_balance"] = {
        "header": table_rows("2.3")[0],
        "rows": table_rows("2.3")[1:],
        "source": "reports/paper.md §2.3 表格",
    }

    # §3.1 基本檢查：檢查項目 -> 結果
    iris["dataset_checks"] = {
        row[0]: row[1] for row in table_rows("3.1")[1:] if len(row) > 1
    }
    iris["dataset_checks_source"] = "reports/paper.md §3.1 表格"

    # §4.1 資料切分 / §4.3 模型設定
    iris["split"] = {
        "header": table_rows("4.1")[0],
        "rows": table_rows("4.1")[1:],
        "source": "reports/paper.md §4.1 表格（random_state = 42 取自 §4.1 內文）",
    }
    iris["model_config"] = {
        row[1]: {"value": row[2], "source": "reports/paper.md §4.3 表格"}
        for row in table_rows("4.3")[1:] if len(row) > 2
    }

    # §5.1 / §3.5 內文數字（regex，無則報錯——絕不自己補）
    m = re.search(r"初始的 ([\d.]+) 下降至 ([\d.]+)，共迭代 (\d+) 次", sec["5.1"]["text"])
    assert m, "§5.1 loss numbers not found in the paper"
    iris["loss"] = {
        "initial": m.group(1), "final": m.group(2), "iterations": m.group(3),
        "source": "reports/paper.md §5.1 內文",
    }
    m = re.search(
        r"解釋變異量比例分別為 ([\d.]+)%、([\d.]+)%、([\d.]+)%，累積達 ([\d.]+)%",
        sec["3.5"]["text"],
    )
    assert m, "§3.5 PCA variance numbers not found in the paper"
    iris["pca"] = {
        "explained_variance_pct": [m.group(1), m.group(2), m.group(3)],
        "cumulative_pct": m.group(4),
        "source": "reports/paper.md §3.5 內文",
    }

    ai4i_note = {
        "status": "ai4i_analysis/evaluate.py 已存在（issue #9），但目前尚未產生任何持久化的指標或圖，因此這裡不放任何 AI4I 數字",
        "expects": [
            "outputs/ai4i_figures/ (evaluate.py 產出)",
            "ai4i_analysis.evaluate.compare_models() 比較表",
        ],
    }
    return {"iris": iris, "ai4i": ai4i_note}


# ---------------------------------------------------------------------------
# 3. Report conversion — reports/paper.md -> docs/report.html
# ---------------------------------------------------------------------------


def build_report_html(paper_path: Path) -> str:
    import markdown

    md = markdown.Markdown(extensions=["tables"])
    body = md.convert(paper_path.read_text(encoding="utf-8"))
    body = body.replace('src="../outputs/figures/', 'src="figures/')
    return (
        '<!DOCTYPE html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Iris 資料集分析小論文</title>\n"
        "<style>body{font-family:sans-serif;max-width:860px;margin:0 auto;padding:24px;"
        "line-height:1.6}img{max-width:100%}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:6px 10px}code{background:#f4f4f4}</style>\n"
        "</head>\n<body>\n"
        '<p><a href="index.html">← 回分析總覽</a>｜<a href="inference.html">互動推論示範</a></p>\n'
        + body
        + "\n</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# 4. Homepage — pure HTML + JS reading results.json
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>機器學習專案分析總覽</title>
<style>
  body{font-family:"PingFang TC","Microsoft JhengHei","Noto Sans TC",sans-serif;
       margin:0;background:#fafafa;color:#222;line-height:1.55}
  main{max-width:900px;margin:0 auto;padding:24px 16px 64px}
  h1{margin:0 0 4px}
  .subtitle{color:#666;margin:0 0 20px}
  .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px 20px;margin-bottom:20px}
  .badges{display:flex;flex-wrap:wrap;gap:12px}
  .badge{background:#eef4fb;border:1px solid #c7d9ee;border-radius:8px;padding:10px 14px;min-width:120px}
  .badge .num{font-size:1.45rem;font-weight:700;font-variant-numeric:tabular-nums}
  .badge .label{color:#555;font-size:.85rem}
  .badge .src{color:#999;font-size:.75rem;margin-top:2px}
  table{border-collapse:collapse;font-size:.9rem}
  th,td{border:1px solid #ddd;padding:5px 10px}
  th{background:#f2f2f2}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
  .fig{background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden}
  .fig img{width:100%;display:block}
  .fig figcaption{padding:8px 10px;font-size:.85rem;color:#444}
  .muted{color:#777}
  .warn{background:#fff6e6;border:1px solid #e8d5a0;border-radius:8px;padding:12px 16px;margin-bottom:20px}
  a{color:#2b62a8}
</style>
</head>
<body>
<main>
  <h1>機器學習專案分析總覽</h1>
  <p class="subtitle">數字與圖 100% 來自既有產出（reports/paper.md、outputs/figures/），由 scripts/build_docs.py 搬移而成，不重新訓練、不重算。</p>

  <div class="card" id="error-box" style="display:none"></div>

  <div id="content" style="display:none">
    <div class="card">
      <h2 id="m-title">—</h2>
      <div class="badges" id="m-badges"></div>
    </div>
    <div class="card">
      <h2>損失收斂、PCA 與資料規模</h2>
      <div id="misc"></div>
    </div>
    <div class="card">
      <h2>類別分布</h2>
      <div id="balance"></div>
    </div>
    <div class="card">
      <h2>既有圖表（原檔複製，未重繪）</h2>
      <div class="grid" id="figs"></div>
    </div>
    <div class="card" id="ai4i-note"></div>
    <div class="card">
      <h2>延伸頁面</h2>
      <ul>
        <li><a id="l-report" href="report.html">小論文全文（HTML 版）</a></li>
        <li><a href="inference.html">互動推論示範（Iris / AI4I）</a></li>
      </ul>
    </div>
  </div>
</main>
<script>
(function () {
  var el = function (id) { return document.getElementById(id); };
  var box = el('error-box');
  fetch('results.json').then(function (r) {
    if (!r.ok) { throw new Error('results.json HTTP ' + r.status); }
    return r.json();
  }).then(function (data) {
    var iris = data.iris;
    el('m-title').textContent = 'Iris 分類指標（來源：' + (data.sources && data.sources.paper || 'reports/paper.md') + '）';
    var badges = '';
    Object.keys(iris.metrics).forEach(function (k) {
      var m = iris.metrics[k];
      badges += '<div class="badge"><div class="num">' + m.value.replace('\\n', '') +
        '</div><div class="label">' + k + '</div><div class="src">' + m.source + '</div></div>';
    });
    el('m-badges').innerHTML = badges;

    var misc = '<table>';
    var loss = iris.loss, pca = iris.pca, checks = iris.dataset_checks;
    misc += '<tr><th>訓練損失初始 / 最終 / 迭代次數</th><td>' + loss.initial + ' / ' + loss.final + ' / ' + loss.iterations +
      '</td><td class="muted">' + loss.source + '</td></tr>';
    misc += '<tr><th>PCA 前三主成分解釋變異量 (累積)</th><td>' + pca.explained_variance_pct.join('% / ') + '%（累積 ' + pca.cumulative_pct + '%）</td><td class="muted">' + pca.source + '</td></tr>';
    Object.keys(checks).forEach(function (k) {
      misc += '<tr><th>' + k + '</th><td>' + checks[k] + '</td><td class="muted">' + iris.dataset_checks_source + '</td></tr>';
    });
    misc += '</table>';
    el('misc').innerHTML = misc;

    var bal = iris.class_balance;
    var t = '<table><tr>';
    bal.header.forEach(function (h) { t += '<th>' + (h || '(空)') + '</th>'; });
    t += '</tr>';
    bal.rows.forEach(function (r) { t += '<tr>'; r.forEach(function (c) { t += '<td>' + c + '</td>'; }); t += '</tr>'; });
    t += '</table><p class="muted">來源：' + bal.source + '</p>';
    el('balance').innerHTML = t;

    var figs = '';
    (data.figures || []).forEach(function (f) {
      figs += '<a class="fig" href="' + f.file + '" target="_blank"><img src="' + f.file + '" alt="' + f.file + '">' +
        '<figcaption>' + f.file + '（複製自 ' + f.from + '）</figcaption></a>';
    });
    el('figs').innerHTML = figs || '<p class="muted">無圖表。</p>';

    el('ai4i-note').innerHTML =
      '<h2>AI4I 2020</h2><p class="muted">' + data.ai4i.status + '。互動推論示範已有 AI4I 表單（inference.html）。</p>' +
      '<ul>' + (data.ai4i.expects || []).map(function (x) { return '<li><code>' + x + '</code></li>'; }).join('') + '</ul>';

    el('content').style.display = '';
  }).catch(function (e) {
    box.style.display = '';
    box.innerHTML = '<strong>無法載入 results.json：' + e.message + '</strong><br>' +
      '由於瀏覽器安全限制，file:// 直開無法 fetch JSON。請在本目錄以靜態伺服器開啟：' +
      '<code>python -m http.server 8000</code> 後訪問 <code>http://localhost:8000/docs/</code>';
  });
})();
</script>
</body>
</html>
"""


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)

    figures = copy_figures()

    results = parse_paper(PAPER)
    results["sources"] = {"paper": "reports/paper.md", "figures_dir": str(FIGURES_OUT.relative_to(ROOT))}
    results["sources_css_note"] = "figures 已複製至 docs/figures/（逐字元複製原檔）"
    results["figures"] = figures

    out_json = DOCS / "results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"results.json written: {out_json}")

    report_html = DOCS / "report.html"
    report_html.write_text(build_report_html(PAPER), encoding="utf-8")
    print(f"report.html written: {report_html}")

    # previous inference page (built earlier) moves to inference.html
    index_path = DOCS / "index.html"
    inference_path = DOCS / "inference.html"
    if index_path.exists() and "互動式推論示範" in index_path.read_text(encoding="utf-8"):
        shutil.copy2(index_path, inference_path)
        txt = inference_path.read_text(encoding="utf-8")
        txt = txt.replace(
            "模型參數於 Python（scikit-learn，random_state=42）訓練後直接複製到前端",
            '<a href="index.html">← 回分析總覽</a>｜模型參數於 Python（scikit-learn，random_state=42）訓練後直接複製到前端',
        )
        inference_path.write_text(txt, encoding="utf-8")
        index_path.write_text(INDEX_HTML, encoding="utf-8")
        print("inference page moved to inference.html; index.html is now the overview homepage")
    else:
        index_path.write_text(INDEX_HTML, encoding="utf-8")
        print("index.html overview written")

    print("done")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FATAL (missing source number): {e}", file=sys.stderr)
        sys.exit(1)
