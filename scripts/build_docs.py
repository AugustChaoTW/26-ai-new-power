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
        "status": (
            "ai4i_analysis/evaluate.py（issue #9）已產出 8 張圖（outputs/ai4i_figures/，"
            "已複製至 docs/figures/），但 PR-AUC 等指標尚未有持久化檔案，因此本頁不放任何 AI4I 數字"
        ),
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
<title>26-ai-new-power — 分析總覽</title>
<style>
  :root{
    --bg:#f5f5f7; --panel:#ffffff; --ink:#1d1d1f; --ink2:#6e6e73; --ink3:#86868b;
    --line:#e5e5ea; --blue:#0071e3; --radius:22px;
    --grad1:#ff6a3d; --grad2:#c155ec; --grad3:#3a7bfd;
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",
      "Helvetica Neue","PingFang TC","Microsoft JhengHei","Noto Sans TC",sans-serif;
    -webkit-font-smoothing:antialiased;line-height:1.6;letter-spacing:.01em}
  main{max-width:1040px;margin:0 auto;padding:120px 24px 80px}

  /* 毛玻璃導覽列 */
  nav{position:fixed;top:0;left:0;right:0;z-index:10;height:52px;
    background:rgba(255,255,255,.72);backdrop-filter:saturate(180%) blur(20px);
    -webkit-backdrop-filter:saturate(180%) blur(20px);
    border-bottom:1px solid rgba(0,0,0,.08)}
  nav .inner{max-width:1040px;margin:0 auto;height:100%;display:flex;align-items:center;
    justify-content:space-between;padding:0 24px}
  nav .brand{font-weight:600;font-size:.95rem;color:var(--ink);text-decoration:none}
  nav .links a{margin-left:22px;font-size:.88rem;color:var(--ink2);text-decoration:none;
    transition:color .25s ease}
  nav .links a:hover{color:var(--ink)}

  /* Hero */
  .hero{text-align:center;padding:56px 0 40px}
  .hero h1{font-size:clamp(2.6rem,6vw,4.6rem);font-weight:700;letter-spacing:-.02em;
    margin:0 0 14px;line-height:1.08;
    background:linear-gradient(96deg,var(--grad1) 0%,var(--grad2) 45%,var(--grad3) 100%);
    -webkit-background-clip:text;background-clip:text;color:transparent}
  .hero p{max-width:620px;margin:0 auto;color:var(--ink2);font-size:1.12rem}
  .hero .cta{display:inline-flex;gap:14px;margin-top:26px}
  .cta a{padding:10px 22px;border-radius:980px;font-size:.95rem;text-decoration:none;
    transition:transform .3s ease, box-shadow .3s ease}
  .cta a.primary{background:var(--blue);color:#fff}
  .cta a.primary:hover{box-shadow:0 8px 24px rgba(0,113,227,.35);transform:translateY(-1px)}
  .cta a.ghost{border:1px solid var(--line);color:var(--ink);background:transparent}
  .cta a.ghost:hover{border-color:var(--ink3)}

  /* 卡片 */
  .card{background:var(--panel);border-radius:var(--radius);padding:34px 36px;margin-bottom:22px;
    box-shadow:0 2px 12px rgba(0,0,0,.05),0 12px 32px rgba(0,0,0,.04);
    transition:transform .35s ease, box-shadow .35s ease}
  .card:hover{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,.06),0 18px 44px rgba(0,0,0,.06)}
  .card h2{font-size:1.55rem;font-weight:700;letter-spacing:-.01em;margin:0 0 18px}
  .card .eyebrow{color:var(--grad2);font-weight:600;font-size:.8rem;text-transform:uppercase;
    letter-spacing:.12em;margin:0 0 6px}

  /* 指標徽章 */
  .badges{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}
  .badge{background:#fbfbfd;border:1px solid var(--line);border-radius:16px;padding:20px 22px;
    transition:transform .3s ease}
  .badge:hover{transform:scale(1.02)}
  .badge .num{font-size:2.2rem;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.02em;
    background:linear-gradient(120deg,var(--grad3),var(--grad2));
    -webkit-background-clip:text;background-clip:text;color:transparent}
  .badge .label{color:var(--ink);font-size:.9rem;font-weight:600;margin-top:2px}

  /* 表格 */
  table{border-collapse:collapse;width:100%;font-size:.92rem}
  th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line)}
  th{color:var(--ink2);font-weight:600;font-size:.82rem}
  td{font-variant-numeric:tabular-nums}
  tr:last-child th,tr:last-child td{border-bottom:none}
  table.round{border:1px solid var(--line);border-radius:14px;overflow:hidden}

  /* 圖表格 */
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:18px}
  .fig{display:block;background:var(--panel);border-radius:18px;overflow:hidden;
    text-decoration:none;color:var(--ink);box-shadow:0 2px 12px rgba(0,0,0,.05);
    transition:transform .35s ease, box-shadow .35s ease}
  .fig:hover{transform:scale(1.025);box-shadow:0 8px 28px rgba(0,0,0,.10)}
  .fig img{width:100%;display:block}
  .fig figcaption{padding:12px 16px;font-size:.85rem;color:var(--ink2)}

  .muted{color:var(--ink3)}
  .small{font-size:.8rem;color:var(--ink3);margin-top:6px}
  .warn{background:linear-gradient(120deg,#fff8f1,#f3f5ff);border:1px solid var(--line);
    border-radius:var(--radius);padding:24px 30px;margin-bottom:22px}
  .warn h2{font-size:1.3rem;margin:0 0 8px}
  a{color:var(--blue);text-decoration:none}
  a:hover{text-decoration:underline}
  footer{text-align:center;color:var(--ink3);font-size:.8rem;padding:30px 0 10px}
  @media (prefers-color-scheme:dark){
    :root{--bg:#000;--panel:#1c1c1e;--ink:#f5f5f7;--ink2:#a1a1a6;--ink3:#6f6f76;--line:#2c2c2e}
    .badge{background:#111114}
    nav{background:rgba(22,22,23,.72);border-bottom:1px solid rgba(255,255,255,.08)}
    .warn{background:#1c1c1e}
  }
</style>
</head>
<body>
<nav><div class="inner">
  <a class="brand" href="#">26-ai-new-power</a>
  <span class="links">
    <a href="#metrics">指標</a>
    <a href="#details">細節</a>
    <a href="#figures">圖表</a>
    <a href="report.html">論文</a>
    <a href="inference.html">互動推論</a>
  </span>
</div></nav>

<main>
  <div class="hero">
    <h1>從資料到洞察。<br>一步到位。</h1>
    <p>所有數字與圖表皆直接沿用既有產出（reports/paper.md、outputs/figures/），由 scripts/build_docs.py 搬移生成——不重新訓練、不重算、不編造。</p>
    <div class="cta">
      <a class="primary" href="#metrics">查看指標</a>
      <a class="ghost" href="inference.html">互動推論示範</a>
    </div>
  </div>

  <div class="card" id="error-box" style="display:none"></div>

  <div id="content" style="display:none">
    <div class="card" id="metrics">
      <p class="eyebrow">Performance</p>
      <h2 id="m-title">—</h2>
      <div class="badges" id="m-badges"></div>
    </div>

    <div class="card" id="details">
      <p class="eyebrow">Pipeline</p>
      <h2>損失收斂、PCA 與資料規模</h2>
      <div id="misc"></div>
    </div>

    <div class="card">
      <p class="eyebrow">Dataset</p>
      <h2>類別分布</h2>
      <div id="balance"></div>
    </div>

    <div class="card" id="figures">
      <p class="eyebrow">Gallery</p>
      <h2>既有圖表</h2>
      <p class="muted" style="margin-top:-10px">原檔逐字元複製，未重繪。</p>
      <div class="grid" id="figs"></div>
    </div>

    <div class="warn" id="ai4i-note"></div>

    <div class="card">
      <p class="eyebrow">Explore</p>
      <h2>延伸頁面</h2>
      <ul style="margin:0;padding-left:18px">
        <li><a href="report.html">小論文全文（HTML 版）</a> — 轉換自 reports/paper.md</li>
        <li><a href="inference.html">互動推論示範（Iris / AI4I）</a> — 參數自 Python 訓練直接複製</li>
      </ul>
    </div>
  </div>

  <footer>Generated by scripts/build_docs.py · GitHub Pages · main /docs</footer>
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
    el('m-title').textContent = 'Iris 分類指標';
    var badges = '';
    Object.keys(iris.metrics).forEach(function (k) {
      var m = iris.metrics[k];
      badges += '<div class="badge"><div class="num">' + m.value +
        '</div><div class="label">' + k + '</div><div class="small">' + m.source + '</div></div>';
    });
    el('m-badges').innerHTML = badges;

    var misc = '<table class="round">';
    var loss = iris.loss, pca = iris.pca, checks = iris.dataset_checks;
    misc += '<tr><th>訓練損失 初始 / 最終 / 迭代</th><td>' + loss.initial + ' / ' + loss.final + ' / ' + loss.iterations +
      '</td><td class="muted">' + loss.source + '</td></tr>';
    misc += '<tr><th>PCA 前三主成分解釋變異量</th><td>' + pca.explained_variance_pct.join('% / ') + '%（累積 ' + pca.cumulative_pct + '%）</td><td class="muted">' + pca.source + '</td></tr>';
    Object.keys(checks).forEach(function (k) {
      misc += '<tr><th>' + k + '</th><td>' + checks[k] + '</td><td class="muted">' + iris.dataset_checks_source + '</td></tr>';
    });
    misc += '</table>';
    el('misc').innerHTML = misc;

    var bal = iris.class_balance;
    var t = '<table class="round"><tr>';
    bal.header.forEach(function (h) { t += '<th>' + (h || '(空)') + '</th>'; });
    t += '</tr>';
    bal.rows.forEach(function (r) { t += '<tr>'; r.forEach(function (c) { t += '<td>' + c + '</td>'; }); t += '</tr>'; });
    t += '</table><p class="small">' + bal.source + '</p>';
    el('balance').innerHTML = t;

    var figs = '';
    (data.figures || []).forEach(function (f) {
      figs += '<a class="fig" href="' + f.file + '" target="_blank"><img src="' + f.file + '" alt="' + f.file + '">' +
        '<figcaption>' + f.file + '<br><span class="small">複製自 ' + f.from + '</span></figcaption></a>';
    });
    el('figs').innerHTML = figs || '<p class="muted">無圖表。</p>';

    el('ai4i-note').innerHTML =
      '<p class="eyebrow">Next up</p><h2>AI4I 2020</h2>' +
      '<p class="muted">' + data.ai4i.status + '。互動推論示範已有 AI4I 表單。</p>' +
      '<ul>' + (data.ai4i.expects || []).map(function (x) { return '<li><code>' + x + '</code></li>'; }).join('') + '</ul>';

    el('content').style.display = '';
  }).catch(function (e) {
    box.style.display = '';
    box.innerHTML = '<strong>無法載入 results.json：' + e.message + '</strong><br>' +
      'file:// 直開無法 fetch JSON，請以靜態伺服器開啟：<code>python -m http.server 8000</code>';
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
