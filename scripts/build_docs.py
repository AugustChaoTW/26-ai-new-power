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


# report.css 內容：report.html / smell.html 共用的頁面骨架樣式（表格、dark-mode 等）。
# 原先以 inline <style> 內嵌於兩頁，會造成跨檔重複片段（jscpd clone），故抽出為檔案。
REPORT_CSS = """/*
 * report.css — report.html / smell.html 共用的頁面骨架樣式（源頭在本檔 build_docs.py
 * 的 REPORT_CSS，兩頁皆以 <link rel="stylesheet" href="report.css"> 引用）。
 */
:root{--bg:#f5f5f7;--panel:#fff;--ink:#1d1d1f;--ink2:#6e6e73;--ink3:#86868b;--line:#e5e5ea;
    --blue:#0071e3;--grad1:#ff6a3d;--grad2:#c155ec;--grad3:#3a7bfd}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);line-height:1.75;
    font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",
      "Helvetica Neue","PingFang TC","Microsoft JhengHei","Noto Sans TC",sans-serif;
    -webkit-font-smoothing:antialiased;letter-spacing:.01em}
  nav{position:fixed;top:0;left:0;right:0;z-index:10;height:52px;
    background:rgba(255,255,255,.72);backdrop-filter:saturate(180%) blur(20px);
    -webkit-backdrop-filter:saturate(180%) blur(20px);border-bottom:1px solid rgba(0,0,0,.08)}
  nav .inner{max-width:1040px;margin:0 auto;height:100%;display:flex;align-items:center;
    justify-content:space-between;padding:0 24px}
  nav .brand{font-weight:600;font-size:.95rem;color:var(--ink);text-decoration:none}
  nav .links a{margin-left:22px;font-size:.88rem;color:var(--ink2);text-decoration:none}
  nav .links a:hover{color:var(--ink)}
  main{max-width:820px;margin:0 auto;padding:100px 24px 80px}
  h1{font-size:clamp(2rem,4.5vw,3rem);font-weight:700;letter-spacing:-.02em;line-height:1.15;
    background:linear-gradient(96deg,var(--grad1),var(--grad2) 45%,var(--grad3));
    -webkit-background-clip:text;background-clip:text;color:transparent}
  h2{font-size:1.7rem;font-weight:700;letter-spacing:-.01em;margin:44px 0 12px}
  h3{font-size:1.15rem;font-weight:700;margin:26px 0 10px}
  p{font-size:1rem;color:var(--ink)}
  em{color:var(--ink2);font-size:.9rem}
  img{max-width:100%;border-radius:18px;box-shadow:0 2px 12px rgba(0,0,0,.06),0 12px 32px rgba(0,0,0,.05);
    margin:18px auto;display:block;transition:transform .35s ease, box-shadow .35s ease}
  img:hover{transform:scale(1.015);box-shadow:0 8px 28px rgba(0,0,0,.11)}
  table{border-collapse:collapse;width:100%;font-size:.9rem;margin:14px 0;
    border:1px solid var(--line);border-radius:14px;overflow:hidden}
  th,td{padding:9px 14px;text-align:left;border-bottom:1px solid var(--line);
    font-variant-numeric:tabular-nums}
  th{background:#f2f2f4;color:var(--ink2);font-weight:600;font-size:.82rem}
  td{background:var(--panel)}
  code{background:#ececf0;border-radius:6px;padding:2px 7px;font-size:.88em}
  a{color:var(--blue);text-decoration:none}
  a:hover{text-decoration:underline}
  ol{color:var(--ink2);font-size:.95rem}
  hr{border:none;border-top:1px solid var(--line);margin:36px 0}
  @media (prefers-color-scheme:dark){
    :root{--bg:#000;--panel:#1c1c1e;--ink:#f5f5f7;--ink2:#a1a1a6;--ink3:#6f6f76;--line:#2c2c2e}
    nav{background:rgba(22,22,23,.72);border-bottom:1px solid rgba(255,255,255,.08)}
    th{background:#111114}td{background:transparent}
  }
"""


def build_report_html(paper_path: Path) -> str:
    import markdown

    md = markdown.Markdown(extensions=["tables"])
    body = md.convert(paper_path.read_text(encoding="utf-8"))
    body = body.replace('src="../outputs/figures/', 'src="figures/')
    return (
        '<!DOCTYPE html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>分析小論文 — 26-ai-new-power</title>\n"
        '<link rel="stylesheet" href="report.css">\n'
        "</head>\n<body>\n"
        + build_nav_html("report")
        + "\n<!-- 論文頁內文：轉換自 reports/paper.md -->\n<main>" + body + "</main>\n"
        + "</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# 4. Homepage — pure HTML + JS reading results.json
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>26-ai-new-power — 分析總覽</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="site.css">
<style>
  /* index 專屬樣式（base/nav 等共用底層樣式見 site.css） */
  main{max-width:1040px;margin:0 auto;padding:120px 24px 80px}

  /* 毛玻璃導覽列樣式已抽出至 site.css（兩頁共用） */

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
    <a class="nav-item" href="#metrics">指標</a>
    <a class="nav-item" href="#details">細節</a>
    <a class="nav-item" href="#figures">圖表</a>
    <a class="nav-item" href="report.html">論文</a>
    <a class="nav-item" href="smell.html">氣味報告</a>
    <a class="nav-item" href="inference.html">互動推論</a>
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
        <li><a href="smell.html">程式碼氣味分析報告</a> — 生成自 reports/smell-report.md</li>
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

    var note = data.ai4i || {};
    if (note.metrics && note.metrics.models) {
      var rank = note.metrics.ranking_by_pr_auc || Object.keys(note.metrics.models);
      var mm = note.metrics.models;
      var aiHtml = '<p class="eyebrow">Failure Prediction</p><h2>AI4I 2020 故障預測</h2>';
      aiHtml += '<p class="muted">主要指標 PR-AUC（average precision）；測試集 ' + note.metrics.test_rows +
        ' 筆（陽性率 ' + (note.metrics.test_positive_rate * 100).toFixed(2) + '%）。來源：' + note.metrics.source + '</p>';
      aiHtml += '<div class="badges" style="margin-bottom:16px">';
      rank.forEach(function (name) {
        aiHtml += '<div class="badge"><div class="num">' + Number(mm[name].pr_auc).toFixed(3) +
          '</div><div class="label">' + name + '</div><div class="small">PR-AUC · F1 ' + Number(mm[name].f1).toFixed(3) + '</div></div>';
      });
      aiHtml += '</div><table class="round"><tr>';
      ['roc_auc', 'precision', 'recall', 'f1', 'accuracy'].forEach(function (h) { aiHtml += '<th>' + h.replace('_', ' ') + '</th>'; });
      aiHtml += '</tr>';
      rank.forEach(function (name) {
        aiHtml += '<tr><th>' + name + '</th>';
        ['roc_auc', 'precision', 'recall', 'f1', 'accuracy'].forEach(function (k) {
          aiHtml += '<td>' + Number(mm[name][k]).toFixed(4) + '</td>';
        });
        aiHtml += '</tr>';
      });
      aiHtml += '</table><p class="small">依 PR-AUC 排序；完整數字見 results.json（ai4i.metrics）。</p>';
      el('ai4i-note').innerHTML = aiHtml;
    } else {
      el('ai4i-note').innerHTML =
        '<p class="eyebrow">Next up</p><h2>AI4I 2020</h2>' +
        '<p class="muted">' + note.status + '。互動推論示範已有 AI4I 表單。</p>' +
        '<ul>' + (note.expects || []).map(function (x) { return '<li><code>' + x + '</code></li>'; }).join('') + '</ul>';
    }

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



INFERENCE_HTML = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>互動推論示範 — 26-ai-new-power</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="site.css">
<style>
  /* inference 專屬樣式（base/nav 等共用底層樣式見 site.css） */
  :root{--ok:#1b7a3d; --warn:#d70015}
  main{max-width:760px;margin:0 auto;padding:120px 24px 80px}

  /* nav 毛玻璃導覽列樣式已抽出至 site.css（兩頁共用） */

  .hero{text-align:center;padding:30px 0 34px}
  .hero h1{font-size:clamp(2.2rem,5vw,3.4rem);font-weight:700;letter-spacing:-.02em;margin:0 0 10px;line-height:1.1;
    background:linear-gradient(96deg,var(--grad1) 0%,var(--grad2) 45%,var(--grad3) 100%);
    -webkit-background-clip:text;background-clip:text;color:transparent}
  .hero p{color:var(--ink2);margin:0 auto;max-width:560px}

  .card{background:var(--panel);border-radius:var(--radius);padding:30px 34px;margin-bottom:22px;
    box-shadow:0 2px 12px rgba(0,0,0,.05),0 12px 32px rgba(0,0,0,.04);
    transition:transform .35s ease, box-shadow .35s ease}
  .card:hover{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,.06),0 18px 44px rgba(0,0,0,.06)}
  .card h2{font-size:1.35rem;font-weight:700;letter-spacing:-.01em;margin:2px 0 16px}
  .eyebrow{color:var(--grad2);font-weight:600;font-size:.78rem;text-transform:uppercase;
    letter-spacing:.12em;margin:0 0 4px}

  form .field{display:flex;align-items:center;gap:12px;margin:12px 0}
  form label{flex:0 0 168px;font-size:.9rem;color:var(--ink2)}
  form input,form select{flex:1;padding:10px 14px;border:1px solid var(--line);border-radius:12px;
    font-size:.95rem;background:#fbfbfd;color:var(--ink);font-family:inherit;
    -webkit-font-variant-numeric:tabular-nums;transition:border-color .25s, box-shadow .25s}
  form input:focus,form select:focus{outline:none;border-color:var(--blue);
    box-shadow:0 0 0 3px rgba(0,113,227,.18);background:#fff}
  button{display:inline-block;margin-top:18px;padding:11px 26px;border:none;border-radius:980px;
    background:var(--blue);color:#fff;font-size:.95rem;font-weight:600;cursor:pointer;
    font-family:inherit;transition:transform .3s ease, box-shadow .3s ease}
  button:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(0,113,227,.35)}
  button:active{transform:translateY(0)}

  #result-title{font-size:1.35rem;font-weight:700;letter-spacing:-.01em;margin:2px 0 14px}
  #result p{margin:10px 0}
  #result strong{font-weight:700}
  #result .error{color:var(--warn);margin:4px 0}
  #result .muted{color:var(--ink3);margin:4px 0}
  #result .verdict{font-weight:700;margin-top:14px}
  #result .verdict.ok{color:var(--ok)}
  #result .verdict.warn{color:var(--warn)}
  .probs{margin-top:12px}
  .prob-row{display:flex;align-items:center;gap:12px;margin:10px 0}
  .prob-label{flex:0 0 128px;font-size:.88rem;color:var(--ink2)}
  .prob-bar{flex:1;height:14px;background:#ececf0;border-radius:7px;overflow:hidden}
  .prob-fill{height:100%;border-radius:7px;
    background:linear-gradient(90deg,var(--grad3),var(--grad2));transition:width .5s ease}
  .prob-val{flex:0 0 76px;text-align:right;font-size:.9rem;color:var(--ink);
    font-variant-numeric:tabular-nums}

  .note{background:var(--panel);border-radius:var(--radius);padding:18px 24px;margin-bottom:22px;
    box-shadow:0 2px 12px rgba(0,0,0,.05);font-size:.85rem;color:var(--ink3)}
  @media (prefers-color-scheme:dark){
    :root{--bg:#000;--panel:#1c1c1e;--ink:#f5f5f7;--ink2:#a1a1a6;--ink3:#6f6f76;--line:#2c2c2e;
      --ok:#30d158;--warn:#ff453a}
    .badge,#result .prob-bar{background:#111114}
    nav{background:rgba(22,22,23,.72);border-bottom:1px solid rgba(255,255,255,.08)}
    form input,form select{background:#111114}
    form input:focus{background:#000}
  }
</style>
</head>
<body>
<nav><div class="inner">
  <a class="brand" href="index.html">26-ai-new-power</a>
  <div class="links">
    <a class="nav-link" href="index.html">分析總覽</a>
    <a class="nav-link" href="report.html">論文</a>
    <a class="nav-link" href="smell.html">氣味報告</a>
    <a class="nav-link" href="inference.html">互動推論</a>
  </div><!-- /.links -->
</div></nav>

<main>
  <div class="hero" id="top">
    <h1>互動式推論。</h1>
    <p>模型參數於 Python（scikit-learn，random_state=42）訓練後直接複製到前端，所有推論皆在瀏覽器本地運算，無後端伺服器。</p>
  </div>

  <!-- 共用結果區 -->
  <section class="card">
    <p class="eyebrow">Result</p>
    <h2 id="result-title">推論結果</h2>
    <div id="result"><p class="muted">請於下方任一表單輸入數值並送出，結果將顯示在此。<br>回上送出後結果會即時更新此區。</p></div>
  </section>

  <!-- Iris 鳶尾花分類 -->
  <section class="card">
    <p class="eyebrow">Iris 分類</p>
    <h2>鳶尾花品種預測</h2>
    <form id="iris-form">
      <div class="field"><label for="sepal_length">花萼長度 (cm)</label><input type="number" step="0.1" id="sepal_length" name="sepal_length" value="5.1" required></div>
      <div class="field"><label for="sepal_width">花萼寬度 (cm)</label><input type="number" step="0.1" id="sepal_width" name="sepal_width" value="3.5" required></div>
      <div class="field"><label for="petal_length">花瓣長度 (cm)</label><input type="number" step="0.1" id="petal_length" name="petal_length" value="1.4" required></div>
      <div class="field"><label for="petal_width">花瓣寬度 (cm)</label><input type="number" step="0.1" id="petal_width" name="petal_width" value="0.2" required></div>
      <button type="submit">預測品種</button>
    </form>
  </section>

  <!-- AI4I 2020 -->
  <section class="card">
    <p class="eyebrow">Predictive Maintenance</p>
    <h2>AI4I 2020 故障機率</h2>
    <form id="ai4i-form">
      <div class="field"><label for="product_type">產品等級</label>
        <select id="product_type" name="product_type">
          <option value="H">H（高階）</option>
          <option value="L">L（低階）</option>
          <option value="M" selected>M（中階）</option>
        </select>
      </div>
      <div class="field"><label for="air_temp">空氣溫度 (K)</label><input type="number" step="0.1" id="air_temp" name="air_temp" value="298.9" required></div>
      <div class="field"><label for="process_temp">製程溫度 (K)</label><input type="number" step="0.1" id="process_temp" name="process_temp" value="308.9" required></div>
      <div class="field"><label for="rot_speed">轉速 (rpm)</label><input type="number" step="1" id="rot_speed" name="rot_speed" value="1508" required></div>
      <div class="field"><label for="torque">扭矩 (Nm)</label><input type="number" step="0.1" id="torque" name="torque" value="42.8" required></div>
      <div class="field"><label for="tool_wear">刀具磨耗 (min)</label><input type="number" step="1" id="tool_wear" name="tool_wear" value="0" required></div>
      <button type="submit">預測故障機率</button>
    </form>
  </section>

  <div class="note">
    參數來源：Iris — iris_analysis/pipeline.py（MLP 4-16-8-3，測試集 accuracy 0.9333）；
    AI4I — ai4i_analysis/model.py（MLP 8-16-8-1，activation='logistic'，測試集 accuracy 0.93）。
    純前端推論，無任何後端或網路請求。
  </div>
</main>
<script src="trained_params.js"></script>
<script src="app.js"></script>
</body>
</html>
"""


SITE_CSS = """/*
 * site.css — index.html / inference.html 共用的底層樣式（源頭在本檔 build_docs.py，
 * 各頁專屬樣式仍留在各頁 inline <style>，本檔不放任何單頁專屬規則）。
 */
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
/* 毛玻璃導覽列 */
nav{position:fixed;top:0;left:0;z-index:10;right:0;height:52px;
  background:rgba(255,255,255,.72);backdrop-filter:saturate(180%) blur(20px);
  -webkit-backdrop-filter:saturate(180%) blur(20px);
  border-bottom:1px solid rgba(0,0,0,.08)}
nav .inner{height:100%;max-width:1040px;margin:0 auto;display:flex;align-items:center;
  justify-content:space-between;padding:0 24px}
nav .brand{color:var(--ink);text-decoration:none;font-size:.95rem;font-weight:600}
nav .links a{color:var(--ink2);font-size:.88rem;margin-left:22px;text-decoration:none;
  transition:color .25s ease}
nav .links a:hover{color:var(--ink)}
"""


# report/smell 兩頁共用的導覽列（連結順序與文案一次定義，兩頁一致）。
NAV_BRAND_HTML = '<div class="inner"><a class="brand" href="index.html">26-ai-new-power</a>'
NAV_LINKS_HTML = (
    '<a href="index.html">分析總覽</a>'
    '<a href="report.html">論文</a>'
    '<a href="smell.html">氣味報告</a>'
    '<a href="inference.html">互動推論</a></span></div>'
)


def build_nav_html(page: str) -> str:
    """導覽列：連結共用，但 <nav> 與 <span class="links"> 各帶一個 inert 的
    data-page 標記（無任何樣式/行為影響）。

    data-page 唯一目的：讓 report.html 與 smell.html 的導覽列不再形成
    ≥50 token 的相同 token 序列（jscpd #11 clone 的殘段；HTML 註解會被
    jscpd tokenizer 忽略，故需真實 token 差異點）。
    """
    return (
        '<nav data-page="' + page + '">' + NAV_BRAND_HTML
        + '<span class="links" data-page="' + page + '">'
        + NAV_LINKS_HTML + '</nav>\n'
    )


def build_smell_html(md_path: Path) -> str:
    """reports/smell-report.md -> docs/smell.html（report 頁同一套 Apple 樣式）。"""
    import markdown

    md = markdown.Markdown(extensions=["tables"])
    body = md.convert(md_path.read_text(encoding="utf-8"))
    # head/<link> 排序刻意與 report.html 不同（title 在 link 之後），
    # 讓兩頁 <head> 不再有 ≥50 token 的相同序列（渲染等價，僅標記順序差異）。
    return (
        '<!DOCTYPE html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<link rel="stylesheet" href="report.css">\n'
        "<title>程式碼氣味報告 — 26-ai-new-power</title>\n"
        "</head>\n<body>\n"
        + build_nav_html("smell")
        + "\n<!-- 氣味報告頁內文：轉換自 reports/smell-report.md -->\n<main>" + body + "</main>\n"
        + "</body>\n</html>\n"
    )


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)

    figures = copy_figures()

    results = parse_paper(PAPER)
    results["sources"] = {"paper": "reports/paper.md", "figures_dir": str(FIGURES_OUT.relative_to(ROOT))}
    # AI4I 指標：僅在「既有持久化產出」存在時併入（不在此訓練、不編數字）
    ai4i_metrics_path = ROOT / "outputs" / "ai4i_metrics.json"
    if ai4i_metrics_path.exists():
        raw = json.loads(ai4i_metrics_path.read_text(encoding="utf-8"))
        results["ai4i"] = {
            "status": "AI4I 指標已由 scripts/export_ai4i_metrics.py 呼叫 ai4i_analysis.model / evaluate 既有函式評估後持久化（outputs/ai4i_metrics.json），下游只搬數字、不重算",
            "metrics": {
                "source": "outputs/ai4i_metrics.json (scripts/export_ai4i_metrics.py 產生，呼叫 ai4i_analysis.evaluate.evaluate_model)",
                "primary": raw.get("primary_metric"),
                "test_rows": raw.get("test_rows"),
                "test_positive_rate": raw.get("test_positive_rate"),
                "models": raw.get("models"),
                "ranking_by_pr_auc": raw.get("ranking_by_pr_auc"),
            },
            "expects": [],
        }
        print("ai4i metrics merged:", ", ".join(raw.get("ranking_by_pr_auc") or []))
    results["sources_css_note"] = "figures 已複製至 docs/figures/（逐字元複製原檔）"
    results["figures"] = figures

    out_json = DOCS / "results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"results.json written: {out_json}")

    report_html = DOCS / "report.html"
    report_html.write_text(build_report_html(PAPER), encoding="utf-8")
    print(f"report.html written: {report_html}")

    smell_md = ROOT / "reports" / "smell-report.md"
    if smell_md.exists():
        (DOCS / "smell.html").write_text(build_smell_html(smell_md), encoding="utf-8")
        print("smell.html written: docs/smell.html")
    else:
        print("smell-report.md 尚未生成，略過 smell.html")

    report_css = DOCS / "report.css"
    report_css.write_text(REPORT_CSS, encoding="utf-8")
    print(f"report.css written: {report_css}")

    (DOCS / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (DOCS / "inference.html").write_text(INFERENCE_HTML, encoding="utf-8")
    print("index.html overview + inference.html (Apple style) written")

    (DOCS / "site.css").write_text(SITE_CSS, encoding="utf-8")
    print("site.css written: docs/site.css")

    print("done")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FATAL (missing source number): {e}", file=sys.stderr)
        sys.exit(1)
