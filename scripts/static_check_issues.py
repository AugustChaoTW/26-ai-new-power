# -*- coding: utf-8 -*-
"""Generate GitHub tracking issues from static-check outputs.

Reads (written by scripts/run_static_checks.sh):
  outputs/static-checks/eslint.json
  outputs/static-checks/jscpd/report.json
  outputs/static-checks/knip.json

Creates GitHub issues (one per finding group) via `gh`, de-duplicating
against open issues with the same title. Always creates/refreshes a baseline
summary issue. Doesn't modify any source file.

Run:  python scripts/static_check_issues.py            # summary only
      python scripts/static_check_issues.py --create   # + create issues
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "static-checks"
REPO = "AugustChaoTW/26-ai-new-power"
LABEL_BASE = "[Static Check]"


def load(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def collect() -> dict:
    issues: dict = {"eslint": [], "jscpd": [], "knip": []}

    # ── eslint.json → 依 ruleId 分組 ──────────────────────────────
    eslint = load(OUT / "eslint.json", [])
    by_rule: dict = {}
    total_messages = 0
    for file_entry in eslint:
        rel = file_entry.get("filePath", "").split("/26-ai-new-power/")[-1]
        for msg in file_entry.get("messages", []):
            total_messages += 1
            rule = msg.get("ruleId") or "(internal)"
            by_rule.setdefault(rule, []).append(
                {"file": rel, "line": msg.get("line"), "column": msg.get("column"),
                 "message": msg.get("message")}
            )
    issues["eslint"] = [
        {"rule": rule, "count": len(items), "items": items}
        for rule, items in by_rule.items()
    ]
    issues["eslint_total"] = total_messages

    # ── jscpd report.json → 複製片段 ─────────────────────────────
    # jscpd 舊/新版輸出檔名不同（report.json / jscpd-report.json）
    jscpd_path = next(
        (candidate for name in ("report.json", "jscpd-report.json")
         if (candidate := OUT / "jscpd" / name).exists()),
        None,
    )
    jscpd = load(jscpd_path, {}) if jscpd_path else {}
    stats = jscpd.get("statistics", {}).get("total", {})
    issues["jscpd_percentage"] = round(stats.get("percentage", 0), 2)
    issues["jscpd_total_files"] = stats.get("files", 0)
    issues["jscpd_clones"] = [
        {
            "first": f"{c['firstFile']['name']}:{c['firstFile']['start']}-{c['firstFile']['end']}",
            "second": f"{c['secondFile']['name']}:{c['secondFile']['start']}-{c['secondFile']['end']}",
            "tokens": c.get("tokens"), "lines": c.get("lines"),
        }
        for c in jscpd.get("duplicates", [])
    ]

    # ── knip.json → 死碼（files/dependencies/exports）───────────
    knip_payload = load(OUT / "knip.json", {})
    # json reporter 為 per-workspace 陣列：
    # [{"file": "...", "files": [...], "dependencies": [...], "devDependencies": [...], "exports": [...]}, ...]
    workspaces = knip_payload.get("issues", [])
    if workspaces and isinstance(workspaces, dict):
        workspaces = [
            dict(workspaces.get("file", "package.json"),
                 **{k: v for k, v in workspaces.items() if k != "file"})
        ] if False else workspaces  # 兼容舊版 dict 格式
    knip_items = []
    for workspace in workspaces:
        ws_file = workspace.get("file", "(workspace)")
        for category in ("files", "dependencies", "devDependencies", "optionalDependencies",
                         "unlisted", "binaries", "unresolved", "exports", "types"):
            entries = workspace.get(category) or []
            for e in entries:
                knip_items.append({
                    "category": category,
                    "file": ws_file,
                    "detail": (e.get("name") or e.get("file") or json.dumps(e))
                    if isinstance(e, dict) else e,
                })
    issues["knip_items"] = knip_items
    return issues


def summarize(issues: dict) -> None:
    print("───── 靜態檢查結果摘要 ─────")
    print(f"eslint 違規總數   : {issues['eslint_total']}")
    print(f"eslint 規則分組   : {len(issues['eslint'])}")
    for g in issues["eslint"]:
        print(f"  [{g['rule']}] {g['count']} 處")
    print(f"jscpd 複製率      : {issues['jscpd_percentage']}%（{len(issues['jscpd_clones'])} 個重複片段）")
    for c in issues["jscpd_clones"][:10]:
        print(f"  {c['first']}  <=>  {c['second']}  ({c['tokens']} tokens / {c['lines']} 行)")
    print(f"knip 死碼項目     : {len(issues['knip_items'])}")


def gh(*args: str, text: str | None = None) -> str:
    cmd = ["gh", *args]
    if text is not None:
        cmd = ["gh", *args[:-1], *args[-1:], "-F", f"body=@-{text}"]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def existing_issue_titles() -> set:
    out = subprocess.run(
        ["gh", "issue", "list", "-R", REPO, "--state", "all", "--limit", "500",
         "--json", "title", "--jq", ".[].title"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {line.strip() for line in out.splitlines()}


def create_issue(title: str, body: str) -> None:
    subprocess.run(
        ["gh", "issue", "create", "-R", REPO, "--title", title, "--body-file", "-"],
        input=body, text=True, check=True,
    )


def to_issues(issues: dict, create: bool) -> None:
    titles = existing_issue_titles() if create else set()
    created = []

    def emit(title: str, body: str) -> None:
        if title in titles:
            print(f"已存在，略過: {title}")
            return
        if create:
            create_issue(title, body)
            created.append(title)
        else:
            print(f"(dry-run) 將建立: {title}")

    # eslint：每個 ruleId 一張
    for g in issues["eslint"]:
        lines = "\n".join(
            f"- `{i['file']}:{i['line']}:{i['column']}` — {i['message']}"
            for i in g["items"]
        )
        emit(
            f"{LABEL_BASE} eslint「{g['rule']}」共 {g['count']} 處",
            f"## eslint — rule `{g['rule']}`\n\n門檻設定見 `eslint.config.mjs`。違規如下：\n\n{lines}\n",
        )

    # jscpd：一張總表
    if issues["jscpd_clones"]:
        lines = "\n".join(
            f"- {c['first']} ⇄ {c['second']}（{c['tokens']} tokens / {c['lines']} 行）"
            for c in issues["jscpd_clones"]
        )
        emit(
            f"{LABEL_BASE} jscpd 複製率 {issues['jscpd_percentage']}%（{len(issues['jscpd_clones'])} 個片段）",
            f"## jscpd 重複率報告\n\n複製率：{issues['jscpd_percentage']}%，"
            f"共 {issues['jscpd_total_files']} 檔、{len(issues['jscpd_clones'])} 個重複片段。\n\n{lines}\n",
        )

    # knip：一張總表
    if issues["knip_items"]:
        lines = "\n".join(f"- `{i['category']}` — {i['file']}：{i['detail']}" for i in issues["knip_items"])
        emit(
            f"{LABEL_BASE} knip 死碼 {len(issues['knip_items'])} 項",
            f"## knip 死碼報告\n\n{lines}\n",
        )

    # 基線摘要（永遠建立一張，討論用）
    body = (
        "## 靜態檢查基線\n\n"
        f"- eslint 怎麼跑：`npx eslint docs scripts tests`（browser/node 全域分開宣告於 `eslint.config.mjs`）\n"
        f"- eslint 違規：**{issues['eslint_total']}**（"
        + (", ".join(f"{g['rule']}×{g['count']}" for g in issues["eslint"]) or "無")
        + "）\n"
        f"- jscpd 複製率：**{issues['jscpd_percentage']}%**，重複片段 {len(issues['jscpd_clones'])} 個\n"
        f"- knip 死碼：**{len(issues['knip_items'])}** 項\n\n"
        "重跑指令：`bash scripts/run_static_checks.sh`（原檔於 `outputs/static-checks/`）\n"
    )
    emit(f"{LABEL_BASE} 基線（本次檢查摘要）", body)

    (OUT / "issues_summary.json").write_text(
        json.dumps({"issues": issues, "created": created}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\n本次實際建立 issues：{len(created)}")


def main() -> None:
    create = "--create" in sys.argv
    issues = collect()
    summarize(issues)
    to_issues(issues, create)


if __name__ == "__main__":
    main()
