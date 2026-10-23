#!/usr/bin/env bash
# scripts/run_static_checks.sh — 一鍵跑完所有靜態檢查（先求跑得動、跑得出數字）
#
#   bash scripts/run_static_checks.sh            # 執行檢查 + 摘要
#   bash scripts/run_static_checks.sh --create   # 執行檢查 + 摘要 + 建立 GitHub issues
#
# 工具：eslint（browser/node 雙範圍全域，含 sonarjs 門檻）、jscpd（重複率）、knip（死碼）
# 產出：outputs/static-checks/{eslint.json, jscpd/report.json, knip.json}
# 任一工具回報違規皆不會中止後續工具；最後回傳整體是否有違規（0=乾淨）。

set -u
cd "$(dirname "$0")/.."
OUT=outputs/static-checks
EXIT_CODE=0

mkdir -p "$OUT"

# ESLint（10）對「只有 .py、沒有任何 .js 的目錄」的 glob 會直接報錯，
# 因此先列出實際存在的 JS 檔再送入 eslint；範圍分組（browser/node）仍由 eslint.config.mjs 决定。
JS_FILES=$(find docs scripts tests -type f \
             \( -name '*.js' -o -name '*.mjs' -o -name '*.cjs' \) \
             -not -path '*/node_modules/*' -not -path '*/__pycache__/*' | sort)
echo "JavaScript 檔案清單："
echo "${JS_FILES:-(無)}"

echo
echo "════════ [1/4] eslint（docs:browser＋scripts,tests:node 分範圍全域）════════"
if [ -n "$JS_FILES" ]; then
  # shellcheck disable=SC2086
  npx eslint $JS_FILES --output-file "$OUT/eslint.json" --format json >/dev/null 2>"$OUT/eslint.stderr"
  RC=$?
  if [ $RC -eq 0 ]; then
    echo "eslint: 0 違規 ✔"
  else
    echo "eslint: 發現違規（exit $RC），詳見 $OUT/eslint.json"
    EXIT_CODE=$RC
  fi
else
  echo "eslint: 找不到任何 JS 檔，略過"
fi

echo
echo "════════ [2/4] jscpd（重複率，min-tokens=50）════════"
npx jscpd --min-tokens 50 --reporters json,consoleFull --output "$OUT/jscpd" docs scripts tests || true

echo
echo "════════ [3/4] knip（死碼）════════"
npx knip --config knip.json --reporter json --no-exit-code > "$OUT/knip.json" 2>"$OUT/knip.stderr"
echo "knip 輸出：$OUT/knip.json（stderr：$OUT/knip.stderr）"

echo
echo "════════ [4/4] 摘要 + issues ════════"
if [ "${1:-}" = "--create" ]; then
  python scripts/static_check_issues.py --create || EXIT_CODE=$?
else
  python scripts/static_check_issues.py || EXIT_CODE=$?
fi

echo
echo "完成。原始輸出於 $OUT/"
exit $EXIT_CODE
