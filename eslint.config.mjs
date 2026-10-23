// eslint.config.mjs — 靜態檢查（CLI 執行，無編輯器整合）
//
// 兩個檔案範圍分別宣告全域（不要共用同一組）：
//   - 瀏覽器端：docs/**/*.js（app.js / trained_params.js 由 HTML <script> 載入）
//     → browser 全域 + 讀取 trained_params.js 提供的兩個模型參數常數
//   - Node 端：scripts/、tests/ 的 .js/.mjs/.cjs
//     → node 全域（require/process/console/__dirname…）
//     ※ 目前這兩個目錄尚無 JS 檔；範圍先固定，日後放入即自動生效
//
// 門檻（本期要求）：complexity 10、max-depth 3、max-lines-per-function 50、
// cognitive-complexity（sonarjs）15
//
// 說明：docs/index.html、docs/inference.html 的內嵌 <script> 屬 HTML，
// ESLint 不解析 HTML 內嵌腳本，本期不納入（如需納入可改為先抽檔）。

import globals from "globals";
import sonarjs from "eslint-plugin-sonarjs";
import { readFileSync } from "node:fs";

const commonRules = {
  "no-undef": "error",
  complexity: ["error", 10],
  "max-depth": ["error", 3],
  "max-lines-per-function": ["error", 50],
};

const sonarRules = {
  "sonarjs/cognitive-complexity": ["error", 15],
};

const browserParams = (() => {
  // trained_params.js 以 var 宣告 → 解析頂層宣告名稱作為 browser 全域
  const src = readFileSync(new URL("./docs/trained_params.js", import.meta.url), "utf8");
  const names = [...src.matchAll(/^var\s+([A-Z_][A-Z0-9_]*)\s*=/gm)].map((m) => m[1]);
  return Object.fromEntries(names.map((n) => [n, "readonly"]));
})();

export default [
  {
    ignores: [
      "node_modules/**",
      "outputs/**",
      "docs/figures/**",
      "data/**",
      "reports/**",
      "**/__pycache__/**",
      ".pytest_cache/**",
    ],
  },
  // ── 瀏覽器端（browser globals）──
  {
    files: ["docs/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: {
        ...globals.browser,
        ...browserParams,
      },
    },
    plugins: { sonarjs },
    rules: { ...commonRules, ...sonarRules },
  },
  // ── Node 端（node globals）──
  {
    files: [
      "scripts/**/*.js",
      "scripts/**/*.mjs",
      "scripts/**/*.cjs",
      "tests/**/*.js",
      "tests/**/*.mjs",
      "tests/**/*.cjs",
    ],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "commonjs",
      globals: {
        ...globals.node,
      },
    },
    plugins: { sonarjs },
    rules: { ...commonRules, ...sonarRules },
  },
];
