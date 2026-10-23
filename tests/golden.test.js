/*
 * tests/golden.test.js — 前端推論黃金測試（Node，CommonJS）
 *
 * 做法：
 *   1. 建立 document stub（app.js 的 wireEvents 只用到 getElementById /
 *      addEventListener / innerHTML），令人「前端 app.js」可原封不動地在 Node 執行。
 *   2. 依序載入 docs/trained_params.js 與 docs/app.js（與瀏覽器載入順序相同）。
 *   3. 以 docs/app.js 暴露的 window.PREDICT_API（predict_iris / predict_ai4i）
 *      跑 tests/golden.json 裡固定的 3 筆 Iris + 3 筆 AI4I 輸入，
 *      與 Python 端（scripts/export_golden.py）重訓模型的 predict_proba 比對，
 *      機率誤差必須 < 1e-6，且 Iris 類別索引一致。
 *
 * 失敗時的三大可能原因（除錯方向）：
 *   1. 前處理不同步（Python 有標準化，JS 沒做）→ 檢查 mlp_forward 的 scalerMean/scale
 *   2. 特徵順序不同 → 檢查 input 向量與 featureNames / FEATURE_COLUMNS 的對位
 *   3. 係數貼錯或貼漏 → 先跑 scripts/export_golden.py 的參數同步檢查
 *
 * Run:  node tests/golden.test.js   （exit 0 = 全部一致）
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');

/* 最小 DOM stub：讓 app.js wireEvents 不炸（無互動需求） */
function stubElement() {
  return {
    addEventListener: function () {},
    innerHTML: '',
    textContent: '',
    value: '',
    style: {},
  };
}
global.document = {
  getElementById: function () {
    return stubElement();
  },
};
/* 讓 app.js 的 window.PREDICT_API 掛到 Node 全域 */
global.window = global;

/* 與瀏覽器相同的載入順序：參數檔 → 推論程式 */
vm.runInThisContext(
  fs.readFileSync(path.join(ROOT, 'docs', 'trained_params.js'), 'utf8'),
  { filename: 'docs/trained_params.js' }
);
vm.runInThisContext(
  fs.readFileSync(path.join(ROOT, 'docs', 'app.js'), 'utf8'),
  { filename: 'docs/app.js' }
);

const golden = JSON.parse(fs.readFileSync(path.join(__dirname, 'golden.json'), 'utf8'));
const TOL = golden.tolerance;
const API = global.PREDICT_API;

if (!API || typeof API.predict_iris !== 'function') {
  console.error('FATAL: window.PREDICT_API 未暴露（檢查 docs/app.js 結尾的測試介面）。');
  process.exit(2);
}

let cases = 0;
let failures = 0;

function report(name, ok, detail) {
  cases++;
  if (!ok) {
    failures++;
    console.error('FAIL', name, detail);
  } else {
    console.log('PASS', name, detail);
  }
}

function probDiff(got, want) {
  return got.map(function (p, k) {
    return Math.abs(p - want[k]);
  });
}

/* ---- Iris（3 筆）---- */
for (const c of golden.iris.cases) {
  const label = 'iris[' + JSON.stringify(c.input) + ']';
  try {
    const r = API.predict_iris(c.input);
    const diffs = probDiff(r.probs, c.probs);
    const maxDiff = Math.max.apply(null, diffs);
    const ok = maxDiff < TOL && r.classIndex === c.class_index;
    report(label, ok, 'max|Δp|=' + maxDiff.toExponential(2) + ' tol=' + TOL +
      ' class=' + r.className + (ok ? '' : ' want=' + c.class_name));
  } catch (err) {
    report(label, false, String(err));
  }
}

/* ---- AI4I（3 筆）---- */
for (const c of golden.ai4i.cases) {
  const label = 'ai4i[' + c.raw.Type + ' ' + JSON.stringify(c.input) + ']';
  try {
    const r = API.predict_ai4i(c.input);
    const diff = Math.abs(r.p - c.p_failure);
    const ok = diff < TOL;
    report(label, ok, 'p=' + r.p.toFixed(8) + ' |Δp|=' + diff.toExponential(2) +
      ' tol=' + TOL + ' (golden=' + c.p_failure.toFixed(8) + ')');
  } catch (err) {
    report(label, false, String(err));
  }
}

console.log('----');
console.log('golden test: ' + (cases - failures) + '/' + cases + ' passed (tol ' + TOL + ')');
if (failures > 0) {
  console.error('不一致！除錯順序：1) 前處理(標準化) 2) 特徵順序 3) 係數貼附（跑 scripts/export_golden.py 的同步檢查）');
  process.exit(1);
}
process.exit(0);
