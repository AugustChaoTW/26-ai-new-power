/*
 * 互動式推論示範 — 前端推論腳本（純前端、ES5、無後端／無任何網路請求）
 *
 * 說明（README 摘要）：
 *   模型於 Python 端以 scikit-learn 的 MLPClassifier（random_state=42）訓練完成後，
 *   將 scalerMean / scalerScale / coefs / intercepts 直接複製到 trained_params.js
 *   （全域變數 IRIS_PARAMS：4-16-8-3 分類；AI4I_PARAMS：8-16-8-1 故障機率）。
 *   本檔在瀏覽器中以前向傳播重現 predict_proba 的計算，步驟：
 *     1. 標準化：z0[i] = (x[i] - scalerMean[i]) / scalerScale[i]
 *     2. 各隱藏層：a_next[k] = activation(intercepts[l][k] + sum_j a[j] * coefs[l][j][k])
 *        activation 依參數檔：Iris 用 relu；AI4I 用 logistic (sigmoid)
 *     3. 輸出層：Iris 用 softmax 取得各類機率；AI4I 單輸出用 sigmoid 取得故障機率
 *   使用方式：以靜態伺服器或 file:// 開啟 docs/index.html 即可。
 */
(function () {
  'use strict';

  var VALID_TYPES = ['H', 'L', 'M'];  // AI4I 產品類型（對應 one-hot 順序 [Type_H, Type_L, Type_M]）

  /* ---------- 前向傳播 ----------
   * sklearn 慣例：coefs[l][j][k] = 第 l 層單元 j 對第 l+1 層單元 k 的權重
   * （rows = 輸入單元，cols = 輸出單元），故對每個輸出單元 k，
   * 以內層迴圈累加 prev[j] * coefs[l][j][k]。全部使用純量 for 迴圈。
   */
  function layer_forward(prev, coefs, intercepts, act) {
    var nIn = prev.length;
    var nOut = intercepts.length;
    var out = new Array(nOut);
    var k, j, sum;
    for (k = 0; k < nOut; k++) {
      sum = intercepts[k];
      for (j = 0; j < nIn; j++) {
        sum += prev[j] * coefs[j][k];
      }
      out[k] = act ? act(sum) : sum;  // act 只作用於隱藏層；輸出層帶 null
    }
    return out;
  }

  function mlp_forward(params, x) {
    var act = params.activation === 'logistic' ? logistic : relu;
    var a = new Array(x.length);
    var i, l;
    // 標準化輸入
    for (i = 0; i < x.length; i++) {
      a[i] = (x[i] - params.scalerMean[i]) / params.scalerScale[i];
    }
    // 逐層前進；最後一層不做激活（保留原始 z 值）
    var nLayers = params.coefs.length;
    for (l = 0; l < nLayers; l++) {
      a = layer_forward(a, params.coefs[l], params.intercepts[l], l < nLayers - 1 ? act : null);
    }
    return a;  // 輸出層未過激活的 z 向量
  }

  /* ---------- 激活函數 ---------- */
  function relu(z) { return z < 0 ? 0 : z; }
  function logistic(z) { return 1 / (1 + Math.exp(-z)); }

  /* ---------- softmax / sigmoid ---------- */
  function softmax(z) {
    var m = z[0];
    var i, sum, out;
    for (i = 1; i < z.length; i++) {
      if (z[i] > m) { m = z[i]; }
    }
    sum = 0;
    out = new Array(z.length);
    for (i = 0; i < z.length; i++) {
      out[i] = Math.exp(z[i] - m);
      sum += out[i];
    }
    for (i = 0; i < z.length; i++) {
      out[i] /= sum;
    }
    return out;
  }

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  function argmax(a) {
    var best = 0;
    var i;
    for (i = 1; i < a.length; i++) {
      if (a[i] > a[best]) { best = i; }
    }
    return best;
  }

  /* ---------- 預測 ---------- */
  function predict_iris(x) {
    var probs = softmax(mlp_forward(IRIS_PARAMS, x));
    var idx = argmax(probs);
    return { probs: probs, classIndex: idx, className: IRIS_PARAMS.classes[idx] };
  }

  function predict_ai4i(x) {
    var z = mlp_forward(AI4I_PARAMS, x);
    return { p: sigmoid(z[0]) };  // P(故障)
  }

  /* ---------- 畫面渲染 ---------- */
  function probRow(label, p) {
    var pct = Math.round(p * 100);
    return '<div class="prob-row">' +
      '<span class="prob-label">' + label + '</span>' +
      '<span class="prob-bar"><span class="prob-fill" style="display:block;width:' + pct + '%"></span></span>' +
      '<span class="prob-val">' + p.toFixed(4) + '</span>' +
      '</div>';
  }

  function render_iris(x) {
    var r = predict_iris(x);
    var html = '<p>預測類別：<strong>' + r.className + '</strong>（第 ' + (r.classIndex + 1) + ' 類）</p>';
    html += '<div class="probs">';
    r.probs.forEach(function (p, k) {
      html += probRow(IRIS_PARAMS.classes[k] + ' 機率', p);
    });
    html += '</div>';
    resultTitle.textContent = 'Iris 預測結果';
    result.innerHTML = html;
  }

  function render_ai4i(x) {
    var r = predict_ai4i(x);
    var pct = (r.p * 100).toFixed(2);
    var verdictClass = r.p < 0.2 ? 'ok' : 'warn';
    var verdictText = r.p < 0.2
      ? '判定：正常運作（低故障風險）'
      : '判定：高故障風險，建議安排停機檢查';
    var html = '<p>故障機率：<strong>' + pct + '%</strong></p>' +
      '<div class="probs">' + probRow('故障機率', r.p) + '</div>' +
      '<p class="verdict ' + verdictClass + '">' + verdictText + '</p>';
    resultTitle.textContent = 'AI4I 故障預測';
    result.innerHTML = html;
  }

  function showInputError() {
    result.innerHTML = '<p class="error">請填入所有數值</p>';
  }

  /* ---------- 事件繫結 ---------- */
  function readNumber(id) {
    var v = parseFloat(document.getElementById(id).value);
    return isNaN(v) ? null : v;
  }

  function onIrisSubmit(e) {
    e.preventDefault();
    var sl = readNumber('sepal_length');
    var sw = readNumber('sepal_width');
    var pl = readNumber('petal_length');
    var pw = readNumber('petal_width');
    if (sl === null || sw === null || pl === null || pw === null) {
      showInputError();
      return;
    }
    // 特徵順序：sepal_length, sepal_width, petal_length, petal_width（與訓練時 featureNames 一致）
    render_iris([sl, sw, pl, pw]);
  }

  function onAi4iSubmit(e) {
    e.preventDefault();
    var type = document.getElementById('product_type').value;
    if (VALID_TYPES.indexOf(type) === -1) {
      showInputError();
      return;
    }
    var air = readNumber('air_temp');
    var proc = readNumber('process_temp');
    var rpm = readNumber('rot_speed');
    var torque = readNumber('torque');
    var wear = readNumber('tool_wear');
    if (air === null || proc === null || rpm === null || torque === null || wear === null) {
      showInputError();
      return;
    }
    // one-hot（順序 [Type_H, Type_L, Type_M]）：'H' -> [1,0,0]、'L' -> [0,1,0]、'M' -> [0,0,1]
    var onehot = [0, 0, 0];
    onehot[VALID_TYPES.indexOf(type)] = 1;
    // 八維特徵順序（與 model.py FEATURE_COLUMNS 一致）：
    // [air_temp, process_temp, rot_speed, torque, tool_wear, Type_H, Type_L, Type_M]
    render_ai4i([air, proc, rpm, torque, wear, onehot[0], onehot[1], onehot[2]]);
  }

  function wireEvents() {
    resultTitle = document.getElementById('result-title');
    result = document.getElementById('result');
    if (typeof IRIS_PARAMS === 'undefined' || typeof AI4I_PARAMS === 'undefined') {
      result.innerHTML = '<p class="error">找不到 trained_params.js（IRIS_PARAMS / AI4I_PARAMS），請先放置參數檔。</p>';
      return;
    }
    document.getElementById('iris-form').addEventListener('submit', onIrisSubmit);
    document.getElementById('ai4i-form').addEventListener('submit', onAi4iSubmit);
  }

  var resultTitle;
  var result;
  wireEvents();
})();
