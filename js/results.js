// Accuracy-vs-runtime scatter + per-dataset table (numbers from paper/tables/main_results_runtime.tex).
(function () {
  const METHODS = [
    { key: "colmap", name: "COLMAP VGC", color: "#c9ced8" },
    { key: "sweeney", name: "Sweeney et al.", color: "#9d86d6", url: "https://openaccess.thecvf.com/content_iccv_2015/html/Sweeney_Optimizing_the_Viewing_ICCV_2015_paper.html" },
    { key: "geppert", name: "Geppert et al.", color: "#4fbfa0", url: "https://openaccess.thecvf.com/content/CVPR2021/html/Geppert_Privacy_Preserving_Localization_and_Mapping_From_Uncalibrated_Cameras_CVPR_2021_paper.html" },
    { key: "fetzer", name: "Fetzer et al.", color: "#d6b44f", url: "https://openaccess.thecvf.com/content_WACV_2020/html/Fetzer_Stable_Intrinsic_Auto-Calibration_from_Fundamental_Matrices_of_Devices_with_Uncorrelated_WACV_2020_paper.html" },
    { key: "svd", name: "Ess. manifold (SVD)", color: "var(--scalar)", url: "https://openaccess.thecvf.com/content/WACV2026/html/Solonets_Global_Focal_and_Radial_Distortion_Averaging_from_Radial_Fundamental_Matrices_WACV_2026_paper.html" },
    { key: "ours", name: "Ours", color: "var(--ours)" },
  ];
  // [median %, AUC@5, AUC@20, time s]
  const DATA = {
    "ETH3D": {
      colmap: [0.71, 0.632, 0.751, 0.22], sweeney: [0.61, 0.752, 0.856, 0.05], geppert: [0.35, 0.855, 0.943, 0.05],
      fetzer: [0.68, 0.742, 0.905, 0.04], svd: [0.24, 0.886, 0.964, 0.10], ours: [0.24, 0.894, 0.966, 0.04],
    },
    "BlendedMVS": {
      colmap: [4.90, 0.383, 0.541, 0.52], sweeney: [99.98, 0.242, 0.286, 0.39], geppert: [0.63, 0.732, 0.860, 0.53],
      fetzer: [1.10, 0.646, 0.839, 0.40], svd: [0.56, 0.762, 0.887, 1.08], ours: [0.56, 0.769, 0.896, 0.43],
    },
    "IMC2023": {
      colmap: [3.83, 0.340, 0.647, 2.76], sweeney: [100.00, 0.015, 0.020, 3.47], geppert: [1.27, 0.596, 0.807, 6.07],
      fetzer: [1.85, 0.524, 0.784, 4.78], svd: [1.24, 0.608, 0.818, 15.47], ours: [1.22, 0.611, 0.818, 5.37],
    },
    "IMC2024": {
      colmap: [8.96, 0.225, 0.468, 1.23], sweeney: [99.99, 0.181, 0.282, 0.75], geppert: [2.65, 0.423, 0.694, 0.90],
      fetzer: [3.84, 0.335, 0.643, 0.38], svd: [2.46, 0.452, 0.738, 1.59], ours: [2.63, 0.435, 0.722, 0.46],
    },
  };
  const METRICS = [
    { label: "AUC @5%", idx: 1, higher: true },
    { label: "AUC @20%", idx: 2, higher: true },
    { label: "Median error", idx: 0, higher: false },
  ];
  let ds = "ETH3D", metric = METRICS[0];

  const svg = document.getElementById("scatter");
  const tip = document.getElementById("scatter-tip");
  const table = document.getElementById("results-table");
  const NS = "http://www.w3.org/2000/svg";

  function tabs(el, items, isSel, onPick) {
    el.innerHTML = "";
    items.forEach((it) => {
      const b = document.createElement("button");
      b.type = "button";
      b.role = "tab";
      b.textContent = it.label;
      b.setAttribute("aria-selected", isSel(it));
      b.onclick = () => { onPick(it); render(); };
      el.appendChild(b);
    });
  }

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    parent.appendChild(e);
    return e;
  }

  function niceLogTicks(lo, hi) {
    const out = [];
    for (let p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++)
      for (const m of [1, 2, 5]) { const v = m * 10 ** p; if (v >= lo && v <= hi) out.push(v); }
    return out;
  }
  const fmt = (v) => (v >= 10 ? v.toFixed(0) : v >= 1 ? +v.toFixed(1) + "" : +v.toPrecision(2) + "");

  function drawScatter() {
    const W = svg.clientWidth, H = svg.clientHeight;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = "";
    const m = { l: 56, r: 24, t: 18, b: 44 };
    const rows = METHODS.map((mt) => ({ mt, v: DATA[ds][mt.key] }));
    const xs = rows.map((r) => r.v[3]);
    const x0 = Math.min(...xs) / 1.6, x1 = Math.max(...xs) * 1.6;
    const X = (v) => m.l + (Math.log(v / x0) / Math.log(x1 / x0)) * (W - m.l - m.r);

    let Y, yTicks;
    if (metric.higher) {
      const ys = rows.map((r) => r.v[metric.idx]);
      const lo = Math.max(0, Math.floor((Math.min(...ys) - 0.05) * 10) / 10), hi = Math.min(1, Math.ceil((Math.max(...ys) + 0.03) * 10) / 10);
      Y = (v) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
      yTicks = [];
      const step = hi - lo > 0.5 ? 0.2 : 0.1;
      for (let v = lo; v <= hi + 1e-9; v += step) yTicks.push(+v.toFixed(2));
    } else {
      const ys = rows.map((r) => r.v[0]);
      const lo = Math.min(...ys) / 1.5, hi = Math.max(...ys) * 1.5;
      Y = (v) => m.t + (Math.log(v / lo) / Math.log(hi / lo)) * (H - m.t - m.b); // lower error = higher on plot
      yTicks = niceLogTicks(lo, hi);
    }

    const g = el("g", {}, svg);
    for (const t of yTicks) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: "var(--grid)" }, g);
      el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, g).textContent = metric.higher ? t.toFixed(1) : fmt(t) + "%";
    }
    for (const t of niceLogTicks(x0, x1)) {
      el("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, stroke: "var(--grid)", "stroke-dasharray": "2 4" }, g);
      el("text", { x: X(t), y: H - m.b + 18, "text-anchor": "middle" }, g).textContent = fmt(t) + " s";
    }
    el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, g).textContent = "calibration time per scene (log) →  faster is left";
    const yl = el("text", { x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }, g);
    yl.textContent = metric.higher ? `${metric.label}  ↑` : "median error (log)  ↑ better";

    for (const r of rows) {
      const cx = X(r.v[3]), cy = Y(r.v[metric.idx]);
      const isOurs = r.mt.key === "ours";
      const c = el("circle", { cx, cy, r: isOurs ? 9 : 7, fill: r.mt.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
      if (isOurs) el("circle", { cx, cy, r: 14, fill: "none", stroke: r.mt.color, "stroke-opacity": 0.4, "stroke-width": 1.5 }, svg);
      if (isOurs || r.mt.key === "svd") {
        const lab = el("text", { x: cx + (isOurs ? 18 : 13), y: cy + 4, class: "lbl" }, svg);
        lab.textContent = isOurs ? "Ours" : "SVD";
        lab.style.fill = r.mt.color;
      }
      const hit = el("circle", { cx, cy, r: 16, fill: "transparent", style: "cursor:pointer" }, svg);
      hit.addEventListener("mouseenter", () => showTip(r, cx, cy));
      hit.addEventListener("mouseleave", () => (tip.hidden = true));
      c.style.pointerEvents = "none";
    }
  }

  function showTip(r, cx, cy) {
    const v = r.v;
    tip.innerHTML = `<b style="color:${r.mt.color}">${r.mt.name}</b><br>median ${v[0].toFixed(2)}%<br>AUC@5 ${v[1].toFixed(3)} · AUC@20 ${v[2].toFixed(3)}<br>time ${v[3].toFixed(2)} s`;
    tip.hidden = false;
    const w = tip.offsetWidth, W = svg.clientWidth;
    tip.style.left = Math.min(Math.max(8, cx + 8 + 14), W - w) + "px";
    tip.style.top = cy - 10 + "px";
  }

  function rank(col, higher) {
    const vals = METHODS.map((mt) => DATA[ds][mt.key][col]);
    const uniq = [...new Set(vals)].sort((a, b) => (higher ? b - a : a - b));
    return (v) => (v === uniq[0] ? "b" : v === uniq[1] ? "u" : "");
  }

  function drawTable() {
    const cols = [
      { idx: 0, label: "med. %", higher: false, d: 2 },
      { idx: 1, label: "AUC @5%", higher: true, d: 3 },
      { idx: 2, label: "AUC @20%", higher: true, d: 3 },
      { idx: 3, label: "time s", higher: false, d: 2 },
    ];
    const ranks = cols.map((c) => rank(c.idx, c.higher));
    let h = `<thead><tr><th>Method</th>${cols.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead><tbody>`;
    for (const mt of METHODS) {
      const v = DATA[ds][mt.key];
      h += `<tr class="${mt.key === "ours" ? "ours" : ""}"><td><span class="dot" style="background:${mt.color}"></span>${mt.url ? `<a href="${mt.url}">${mt.name}</a>` : mt.name}</td>`;
      cols.forEach((c, i) => {
        const s = v[c.idx].toFixed(c.d), r = ranks[i](v[c.idx]);
        h += `<td>${r ? `<${r}>${s}</${r}>` : s}</td>`;
      });
      h += "</tr>";
    }
    table.innerHTML = h + "</tbody>";
  }

  function render() {
    tabs(document.getElementById("ds-tabs"), Object.keys(DATA).map((k) => ({ label: k })), (it) => it.label === ds, (it) => (ds = it.label));
    tabs(document.getElementById("metric-tabs"), METRICS, (it) => it === metric, (it) => (metric = it));
    drawScatter();
    drawTable();
  }

  render();
  window.addEventListener("resize", drawScatter);
})();
