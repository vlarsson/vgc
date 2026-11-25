// Interactive 3D landscape: true essential-manifold cost of one synthetic edge vs. the
// Gauss-Newton models of the scalar residual and of the smooth 2x2 residual.
// Edge = viewgraph_calib.synthetic.synthetic_edge(seed=19), rescaled to units f/f*,
// principal points at the origin. Mirrors scripts/plot_teaser.py.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ---------------------------------------------------------------- math
const F = [
  [-0.02516811506621486, 0.5589624269834456, 0.34908966190529134],
  [-0.645147382276788, 0.07258475844851699, 0.03992088499285716],
  [-0.28831557930025753, -0.20464343318452652, -0.13031424352929344],
];
const NULL_L = [0.3607325301539618, -0.3922311331951237, 0.8461836560943437]; // F^T a = 0
const NULL_R = [0.00714812677260777, 0.5299311715990993, -0.8480105292101325]; // F b = 0

// A = K_j^T F K_i with K = diag(f, f, 1); f_i = x (first camera), f_j = y.
function calibrated(x, y) {
  const si = [x, x, 1], sj = [y, y, 1];
  return F.map((row, r) => row.map((v, c) => v * sj[r] * si[c]));
}
const fro2 = (A) => A.reduce((s, row) => s + row.reduce((t, v) => t + v * v, 0), 0);

// SVD-free scalar residual sqrt(l), identical to the SVD formula for rank-2 A.
function scalarResidual(x, y) {
  const A = calibrated(x, y);
  const a = fro2(A);
  let q = 0;
  for (let r = 0; r < 3; r++)
    for (let s = 0; s < 3; s++) {
      let v = 0;
      for (let k = 0; k < 3; k++) v += A[r][k] * A[s][k];
      q += v * v;
    }
  const b = Math.max(0.5 * (a * a - q), 0);
  return [Math.sqrt(Math.max(0.5 - Math.sqrt(b) / a, 0))];
}

const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
const unit = (v) => { const n = Math.hypot(...v); return v.map((t) => t / n); };
const AXIS_L = unit([NULL_L[1], -NULL_L[0], 0]);
const AXIS_R = unit([NULL_R[1], -NULL_R[0], 0]);

function reduce2x2(x, y) {
  const ah = unit([NULL_L[0] / y, NULL_L[1] / y, NULL_L[2]]);
  const bh = unit([NULL_R[0] / x, NULL_R[1] / x, NULL_R[2]]);
  const Pa = [AXIS_L, cross(ah, AXIS_L)], Pb = [AXIS_R, cross(bh, AXIS_R)];
  const A = calibrated(x, y);
  const C = [[0, 0], [0, 0]];
  for (let r = 0; r < 2; r++)
    for (let c = 0; c < 2; c++)
      for (let k = 0; k < 3; k++)
        for (let l = 0; l < 3; l++) C[r][c] += Pa[r][k] * A[k][l] * Pb[c][l];
  return { C, n: Math.sqrt(fro2(A)) };
}
const C0 = reduce2x2(1.12, 1.12).C;
const ETA = Math.sign(C0[0][0] * C0[1][1] - C0[0][1] * C0[1][0]); // fixed branch

function oursResidual(x, y) {
  const { C, n } = reduce2x2(x, y);
  const s = Math.SQRT2 * n;
  return [(C[0][0] - ETA * C[1][1]) / s, (C[0][1] + ETA * C[1][0]) / s];
}

function jacobian(fun, x, y, eps = 1e-7) {
  const a = fun(x + eps, y), b = fun(x - eps, y), c = fun(x, y + eps), d = fun(x, y - eps);
  return a.map((_, k) => [(a[k] - b[k]) / (2 * eps), (c[k] - d[k]) / (2 * eps)]);
}

// Gauss-Newton model |r + J d|^2 around p.
function gnModel(fun, p) {
  const r = fun(p[0], p[1]), J = jacobian(fun, p[0], p[1]);
  return (x, y) => {
    const dx = x - p[0], dy = y - p[1];
    let s = 0;
    for (let k = 0; k < r.length; k++) { const v = r[k] + J[k][0] * dx + J[k][1] * dy; s += v * v; }
    return s;
  };
}

// Levenberg-Marquardt step d = -(J^T J + lambda I)^{-1} J^T r with a small fixed lambda
// (J^T J is rank one for the scalar residual).
const LAMBDA = 1e-6;
function gnStep(fun, p) {
  const r = fun(p[0], p[1]), J = jacobian(fun, p[0], p[1]);
  let h00 = LAMBDA, h01 = 0, h11 = LAMBDA, g0 = 0, g1 = 0;
  for (let k = 0; k < r.length; k++) {
    h00 += J[k][0] * J[k][0]; h01 += J[k][0] * J[k][1]; h11 += J[k][1] * J[k][1];
    g0 += J[k][0] * r[k]; g1 += J[k][1] * r[k];
  }
  const det = h00 * h11 - h01 * h01;
  return [p[0] - (h11 * g0 - h01 * g1) / det, p[1] - (h00 * g1 - h01 * g0) / det];
}

const TOL = 1e-6, MAX_IT = 25;
const err = (p) => Math.max(Math.abs(p[0] - 1), Math.abs(p[1] - 1));
function gnPath(fun, p0) {
  const path = [p0];
  let p = p0;
  for (let k = 0; k < MAX_IT && err(p) > TOL; k++) {
    p = gnStep(fun, p);
    if (!(p[0] > 0 && p[1] > 0)) break;
    path.push(p);
  }
  return path;
}

// ---------------------------------------------------------------- scene helpers
const LO = 0.75, HI = 1.45, MID = (LO + HI) / 2, HALF = (HI - LO) / 2;
const N = 81;
const HS = 0.85; // world height of the clip level
const toW = (x, y, h) => new THREE.Vector3((x - MID) / HALF, h, -(y - MID) / HALF);

const xs = Array.from({ length: N }, (_, i) => LO + ((HI - LO) * i) / (N - 1));
const trueCost = xs.map((y) => xs.map((x) => scalarResidual(x, y)[0] ** 2));
const ZMAX = (() => {
  const all = trueCost.flat().sort((a, b) => a - b);
  return all[Math.floor(0.92 * all.length)];
})();
const LEVELS = Array.from({ length: 6 }, (_, k) => ((k + 1) / 6) ** 2 * ZMAX);
const height = (z) => (z / ZMAX) * HS;
// finer grid for the 2D inset's filled level bands
const NP = 220;
const pipBands = Array.from({ length: NP }, (_, j) => Array.from({ length: NP }, (_, i) => {
  const x = LO + ((HI - LO) * (i + 0.5)) / NP, y = HI - ((HI - LO) * (j + 0.5)) / NP;
  const z = scalarResidual(x, y)[0] ** 2;
  return LEVELS.filter((L) => z > L).length;
}));

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function gridGeometry() {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * N * 3), 3));
  const idx = [];
  for (let j = 0; j < N - 1; j++)
    for (let i = 0; i < N - 1; i++) {
      const a = j * N + i, b = a + 1, c = a + N, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  g.setIndex(idx);
  return g;
}
function fillGrid(geom, Z) {
  const pos = geom.attributes.position;
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const v = toW(xs[i], xs[j], height(Z[j][i]));
      pos.setXYZ(j * N + i, v.x, v.y, v.z);
    }
  pos.needsUpdate = true;
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
}
// Iso-parameter wire lines over a surface.
function wirePositions(Z, every = 8) {
  const out = [];
  const push = (i, j) => { const v = toW(xs[i], xs[j], height(Z[j][i])); out.push(v.x, v.y, v.z); };
  for (let j = 0; j < N; j += every) for (let i = 0; i < N - 1; i++) { push(i, j); push(i + 1, j); }
  for (let i = 0; i < N; i += every) for (let j = 0; j < N - 1; j++) { push(i, j); push(i, j + 1); }
  return new Float32Array(out);
}
// Marching squares: flat [x1, y1, x2, y2, ...] segments in focal units.
function contourSegments(Z, levels) {
  const out = [];
  const P = (x, y) => out.push(x, y);
  for (const L of levels)
    for (let j = 0; j < N - 1; j++)
      for (let i = 0; i < N - 1; i++) {
        const v = [Z[j][i], Z[j][i + 1], Z[j + 1][i + 1], Z[j + 1][i]];
        const c = [[xs[i], xs[j]], [xs[i + 1], xs[j]], [xs[i + 1], xs[j + 1]], [xs[i], xs[j + 1]]];
        const pts = [];
        for (let e = 0; e < 4; e++) {
          const a = v[e], b = v[(e + 1) % 4];
          if ((a < L) !== (b < L)) {
            const t = (L - a) / (b - a), p = c[e], q = c[(e + 1) % 4];
            pts.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
          }
        }
        for (let k = 0; k + 1 < pts.length; k += 2) { P(...pts[k]); P(...pts[k + 1]); }
      }
  return out;
}
function setLine(obj, arr) {
  obj.geometry.dispose();
  obj.geometry = new THREE.BufferGeometry();
  obj.geometry.setAttribute("position", new THREE.BufferAttribute(arr, 3));
}
function textSprite(text, size = 0.11) {
  const cv = document.createElement("canvas");
  const ctx = cv.getContext("2d");
  const font = "500 44px 'JetBrains Mono', monospace";
  ctx.font = font;
  cv.width = Math.ceil(ctx.measureText(text).width) + 8; cv.height = 56;
  ctx.font = font;
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 4, 30);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
  const s = new THREE.Sprite(mat);
  s.scale.set((size * cv.width) / cv.height, size, 1);
  s.renderOrder = 10;
  return s;
}

// ---------------------------------------------------------------- panel
const clip = [new THREE.Plane(new THREE.Vector3(0, -1, 0), HS)];

class Panel {
  constructor(canvas, fun, colorVar) {
    this.canvas = canvas; this.fun = fun; this.colorVar = colorVar;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
    this.camera.position.set(3.25, 3.0, 4.35);

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 4, 3);
    this.scene.add(sun);

    // true cost
    this.trueGeom = gridGeometry(); fillGrid(this.trueGeom, trueCost);
    this.trueGeom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(N * N * 3), 3));
    this.trueMat = new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.85, transparent: true, opacity: 0.6, depthWrite: false, clippingPlanes: clip });
    this.scene.add(new THREE.Mesh(this.trueGeom, this.trueMat));
    this.trueWireMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.25, clippingPlanes: clip });
    const tw = new THREE.LineSegments(new THREE.BufferGeometry(), this.trueWireMat); setLine(tw, wirePositions(trueCost));
    this.scene.add(tw);

    // GN model
    this.modelGeom = gridGeometry();
    this.modelMat = new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, roughness: 0.6, transparent: true, opacity: 0.45, depthWrite: false, clippingPlanes: clip });
    const modelMesh = new THREE.Mesh(this.modelGeom, this.modelMat);
    modelMesh.renderOrder = 2;
    this.scene.add(modelMesh);
    this.modelWireMat = new THREE.LineBasicMaterial({ clippingPlanes: clip });
    this.modelWire = new THREE.LineSegments(new THREE.BufferGeometry(), this.modelWireMat);
    this.modelWire.renderOrder = 3;
    this.scene.add(this.modelWire);

    // picture-in-picture: 2D level sets, as in the paper figure
    this.pip = document.createElement("canvas");
    this.pip.className = "pip";
    canvas.parentElement.appendChild(this.pip);
    const pick = (ev) => {
      const r = this.pip.getBoundingClientRect();
      const x = LO + ((ev.clientX - r.left) / r.width) * (HI - LO), y = HI - ((ev.clientY - r.top) / r.height) * (HI - LO);
      if (this.onPick && x >= LO && x <= HI && y >= LO && y <= HI) this.onPick([x, y]);
    };
    this.pip.addEventListener("pointerdown", (ev) => { this.pip.setPointerCapture(ev.pointerId); pick(ev); });
    this.pip.addEventListener("pointermove", (ev) => { if (ev.buttons) pick(ev); });

    // frame + ticks
    this.frameMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.7 });
    const fr = [];
    const corner = [[LO, LO], [HI, LO], [HI, HI], [LO, HI]];
    for (let k = 0; k < 4; k++) { const a = toW(...corner[k], 0), b = toW(...corner[(k + 1) % 4], 0); fr.push(a.x, 0, a.z, b.x, 0, b.z); }
    for (const t of [0.8, 1.0, 1.2, 1.4]) {
      const a = toW(t, LO, 0), b = toW(HI, t, 0);
      fr.push(a.x, 0, a.z, a.x, 0, a.z + 0.06, b.x, 0, b.z, b.x + 0.06, 0, b.z);
      const sx = textSprite(t.toFixed(1), 0.09); sx.position.copy(toW(t, LO - 0.06, 0)); this.scene.add(sx);
      const sy = textSprite(t.toFixed(1), 0.09); sy.position.copy(toW(HI + 0.07, t, 0)); this.scene.add(sy);
    }
    const frame = new THREE.LineSegments(new THREE.BufferGeometry(), this.frameMat); setLine(frame, new Float32Array(fr));
    this.scene.add(frame);
    this.labels = [textSprite("f₁/f₁*", 0.11), textSprite("f₂/f₂*", 0.11)];
    this.labels[0].position.copy(toW(MID, LO - 0.15, 0));
    this.labels[1].position.copy(toW(HI + 0.17, MID, 0));
    this.scene.add(...this.labels);
    this.tickSprites = this.scene.children.filter((o) => o.isSprite);

    // solution star
    this.starMat = new THREE.MeshBasicMaterial();
    const star = new THREE.Mesh(new THREE.OctahedronGeometry(0.035), this.starMat);
    star.position.copy(toW(1, 1, 0.01));
    this.scene.add(star);

    // iterate marker
    this.dotMat = new THREE.MeshBasicMaterial();
    this.dot = new THREE.Mesh(new THREE.SphereGeometry(0.045, 24, 16), this.dotMat);
    this.dotTop = new THREE.Mesh(new THREE.SphereGeometry(0.03, 20, 12), this.dotMat);
    this.stemMat = new THREE.LineDashedMaterial({ dashSize: 0.03, gapSize: 0.025 });
    this.stem = new THREE.Line(new THREE.BufferGeometry(), this.stemMat);
    this.handle = new THREE.Mesh(new THREE.SphereGeometry(0.13, 8, 6), new THREE.MeshBasicMaterial({ visible: false }));
    this.scene.add(this.dot, this.dotTop, this.stem, this.handle);

    // GN path
    this.pathMat = new THREE.LineBasicMaterial();
    this.path = new THREE.Line(new THREE.BufferGeometry(), this.pathMat);
    this.pathDots = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 6, sizeAttenuation: false }));
    this.scene.add(this.path, this.pathDots);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.18, 0);
    this.controls.enableZoom = false;
    this.controls.enablePan = false;
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.update();

    this.applyTheme();
  }

  applyTheme() {
    const col = new THREE.Color(css(this.colorVar)), text = new THREE.Color(css("--text")), surf = new THREE.Color(css("--surf-line"));
    const lo = new THREE.Color(css("--surf-lo")), hi = new THREE.Color(css("--surf-hi")), c = new THREE.Color();
    const colors = this.trueGeom.attributes.color;
    for (let j = 0; j < N; j++)
      for (let i = 0; i < N; i++) {
        c.lerpColors(lo, hi, Math.sqrt(Math.min(trueCost[j][i] / ZMAX, 1)));
        colors.setXYZ(j * N + i, c.r, c.g, c.b);
      }
    colors.needsUpdate = true;
    const muted = new THREE.Color(css("--muted"));
    this.trueWireMat.color = surf;
    this.modelMat.color = col; this.modelWireMat.color = col;
    this.dotMat.color = col; this.pathMat.color = col; this.pathDots.material.color = col; this.stemMat.color = col;
    this.frameMat.color = muted; this.starMat.color = text;
    for (const s of this.tickSprites) s.material.color = muted;
    if (this.last) this.drawPip(...this.last);
    this.render();
  }

  drawPip(Z, p, path) {
    this.last = [Z, p, path];
    const dpr = Math.min(window.devicePixelRatio, 2), W = this.pip.clientWidth;
    if (!W) return;
    this.pip.width = this.pip.height = Math.round(W * dpr);
    const ctx = this.pip.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const X = (x) => ((x - LO) / (HI - LO)) * W, Y = (y) => ((HI - y) / (HI - LO)) * W;

    // filled level bands of the true cost
    const lo = new THREE.Color(css("--surf-lo")), hi = new THREE.Color(css("--surf-hi")), c = new THREE.Color();
    const img = new ImageData(NP, NP);
    for (let j = 0; j < NP; j++)
      for (let i = 0; i < NP; i++) {
        c.lerpColors(lo, hi, pipBands[j][i] / LEVELS.length);
        const o = 4 * (j * NP + i);
        img.data[o] = c.r * 255; img.data[o + 1] = c.g * 255; img.data[o + 2] = c.b * 255; img.data[o + 3] = 255;
      }
    const off = new OffscreenCanvas(NP, NP);
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, W, W);

    const col = css(this.colorVar);
    const seg = (arr) => { ctx.beginPath(); for (let k = 0; k < arr.length; k += 4) { ctx.moveTo(X(arr[k]), Y(arr[k + 1])); ctx.lineTo(X(arr[k + 2]), Y(arr[k + 3])); } ctx.stroke(); };
    ctx.lineWidth = 1.3; ctx.strokeStyle = col;
    seg(contourSegments(Z, LEVELS));

    if (path && path.length > 1) {
      ctx.lineWidth = 1.5; ctx.beginPath();
      path.forEach((q, k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(q[0]), Y(q[1])));
      ctx.stroke();
    }
    ctx.fillStyle = css("--text");
    ctx.font = "13px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("★", X(1), Y(1));
    ctx.fillStyle = col; ctx.strokeStyle = css("--surface"); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(X(Math.min(Math.max(p[0], LO), HI)), Y(Math.min(Math.max(p[1], LO), HI)), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }

  setIterate(p, path) {
    const model = gnModel(this.fun, p);
    const Z = xs.map((y) => xs.map((x) => model(x, y)));
    fillGrid(this.modelGeom, Z);
    setLine(this.modelWire, wirePositions(Z));
    this.drawPip(Z, p, path);

    const show = [Math.min(Math.max(p[0], LO), HI), Math.min(Math.max(p[1], LO), HI)];
    const h = Math.min(height(this.fun(...p)[0] ** 2 + (this.fun(...p)[1] ?? 0) ** 2), HS);
    this.dot.position.copy(toW(...show, 0.012));
    this.handle.position.copy(this.dot.position);
    this.dotTop.position.copy(toW(...show, h));
    setLine(this.stem, new Float32Array([...this.dot.position.toArray(), ...this.dotTop.position.toArray()]));
    this.stem.computeLineDistances();

    const pts = (path || []).map((q) => toW(Math.min(Math.max(q[0], LO), HI), Math.min(Math.max(q[1], LO), HI), 0.012).toArray()).flat();
    setLine(this.path, new Float32Array(pts));
    setLine(this.pathDots, new Float32Array(pts));
    this.render();
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // shift the plot left to leave room for the inset in the top-right corner
    this.camera.setViewOffset(w, h, 0.07 * w, 0, w, h);
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() { this.renderer.render(this.scene, this.camera); }
}

// ---------------------------------------------------------------- wiring
const panels = [
  new Panel(document.getElementById("canvas-scalar"), scalarResidual, "--scalar"),
  new Panel(document.getElementById("canvas-ours"), oursResidual, "--ours"),
];
const iterEls = [document.getElementById("iter-scalar"), document.getElementById("iter-ours")];
const START = [1.3, 1.18];
let start = START.slice();
let run = null; // { paths, k, timer }

// synced cameras
let syncing = false;
panels.forEach((p, i) => {
  p.controls.addEventListener("change", () => {
    p.render();
    if (syncing) return;
    syncing = true;
    const o = panels[1 - i];
    o.camera.position.copy(p.camera.position);
    o.controls.target.copy(p.controls.target);
    o.controls.update();
    o.render();
    syncing = false;
  });
});

function fmtErr(e) { return e < TOL ? "&lt;10⁻⁶" : e.toExponential(1).replace("e", "·10^").replace(/\^([-+]?)(\d+)/, (_, s, d) => sup(s + d)); }
function sup(s) { const m = { "-": "⁻", "+": "", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" }; return [...s].map((c) => m[c]).join(""); }

function update() {
  panels.forEach((pn, i) => {
    let p = start, path = null, k = 0, done = false;
    if (run) {
      const P = run.paths[i];
      k = Math.min(run.k, P.length - 1);
      p = P[k]; path = P.slice(0, k + 1);
      done = run.k >= P.length - 1;
    }
    pn.setIterate(p, path);
    const e = err(p);
    iterEls[i].innerHTML = run
      ? `iteration <b>${k}</b> · error ${fmtErr(e)}${done && e < TOL ? ` · <b>converged</b>` : ""}`
      : `start (${p[0].toFixed(2)}, ${p[1].toFixed(2)}) · error ${fmtErr(e)}`;
  });
  drawConv();
}

// convergence mini plot
const conv = document.getElementById("conv-plot");
function drawConv() {
  const W = 260, H = 120, m = { l: 34, r: 8, t: 10, b: 22 };
  const paths = run ? run.paths : [gnPath(scalarResidual, start), gnPath(oursResidual, start)];
  const kmax = Math.max(8, ...paths.map((p) => p.length - 1));
  const X = (k) => m.l + (k / kmax) * (W - m.l - m.r);
  const Y = (e) => m.t + (-Math.log10(Math.max(e, 1e-8)) / 8) * (H - m.t - m.b);
  let s = "";
  for (const p of [0, -2, -4, -6, -8]) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(10 ** p)}" y2="${Y(10 ** p)}" stroke="var(--grid)"/><text x="${m.l - 5}" y="${Y(10 ** p) + 3}" text-anchor="end">1e${p}</text>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">LM iteration</text>`;
  ["--scalar", "--ours"].forEach((cv, i) => {
    const P = paths[i];
    const upto = run ? Math.min(run.k, P.length - 1) : P.length - 1;
    const pts = P.slice(0, upto + 1).map((q, k) => `${X(k).toFixed(1)},${Y(err(q)).toFixed(1)}`).join(" ");
    const op = run ? 1 : 0.35;
    s += `<polyline points="${pts}" fill="none" stroke="var(${cv})" stroke-width="2" opacity="${op}"/>`;
    s += P.slice(0, upto + 1).map((q, k) => `<circle cx="${X(k)}" cy="${Y(err(q))}" r="2.2" fill="var(${cv})" opacity="${op}"/>`).join("");
  });
  conv.innerHTML = s;
}

function stopRun() { if (run) { clearInterval(run.timer); run = null; } }

document.getElementById("gn-run").addEventListener("click", () => {
  stopRun();
  run = { paths: [gnPath(scalarResidual, start), gnPath(oursResidual, start)], k: 0 };
  update();
  run.timer = setInterval(() => {
    run.k++;
    update();
    if (run.paths.every((p) => run.k >= p.length - 1)) clearInterval(run.timer);
  }, 700);
});
document.getElementById("gn-reset").addEventListener("click", () => { stopRun(); start = START.slice(); update(); });

// dragging / click-to-place the iterate
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
function floorPoint(pn, ev) {
  const r = pn.canvas.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, pn.camera);
  return ray;
}
function placeFrom(pn, ev) {
  const hit = new THREE.Vector3();
  if (!floorPoint(pn, ev).ray.intersectPlane(floor, hit)) return;
  const x = MID + hit.x * HALF, y = MID - hit.z * HALF;
  if (x < LO || x > HI || y < LO || y > HI) return;
  stopRun();
  start = [x, y];
  update();
}
panels.forEach((pn) => {
  let dragging = false, down = null;
  // registered after OrbitControls, so use capture to run first
  pn.canvas.addEventListener("pointerdown", (ev) => {
    down = [ev.clientX, ev.clientY];
    if (floorPoint(pn, ev).intersectObject(pn.handle).length) {
      dragging = true;
      pn.controls.enabled = false;
      pn.canvas.classList.add("dragging");
      pn.canvas.setPointerCapture(ev.pointerId);
    }
  }, { capture: true });
  pn.canvas.addEventListener("pointermove", (ev) => { if (dragging) placeFrom(pn, ev); });
  pn.canvas.addEventListener("pointerup", (ev) => {
    if (dragging) {
      dragging = false;
      pn.controls.enabled = true;
      pn.canvas.classList.remove("dragging");
    } else if (down && Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) < 4) {
      placeFrom(pn, ev);
    }
    down = null;
  });
});

panels.forEach((pn) => (pn.onPick = (q) => { stopRun(); start = q; update(); }));
const ro = new ResizeObserver(() => panels.forEach((p) => { p.resize(); if (p.last) p.drawPip(...p.last); }));
panels.forEach((p) => ro.observe(p.canvas));
window.addEventListener("themechange", () => { panels.forEach((p) => p.applyTheme()); drawConv(); });

update();
