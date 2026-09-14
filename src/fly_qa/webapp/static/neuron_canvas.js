// Renders a real sampled subset of the MaleCNS v1.0 connectome, laid out as a
// stylized bilobed brain silhouette (left lobe = R1-R6 input, right lobe = R8
// input, both lobes' shared mass = high-degree "hidden" neurons, a small cluster
// below = the DNp20/DNpe017 descending outputs projecting toward the "neck"),
// and animates ACTUAL simulated activity through it as the rate model settles,
// step by step. Not a decorative animation -- every value shown is a real number
// out of the frozen-weight simulation for the current image. The lobe geometry
// is a stylized fixed shape (not real fly-brain anatomy or actual soma
// positions), chosen only so the network reads visually as "a brain" and stays
// proportionate regardless of the canvas's own box shape.

const BRAIN_ASPECT = 1.6; // width:height of the two-lobe silhouette itself
const ROLE_COLOR = {
  r1r6: "#4ec9ff",
  r8: "#4ec9ff",
  hidden: "#39ff6a",
  dnp20: "#ff8a4e",
  dnpe017: "#ff8a4e",
};
const VERDICT_COLORS = { pass: "#39ff6a", flag: "#ffd23f", fail: "#ff4d4d" };
const STEP_DURATION_MS = 70;

function sampleInEllipse(cx, cy, rx, ry, angleMin, angleMax, rng) {
  const angle = angleMin + rng() * (angleMax - angleMin);
  const radius = Math.sqrt(rng()); // uniform-density disk sampling, not edge-biased
  return { x: cx + Math.cos(angle) * radius * rx, y: cy + Math.sin(angle) * radius * ry };
}

// Deterministic per-node pseudo-random stream so layout doesn't jitter on redraw/resize.
function mulberry32(seed) {
  let a = seed;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class NeuronCanvas {
  constructor(canvasEl) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext("2d");
    this.nodes = []; // {id, role, x, y} -- x/y in canvas CSS pixels
    this.edges = []; // {s, t} node indices
    this.activity = new Map(); // id -> current normalized [0,1] value
    this.verdictTint = null;
    this._playToken = 0;

    this._resize();
    const observer = new ResizeObserver(() => this._resize());
    observer.observe(this.canvas);
    window.addEventListener("resize", () => this._resize());

    requestAnimationFrame(() => this._render());
  }

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = rect.width;
    this.height = rect.height;
    this._layout();
  }

  setGraph(nodes, edges) {
    this.nodes = nodes;
    const idToIndex = new Map(nodes.map((n, i) => [n.id, i]));
    this.edges = edges
      .map((e) => ({ s: idToIndex.get(e.source), t: idToIndex.get(e.target) }))
      .filter((e) => e.s !== undefined && e.t !== undefined);
    this._layout();
  }

  // Fits a fixed-aspect-ratio bilobed "brain" silhouette inside the current
  // canvas box (letterboxed, centered) so the shape never distorts even if
  // the box itself is stretched wider or taller by the surrounding layout.
  _brainGeometry() {
    const pad = 0.92;
    let brainW = this.width * pad;
    let brainH = brainW / BRAIN_ASPECT;
    if (brainH > this.height * pad) {
      brainH = this.height * pad;
      brainW = brainH * BRAIN_ASPECT;
    }
    const cx = this.width / 2;
    const cy = this.height / 2 - brainH * 0.08;
    const lobeRx = brainW * 0.32;
    const lobeRy = brainH * 0.42;
    const lobeOffsetX = brainW * 0.19;
    return {
      cx, cy, lobeRx, lobeRy, lobeOffsetX,
      leftCx: cx - lobeOffsetX,
      rightCx: cx + lobeOffsetX,
      neckCx: cx,
      neckCy: cy + lobeRy * 0.95,
      neckRx: brainW * 0.1,
      neckRy: brainH * 0.13,
    };
  }

  _layout() {
    if (this.nodes.length === 0 || !this.width) return;
    const g = this._brainGeometry();
    this._geometry = g;
    const rng = mulberry32(1337);

    for (const n of this.nodes) {
      let p;
      if (n.role === "r1r6") {
        // left lobe, biased toward the outward-facing (left) half
        p = sampleInEllipse(g.leftCx, g.cy, g.lobeRx, g.lobeRy, Math.PI * 0.5, Math.PI * 1.5, rng);
      } else if (n.role === "r8") {
        // right lobe, biased toward the outward-facing (right) half
        p = sampleInEllipse(g.rightCx, g.cy, g.lobeRx, g.lobeRy, -Math.PI * 0.5, Math.PI * 0.5, rng);
      } else if (n.role === "dnp20" || n.role === "dnpe017") {
        p = sampleInEllipse(g.neckCx, g.neckCy, g.neckRx, g.neckRy, 0, Math.PI * 2, rng);
      } else {
        // hidden: fills the central mass, drawn from whichever lobe is closer
        // to a random point between the two lobe centers (covers the overlap).
        const lobeCx = rng() < 0.5 ? g.leftCx : g.rightCx;
        p = sampleInEllipse(lobeCx, g.cy, g.lobeRx * 1.05, g.lobeRy, 0, Math.PI * 2, rng);
      }
      n.x = p.x;
      n.y = p.y;
    }
  }

  // stepActivity: array of arrays, stepActivity[step][nodeIndex] = real rate value.
  // Node order must match the order this.nodes was set in (the /api/viz-subset response order).
  //
  // Normalized PER ROLE, not by one shared global max: R1-R6 (luminance-driven)
  // activity runs ~100-200x larger in magnitude than R8 (chrominance) activity in
  // practice, since the encoder's confidence-gated hue signal is naturally much
  // smaller than raw luminance. A single global max makes the whole R8 lobe (and
  // most hidden/output neurons) sit under the glow threshold permanently -- not
  // silent, just ~200x dimmer than the R1-R6 side by comparison. Per-role scaling
  // shows each channel's own internal activity pattern instead of only the
  // largest-magnitude one.
  playSequence(stepActivity, verdict) {
    const token = ++this._playToken;
    this.verdictTint = null;
    if (!stepActivity || stepActivity.length === 0) return;

    const roleMax = {};
    this.nodes.forEach((n, i) => {
      const nodeMax = Math.max(...stepActivity.map((row) => row[i] ?? 0));
      roleMax[n.role] = Math.max(roleMax[n.role] || 1e-9, nodeMax);
    });

    let step = 0;
    const advance = () => {
      if (token !== this._playToken) return; // superseded by a newer image
      const row = stepActivity[step];
      this.activity = new Map(
        this.nodes.map((n, i) => [n.id, (row[i] ?? 0) / roleMax[n.role]])
      );
      step++;
      if (step < stepActivity.length) {
        setTimeout(advance, STEP_DURATION_MS);
      } else {
        this.verdictTint = VERDICT_COLORS[verdict] || null;
      }
    };
    advance();
  }

  _renderBrainOutline() {
    if (!this._geometry) return;
    const { ctx } = this;
    const g = this._geometry;
    ctx.strokeStyle = "rgba(57, 255, 106, 0.12)";
    ctx.lineWidth = 1.5;
    for (const cx of [g.leftCx, g.rightCx]) {
      ctx.beginPath();
      ctx.ellipse(cx, g.cy, g.lobeRx, g.lobeRy, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.ellipse(g.neckCx, g.neckCy, g.neckRx, g.neckRy, 0, 0, Math.PI * 2);
    ctx.stroke();
  }

  _render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    this._renderBrainOutline();

    for (const e of this.edges) {
      const a = this.nodes[e.s];
      const b = this.nodes[e.t];
      const activity = Math.max(this.activity.get(a.id) || 0, this.activity.get(b.id) || 0);
      ctx.strokeStyle = `rgba(57, 255, 106, ${0.03 + activity * 0.35})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    for (const n of this.nodes) {
      const activity = this.activity.get(n.id) || 0;
      const color = ROLE_COLOR[n.role] || "#39ff6a";
      const r = 2 + activity * 6;

      if (activity > 0.02) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 5, 0, Math.PI * 2);
        ctx.fillStyle = withAlpha(color, activity * 0.25);
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = withAlpha(color, 0.3 + activity * 0.7);
      ctx.fill();
    }

    if (this.verdictTint) {
      ctx.strokeStyle = withAlpha(this.verdictTint, 0.6);
      ctx.lineWidth = 4;
      ctx.strokeRect(2, 2, this.width - 4, this.height - 4);
    }

    requestAnimationFrame(() => this._render());
  }
}

function withAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
