// Floor tracking for the web AR mode: where the phone is and how it is turned, from the camera
// image, relative to the floor. What WebXR/ARKit would give, done by the page itself.
// The pixel work runs in the HA++ module web/track.ha (WebAssembly); this file keeps the map of
// floor points and decides what to call.
//
// World frame (as in ar.js): x east, y north, z up, metres. The floor is z = -height (the phone's
// height above it when tracking starts); the camera starts at the origin. Camera frame: x right,
// y down, z forward. C: camera -> world rotation (row-major 3x3); T: the camera's position.
//
// Each frame:
//   1. predict: the rotation from the gyroscope (how it changed since the last frame), the
//      position from the last frames. A camera image reaches the page later than the motion
//      sensor readings (about 50-100 ms on phones), so the gyroscope is read at the frame's time
//      minus that delay, which is measured on the fly: it is the shift that makes the gyroscope's
//      turning match the turning seen in the image;
//   2. follow the floor points into the new frame (optical flow), and back again as a check. The
//      whole image's shift is measured first on a small blurred copy (like PTAM's "small blurry
//      image"): if the gyroscope's prediction is off (a wrong delay, a sensor glitch), the image's
//      own shift corrects the starting points of the optical flow;
//   3. solve for the pose that puts the known floor points where they are seen (robust, pulled a
//      little toward the gyroscope's rotation);
//   4. drop the points that don't fit (not on the floor, or followed wrongly); add points where
//      the floor has none: each new point is where its pixel's ray meets the floor.
// The floor height sets the scale, and because every point lies on the floor, the depth of a new
// point is known at once: no second view is needed to start.
//
// What is drawn: the page draws this very camera frame and the scene with this frame's pose, so
// the two always match, with no timing to guess (ar.js). The pose is not smoothed: it is the one
// that puts this frame's floor points where they are seen. Measured on simulated walks, any
// smoothing lags behind the hand's own shake and moves the scene against the image more than the
// pose's own noise does (under 0.1 pixel from frame to frame).

const LEVELS = 4;
const MAX_POINTS = 160;
const TARGET = 100;       // look for new points when fewer are followed
const MIN_POINTS = 10;    // fewer inliers than this: tracking is lost
const CELL = 16;          // grid for new points, in tracking-image pixels
const BORDER = 8;
const INLIER_PX = 2.0;    // reprojection error of a good point, tracking-image pixels
const FB_PX = 0.7;        // forward-backward disagreement allowed
const MAX_ERR = 28;       // grey levels (RMS) left between the two windows
const MIN_SCORE = 10;     // Shi-Tomasi score (mean squared gradient) of a new point
const MAX_DIST = 6;       // metres: new floor points further away than this are too imprecise
const MIN_DOWN = 0.2;     // new points at least ~11.5 degrees below the horizon
const PRIOR = 12;         // weight of the gyroscope's rotation, in points, when it agrees with the image
const WEAK_PRIOR = 0.5;   // its weight otherwise
const AGREE = 0.6 * Math.PI / 180;   // radians
const SHIFT_LEVEL = 2;    // pyramid level for the whole-image shift
const SHIFT_RANGE = 7;    // pixels searched around the prediction at that level (28 at full size)
const LAG_MAX = 200;      // ms: the camera delays searched
const LAG_STEP = 5;
const HISTORY_MS = 2500;  // gyroscope readings and tracked frames kept for measuring the delay

const mul = (a, b) => {
  const r = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  }
  return r;
};
const transpose = (a) => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
const mulv = (a, v) => [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
                        a[6] * v[0] + a[7] * v[1] + a[8] * v[2]];

// Rotation vector (axis * angle) of a rotation matrix.
export function rotvec(m) {
  const c = Math.max(-1, Math.min(1, (m[0] + m[4] + m[8] - 1) / 2));
  const a = Math.acos(c);
  const v = [m[7] - m[5], m[2] - m[6], m[3] - m[1]];
  const k = a < 1e-6 ? 0.5 : a / (2 * Math.sin(a));
  return v.map((x) => x * k);
}

// Nearest rotation (Gram-Schmidt on the columns), against rounding creeping in frame after frame.
export function orthonormalize(m) {
  let a = [m[0], m[3], m[6]], b = [m[1], m[4], m[7]];
  const na = Math.hypot(...a);
  a = a.map((v) => v / na);
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  b = b.map((v, i) => v - d * a[i]);
  const nb = Math.hypot(...b);
  b = b.map((v) => v / nb);
  const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  return [a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]];
}

export class FloorTracker {
  constructor(instance) {
    this.ex = instance.exports;
    this.memory = this.ex.memory;
    this.base = Math.ceil(Number(this.ex.__heap_base.value) / 16) * 16;
    this.w = 0;
    this.h = 0;
    this.px = new Float32Array(2 * MAX_POINTS);       // each point's pixel in the last frame
    this.wx = new Float64Array(3 * MAX_POINTS);       // and its place on the floor
    this.bad = new Uint8Array(MAX_POINTS);            // frames in a row it didn't fit
    this.reset();
  }

  static async create(source) {
    let bytes = source;
    if (typeof source === "string" || source instanceof URL) bytes = await (await fetch(source)).arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new FloorTracker(instance);
  }

  // Start again: the next frame sets the world frame (camera at the origin).
  reset() {
    this.ready = false;
    this.n = 0;
    this.C = null;
    this.T = [0, 0, 0];
    this.v = [0, 0, 0];
    this.prevTime = 0;
    this.lostFrames = 0;
    this.inliers = 0;
    this.gyro = [];                   // { t, C }: camera -> world rotation from the motion sensors
    this.seen = [];                   // { t, C }: rotation of the last tracked frames, from the image
    this.lag = 60;                    // ms between a frame's time and the moment it shows
    this.lags = [];
    this.lagCheck = 0;

  }

  // The gyroscope's camera rotation at time t (ms), between the two nearest readings.
  gyroAt(t) {
    const g = this.gyro;
    if (!g.length) return null;
    if (t <= g[0].t) return g[0].C;
    if (t >= g[g.length - 1].t) return g[g.length - 1].C;
    let lo = 0, hi = g.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (g[mid].t <= t) lo = mid; else hi = mid;
    }
    const a = g[lo], b = g[hi];
    const k = (t - a.t) / Math.max(1e-6, b.t - a.t);
    return orthonormalize(a.C.map((x, i) => x + k * (b.C[i] - x)));
  }

  // The camera delay: the shift that makes the gyroscope's turning between tracked frames match
  // the turning seen in the image. Needs some turning to tell; until then the guess stays.
  measureLag() {
    const s = this.seen;
    if (s.length < 20) return;
    const seenTurns = [];
    let total = 0;
    for (let k = 1; k < s.length; k++) {
      const r = rotvec(mul(transpose(s[k - 1].C), s[k].C));
      seenTurns.push(r);
      total += Math.hypot(...r);
    }
    if (total < 0.35) return;         // less than 20 degrees of turning in the window
    let best = this.lag, bestCost = Infinity;
    const costs = [];
    for (let lag = 0; lag <= LAG_MAX; lag += LAG_STEP) {
      let cost = 0;
      let prev = this.gyroAt(s[0].t - lag);
      for (let k = 1; k < s.length; k++) {
        const cur = this.gyroAt(s[k].t - lag);
        const r = rotvec(mul(transpose(prev), cur));
        const e = seenTurns[k - 1];
        cost += (r[0] - e[0]) ** 2 + (r[1] - e[1]) ** 2 + (r[2] - e[2]) ** 2;
        prev = cur;
      }
      costs.push(cost);
      if (cost < bestCost) { bestCost = cost; best = lag; }
    }
    // keep it only if the minimum is clear: the cost at least doubles 40 ms away
    const i = best / LAG_STEP, j = Math.min(costs.length - 1, i + 8), h = Math.max(0, i - 8);
    if (costs[j] > 2 * bestCost && costs[h] > 2 * bestCost) {
      this.lags.push(best);            // the median of the last measurements: one odd window can't move it
      if (this.lags.length > 7) this.lags.shift();
      const sorted = this.lags.slice().sort((a, b) => a - b);
      this.lag = sorted[sorted.length >> 1];
    }
  }

  alloc(w, h) {
    const at = {};
    let o = this.base;
    const take = (name, bytes) => { at[name] = o; o += Math.ceil(bytes / 16) * 16; };
    const px = w * h;
    const pyr = this.ex.level_offset(w, h, LEVELS);
    const cells = Math.floor(w / CELL) * Math.floor(h / CELL);
    take("rgba", 4 * px); take("pyrA", pyr); take("pyrB", pyr);
    take("sxx", 4 * px); take("sxy", 4 * px); take("syy", 4 * px); take("tmp", 4 * px);
    take("skip", cells); take("cand", 12 * cells);
    take("src", 8 * MAX_POINTS); take("dst", 8 * MAX_POINTS); take("back", 8 * MAX_POINTS);
    take("st1", MAX_POINTS); take("st2", MAX_POINTS); take("err1", 4 * MAX_POINTS); take("err2", 4 * MAX_POINTS);
    take("obs", 8 * MAX_POINTS); take("xyz", 12 * MAX_POINTS); take("wts", 4 * MAX_POINTS);
    take("resid", 4 * MAX_POINTS); take("pose", 48); take("prm", 48);
    const sw = w >> SHIFT_LEVEL, sh = h >> SHIFT_LEVEL;
    take("hpA", 4 * sw * sh); take("hpB", 4 * sw * sh); take("shift", 16);
    const have = this.memory.buffer.byteLength;
    if (o > have) this.memory.grow(Math.ceil((o - have) / 65536));
    const buf = this.memory.buffer;
    const f32 = (name, count) => new Float32Array(buf, at[name], count);
    const u8 = (name, count) => new Uint8Array(buf, at[name], count);
    this.mem = {
      rgba: u8("rgba", 4 * px), skip: u8("skip", cells), cand: f32("cand", 3 * cells),
      src: f32("src", 2 * MAX_POINTS), dst: f32("dst", 2 * MAX_POINTS), back: f32("back", 2 * MAX_POINTS),
      st1: u8("st1", MAX_POINTS), st2: u8("st2", MAX_POINTS), err1: f32("err1", MAX_POINTS),
      obs: f32("obs", 2 * MAX_POINTS), xyz: f32("xyz", 3 * MAX_POINTS), wts: f32("wts", MAX_POINTS),
      resid: f32("resid", MAX_POINTS), pose: f32("pose", 12), prm: f32("prm", 12), shift: f32("shift", 4),
      prmU: new Uint32Array(buf, at.prm, 12),
    };
    this.at = at;
    this.w = w;
    this.h = h;
    this.cw = Math.floor(w / CELL);
    this.ch = Math.floor(h / CELL);
    this.prev = at.pyrA;
    this.cur = at.pyrB;
    this.hpPrev = at.hpA;
    this.hpCur = at.hpB;
    this.sw = sw;
    this.sh = sh;
  }

  // One camera frame. rgba: w x h RGBA pixels; f: focal length in pixels of this image; time: the
  // frame's time (ms, the clock of the gyroscope readings); gyro: the motion sensor readings that
  // arrived since the last frame, [{ t, C }] with C the camera -> world rotation; height: the
  // phone's height above the floor when tracking starts (metres). Returns the pose for this frame.
  frame(rgba, w, h, f, time, gyro, height = 1.35) {
    const t0 = performance.now();
    for (const g of gyro) this.gyro.push(g);
    while (this.gyro.length > 2 && this.gyro[0].t < time - HISTORY_MS) this.gyro.shift();
    const Cg = this.gyroAt(time - this.lag);
    if (!Cg) return { state: "waiting", ok: false, C: null, T: this.T.slice(), points: 0, inliers: 0, ms: 0, lag: this.lag };
    if (w !== this.w || h !== this.h) {
      this.alloc(w, h);
      this.ready = false;
      this.n = 0;
    }
    this.f = f;
    this.cx = (w - 1) / 2;
    this.cy = (h - 1) / 2;
    this.height = height;
    this.mem.rgba.set(rgba.subarray(0, 4 * w * h));
    this.ex.gray_image(this.at.rgba, w * h, this.cur);
    this.ex.pyramid(this.cur, w, h, LEVELS);
    this.ex.highpass(this.cur + this.ex.level_offset(w, h, SHIFT_LEVEL), this.sw, this.sh, this.hpCur, this.at.tmp);
    let state;
    if (!this.ready) {
      // not tracking yet (or lost): the rotation follows the gyroscope, from where it was
      if (!this.C) this.C = orthonormalize(Cg);
      else this.C = orthonormalize(mul(this.C, mul(transpose(this.gyroAt(this.prevTime - this.lag)), Cg)));
      this.n = 0;
      this.addPoints();
      this.ready = this.n >= MIN_POINTS;
      this.v = [0, 0, 0];
      state = this.ready ? "started" : "searching";
    } else {
      state = this.follow(Cg);
      if (this.n < TARGET) this.addPoints();
    }
    [this.prev, this.cur] = [this.cur, this.prev];
    [this.hpPrev, this.hpCur] = [this.hpCur, this.hpPrev];
    this.prevTime = time;
    if (state === "tracking" || state === "started") {
      this.seen.push({ t: time, C: this.C });
      while (this.seen.length > 2 && this.seen[0].t < time - HISTORY_MS) this.seen.shift();
      if (++this.lagCheck % 10 === 0) this.measureLag();
    } else {
      this.seen = [];
    }
    return { state, ok: state === "tracking" || state === "started", C: this.C.slice(), T: this.T.slice(),
             points: this.n, inliers: this.inliers, ms: performance.now() - t0, lag: this.lag,
             followed: this.followed, shiftFix: this.shiftFix, gyroOff: this.gyroOff };
  }

  follow(Cg) {
    const { f, cx, cy, mem, at, w, h } = this;
    let n = this.n;
    // 1. predict
    // both ends of the gyroscope's turn read with the same delay (it can change between frames)
    const Cg0 = this.gyroAt(this.prevTime - this.lag);
    const Cp = orthonormalize(mul(this.C, mul(transpose(Cg0), Cg)));
    const Tp = this.T.map((x, i) => x + 0.6 * this.v[i]);
    const Wp = transpose(Cp);
    for (let i = 0; i < n; i++) {
      const X = [this.wx[3 * i] - Tp[0], this.wx[3 * i + 1] - Tp[1], this.wx[3 * i + 2] - Tp[2]];
      const c = mulv(Wp, X);
      mem.src[2 * i] = this.px[2 * i];
      mem.src[2 * i + 1] = this.px[2 * i + 1];
      const inside = c[2] > 0.05;
      mem.dst[2 * i] = inside ? f * c[0] / c[2] + cx : this.px[2 * i];
      mem.dst[2 * i + 1] = inside ? f * c[1] / c[2] + cy : this.px[2 * i + 1];
      mem.back[2 * i] = this.px[2 * i];
      mem.back[2 * i + 1] = this.px[2 * i + 1];
    }
    // the whole image's shift: correct the predicted positions if it disagrees
    this.shiftFix = 0;
    if (n) {
      let gx = 0, gy = 0;
      for (let i = 0; i < n; i++) { gx += mem.dst[2 * i] - mem.src[2 * i]; gy += mem.dst[2 * i + 1] - mem.src[2 * i + 1]; }
      gx /= n;
      gy /= n;
      const k = 1 << SHIFT_LEVEL;
      const found = this.ex.global_shift(this.hpPrev, this.hpCur, this.sw, this.sh, Math.round(gx / k),
                                         Math.round(gy / k), SHIFT_RANGE, at.shift);
      if (found && mem.shift[2] < 0.7 * mem.shift[3]) {
        const cx2 = mem.shift[0] * k - gx, cy2 = mem.shift[1] * k - gy;
        if (Math.hypot(cx2, cy2) > 2) {
          for (let i = 0; i < n; i++) { mem.dst[2 * i] += cx2; mem.dst[2 * i + 1] += cy2; }
          this.shiftFix = Math.hypot(cx2, cy2);
        }
      }
    }
    // 2. optical flow, forward then backward
    this.ex.track_points(this.prev, this.cur, w, h, LEVELS, n, at.src, at.dst, at.st1, at.err1);
    this.ex.track_points(this.cur, this.prev, w, h, LEVELS, n, at.dst, at.back, at.st2, at.err2);
    const good = new Uint8Array(n);
    this.followed = 0;
    for (let i = 0; i < n; i++) {
      const fb = Math.hypot(mem.back[2 * i] - mem.src[2 * i], mem.back[2 * i + 1] - mem.src[2 * i + 1]);
      good[i] = mem.st1[i] && mem.st2[i] && fb < FB_PX && mem.err1[i] < MAX_ERR ? 1 : 0;
      this.followed += good[i];
      mem.obs[2 * i] = (mem.dst[2 * i] - cx) / f;
      mem.obs[2 * i + 1] = (mem.dst[2 * i + 1] - cy) / f;
      mem.xyz[3 * i] = this.wx[3 * i];
      mem.xyz[3 * i + 1] = this.wx[3 * i + 1];
      mem.xyz[3 * i + 2] = this.wx[3 * i + 2];
      mem.wts[i] = good[i];
    }
    // 3. pose. First from the image alone (the gyroscope barely counts): with a robust loss each
    // point's pull is capped, so a wrong gyroscope reading (a wrong delay, a glitch) could win.
    // Then, without the points that don't fit, again with the gyroscope's rotation as a prior if it
    // agrees with the image: it steadies what the image tells least well (turning vs. moving
    // sideways, when the floor points are far).
    const t = mulv(Wp, Tp).map((x) => -x);
    mem.pose.set([...Wp, ...t]);
    mem.prm.set([...Wp, WEAK_PRIOR, 1.5 / f]);
    mem.prmU[11] = 10;
    this.ex.solve_pose(n, at.obs, at.xyz, at.wts, at.pose, at.prm, at.resid);
    for (let i = 0; i < n; i++) mem.wts[i] = good[i] && mem.resid[i] * f < 3 * INLIER_PX ? 1 : 0;
    const Cv = transpose(Array.from(mem.pose.subarray(0, 9)));
    const disagree = Math.hypot(...rotvec(mul(transpose(Cp), Cv)));
    this.gyroOff = disagree;
    mem.prm[9] = disagree < AGREE ? PRIOR : WEAK_PRIOR;
    mem.prmU[11] = 6;
    this.ex.solve_pose(n, at.obs, at.xyz, at.wts, at.pose, at.prm, at.resid);
    let inliers = 0;
    for (let i = 0; i < n; i++) if (good[i] && mem.resid[i] * f < INLIER_PX) inliers++;
    this.inliers = inliers;
    if (inliers < MIN_POINTS) {
      // lost: keep the position, turn with the gyroscope, start a new map from here
      this.C = Cp;
      this.v = [0, 0, 0];
      this.n = 0;
      this.lostFrames++;
      this.ready = false;
      return "lost";
    }
    const W = Array.from(mem.pose.subarray(0, 9));
    const C = orthonormalize(transpose(W));
    const T = mulv(C, [-mem.pose[9], -mem.pose[10], -mem.pose[11]]);
    this.v = T.map((x, i) => x - this.T[i]);
    this.C = C;
    this.T = T;
    this.lostFrames = 0;
    // 4. keep the points that fit
    let m = 0;
    for (let i = 0; i < n; i++) {
      const e = mem.resid[i] * f;
      if (!good[i] || e > 3 * INLIER_PX) continue;
      const bad = e > INLIER_PX ? this.bad[i] + 1 : 0;
      if (bad > 2) continue;
      this.px[2 * m] = mem.dst[2 * i];
      this.px[2 * m + 1] = mem.dst[2 * i + 1];
      this.wx[3 * m] = this.wx[3 * i];
      this.wx[3 * m + 1] = this.wx[3 * i + 1];
      this.wx[3 * m + 2] = this.wx[3 * i + 2];
      this.bad[m] = bad;
      m++;
    }
    this.n = m;
    return "tracking";
  }

  // The floor point seen at pixel (u, v) of the current frame, or null.
  floorPoint(u, v, down = MIN_DOWN) {
    const x = (u - this.cx) / this.f, y = (v - this.cy) / this.f;
    const len = Math.hypot(x, y, 1);
    const d = mulv(this.C, [x / len, y / len, 1 / len]);
    if (d[2] > -down) return null;
    const s = (-this.height - this.T[2]) / d[2];
    if (s <= 0 || s * Math.hypot(d[0], d[1]) > MAX_DIST) return null;
    return [this.T[0] + s * d[0], this.T[1] + s * d[1], -this.height];
  }

  addPoints() {
    const { mem, at, cw, ch, w, h } = this;
    for (let j = 0; j < ch; j++) {
      for (let i = 0; i < cw; i++) {
        mem.skip[j * cw + i] = this.floorPoint((i + 0.5) * CELL, (j + 0.5) * CELL) ? 0 : 1;
      }
    }
    for (let k = 0; k < this.n; k++) {
      const i = Math.floor(this.px[2 * k] / CELL), j = Math.floor(this.px[2 * k + 1] / CELL);
      if (i >= 0 && j >= 0 && i < cw && j < ch) mem.skip[j * cw + i] = 1;
    }
    const m = this.ex.corners(this.cur, w, h, CELL, at.skip, BORDER, at.sxx, at.sxy, at.syy, at.tmp, at.cand);
    const order = [];
    for (let k = 0; k < m; k++) if (mem.cand[3 * k + 2] >= MIN_SCORE) order.push(k);
    order.sort((a, b) => mem.cand[3 * b + 2] - mem.cand[3 * a + 2]);
    for (const k of order) {
      if (this.n >= MAX_POINTS) break;
      const u = mem.cand[3 * k], v = mem.cand[3 * k + 1];
      const X = this.floorPoint(u, v, 0.8 * MIN_DOWN);
      if (!X) continue;
      const i = this.n++;
      this.px[2 * i] = u;
      this.px[2 * i + 1] = v;
      this.wx[3 * i] = X[0];
      this.wx[3 * i + 1] = X[1];
      this.wx[3 * i + 2] = X[2];
      this.bad[i] = 0;
    }
  }
}
