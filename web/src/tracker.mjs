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
// Tables (and other flat surfaces above the floor): a point that doesn't fit the floor isn't thrown
// away but followed on as a free point, and its real place is triangulated from the rays it was
// seen along as the phone moves (once they are 4 degrees apart). Free points at the same height
// (8 or more within 5 cm, 15 cm to 1.6 m above the floor) make a surface, with the outline of its
// points. New points seen on a known surface get their depth at once, like floor points, and the
// ring and the scene go on the nearest surface under the screen. A new point counts for little
// (on probation) until the phone has moved enough to see it from 3 degrees further round and it
// still fits: until then, points on a table that are taken for floor can't bend the pose.
//
// Floor memory: while tracking works, what the camera sees of the floor is added to a top-down
// picture of it (2 cm cells, track.ha). When tracking is lost, it starts again at once from where
// it was (the rotation from the gyroscope), and meanwhile the floor it sees now, seen from above,
// is searched for in the picture: first in 8 cm cells as far as one could have walked, then the
// best few places again in 2 cm cells (repeating floors like tiles look alike in 8 cm cells; the
// fine grain of each tile tells them apart). A clear winner gives how far the phone really moved,
// and everything (the phone, the points, the scene placed on the floor) is put back in its place.
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
const MAP_CELL = 0.02;    // metres per cell of the floor picture
const MAP_HALF = 6;       // metres it reaches from where tracking started, each way
const MAP_DIST = 3;       // metres: floor further away is too blurry to remember
const MAP_EVERY = 3;      // tracked frames between additions to the picture
const RELOC_DOWN = 4;     // the first search is in cells this many times bigger (8 cm)
const RELOC_MIN = 0.35;   // correlation a match needs
const RELOC_CLEAR = 0.08; // and by how much it must beat the other candidates, in 2 cm cells
const RELOC_TRIES = 5;    // candidates from the 8 cm search looked at in 2 cm cells
const RELOC_GIVE_UP = 10000;  // ms: then the old picture is dropped and a new one started
const PATCH_MAX = 160;    // cells across the patch compared with the picture
const MAX_FREE = 60;      // points being triangulated at a time
const TRI_ANGLE = 4 * Math.PI / 180;   // rays this far apart give a point's place
const PLANE_MIN = 8;      // points at one height that make a surface
const PLANE_BAND = 0.05;  // metres: how close in height they must be
const PLANE_LOW = 0.15, PLANE_HIGH = 1.6;   // metres above the floor a surface can be
const CONFIRM_ANGLE = 3 * Math.PI / 180;    // a new point is confirmed once seen this far round
const PROBATION = 0.05;   // how much a point on probation counts
const PROBATION_PX = 1.5; // a point on probation that is off by more than this is freed at once

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

// Convex hull of 2D points (Andrew's monotone chain), counter-clockwise.
export function convexHull(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

// Whether (x, y) is inside a counter-clockwise convex polygon, or within `margin` metres of it.
export function insideHull(hull, x, y, margin = 0) {
  if (hull.length < 3) return hull.some(([a, b]) => Math.hypot(a - x, b - y) <= margin);
  for (let i = 0; i < hull.length; i++) {
    const [ax, ay] = hull[i], [bx, by] = hull[(i + 1) % hull.length];
    const ex = bx - ax, ey = by - ay, len = Math.hypot(ex, ey) || 1;
    if ((ex * (y - ay) - ey * (x - ax)) / len < -margin) return false;
  }
  return true;
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
    this.kind = new Uint8Array(MAX_POINTS);           // 0: on a known surface, 1: free, 2: triangulated
    this.tri = new Float64Array(9 * MAX_POINTS);      // sum of (I - d d^T), sum of (I - d d^T) o over its rays
    this.d0 = new Float64Array(3 * MAX_POINTS);       // the first ray it was seen along
    this.rays = new Uint16Array(MAX_POINTS);          // how many rays it was seen along
    this.conf = new Uint8Array(MAX_POINTS);           // confirmed (seen from 3 degrees round, fitting)
    // the floor picture lives at the start of the free memory; the per-image buffers come after it
    this.mw = this.mh = Math.round(2 * MAP_HALF / MAP_CELL);
    const cw = this.mw / RELOC_DOWN, cells = this.mw * this.mh, ccells = cw * cw;
    const at = {};
    let o = this.base;
    const take = (name, bytes) => { at[name] = o; o += Math.ceil(bytes / 16) * 16; };
    take("map", cells); take("mapWt", cells); take("cmap", ccells); take("cmapWt", ccells);
    take("patch", 4 * PATCH_MAX * PATCH_MAX); take("scores", 4 * 65 * 65); take("view", 80); take("match", 16);
    const have = this.memory.buffer.byteLength;
    if (o > have) this.memory.grow(Math.ceil((o - have) / 65536));
    this.mapAt = at;
    this.base = o;
    this.opts = { relocalize: true, triWeight: 0.5 };   // triWeight: how much a triangulated point counts
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
    this.verified = true;             // the pose agrees with the floor picture (false after a loss)
    this.lostAt = 0;
    this.mapLooks = 0;                // additions to the floor picture so far
    this.mapTick = 0;
    this.relocTick = 0;
    this.relocs = 0;                  // times the place was found again
    this.planes = [];                 // surfaces above the floor: { z, hull: [[x, y], ...], n }
    this.planeTick = 0;
    this.coarseReady = false;
    if (this.mapAt) this.clearMap();

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

  clearMap() {
    new Uint8Array(this.memory.buffer, this.mapAt.mapWt, this.mw * this.mh).fill(0);
    this.mapLooks = 0;
  }

  // The camera and the floor for track.ha's floor functions, with a grid whose corner is (x0, y0).
  setView(x0, y0, cell) {
    const W = transpose(this.C);
    new Float32Array(this.memory.buffer, this.mapAt.view, 20).set(
      [...W, ...this.T, this.f, this.cx, this.cy, -this.height, x0, y0, cell, MAP_DIST]);
  }

  // The part of the floor the camera sees (within MAP_DIST), as a box [x0, y0, x1, y1], or null.
  visibleFloor() {
    const { w, h, f, cx, cy, C, T } = this;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, any = false;
    for (let a = 0; a <= 4; a++) for (let b = 0; b <= 4; b++) {
      const x = (a * (w - 1) / 4 - cx) / f, y = (b * (h - 1) / 4 - cy) / f;
      const d = mulv(C, [x, y, 1]);
      if (d[2] > -0.02) continue;
      const s = (-this.height - T[2]) / d[2];
      const r = Math.min(MAP_DIST, s * Math.hypot(d[0], d[1]));
      const k = r / Math.max(1e-9, Math.hypot(d[0], d[1]));
      const px = T[0] + k * d[0], py = T[1] + k * d[1];
      x0 = Math.min(x0, px); y0 = Math.min(y0, py); x1 = Math.max(x1, px); y1 = Math.max(y1, py);
      any = true;
    }
    return any ? [x0, y0, x1, y1] : null;
  }

  // Adds what the camera sees of the floor now to the floor picture.
  mapUpdate() {
    const box = this.visibleFloor();
    if (!box) return;
    const g = (v) => Math.max(0, Math.min(this.mw, Math.floor((v + MAP_HALF) / MAP_CELL)));
    this.setView(-MAP_HALF, -MAP_HALF, MAP_CELL);
    const a = this.mapAt;
    this.ex.floor_map_update(this.cur, this.w, this.h, a.view, a.map, a.mapWt, this.mw,
                             g(box[0]), g(box[1]), Math.min(this.mw, g(box[2]) + 1), Math.min(this.mh, g(box[3]) + 1));
    this.mapLooks++;
  }

  // Compares the floor seen now with the picture, in cells of `cell` metres (map: the picture at
  // that size, mw cells wide), within `range` cells of (di0, dj0). Returns the best shifts, apart
  // from each other: [{ di, dj, score }], best first.
  matchFloor(map, mapWt, mw, cell, di0, dj0, range, count = 1) {
    const box = this.visibleFloor();
    if (!box) return null;
    const a = this.mapAt;
    const mx = (box[0] + box[2]) / 2, my = (box[1] + box[3]) / 2;
    const pw = Math.min(PATCH_MAX, Math.ceil((box[2] - box[0]) / cell) + 1);
    const ph = Math.min(PATCH_MAX, Math.ceil((box[3] - box[1]) / cell) + 1);
    const pi = Math.round((mx + MAP_HALF) / cell - pw / 2), pj = Math.round((my + MAP_HALF) / cell - ph / 2);
    this.setView(-MAP_HALF + pi * cell, -MAP_HALF + pj * cell, cell);
    this.ex.floor_patch(this.cur, this.w, this.h, a.view, pw, ph, a.patch);
    this.ex.floor_map_match(map, mapWt, mw, mw, a.patch, pw, ph, pi + di0, pj + dj0, range, a.scores, a.match);
    const side = 2 * range + 1, sc = new Float32Array(this.memory.buffer, a.scores, side * side);
    const peaks = [];
    for (let j = 0; j < side; j++) for (let i = 0; i < side; i++) {
      const v = sc[j * side + i];
      if (v < RELOC_MIN - 0.1) continue;
      let top = true;                  // a local maximum over its 3 x 3 neighbours
      for (let b = -1; b <= 1 && top; b++) for (let c = -1; c <= 1; c++) {
        const y = j + b, x = i + c;
        if ((b || c) && y >= 0 && x >= 0 && y < side && x < side && sc[y * side + x] > v) { top = false; break; }
      }
      if (top) peaks.push({ di: di0 + i - range, dj: dj0 + j - range, score: v });
    }
    peaks.sort((p, q) => q.score - p.score);
    const kept = [];
    for (const pk of peaks) {
      if (kept.every((k) => Math.abs(k.di - pk.di) > 3 || Math.abs(k.dj - pk.dj) > 3)) kept.push(pk);
      if (kept.length >= count) break;
    }
    return kept;
  }

  // After a loss: find the place again in the floor picture. See "Floor memory" at the top.
  relocalize(time) {
    const a = this.mapAt, cmw = this.mw / RELOC_DOWN, cell = MAP_CELL * RELOC_DOWN;
    if (!this.coarseReady) {
      this.ex.floor_map_down(a.map, a.mapWt, this.mw, this.mh, RELOC_DOWN, a.cmap, a.cmapWt);
      this.coarseReady = true;
    }
    const reach = Math.min(2.5, 0.4 + 1.5 * (time - this.lostAt) / 1000);   // metres one could have walked
    const coarse = this.matchFloor(a.cmap, a.cmapWt, cmw, cell, 0, 0, Math.min(32, Math.ceil(reach / cell)), RELOC_TRIES);
    if (!coarse || !coarse.length) return false;
    const fine = [];
    for (const c of coarse) {
      const m = this.matchFloor(a.map, a.mapWt, this.mw, MAP_CELL, RELOC_DOWN * c.di, RELOC_DOWN * c.dj, 5);
      if (m && m.length) fine.push(m[0]);
    }
    fine.sort((p, q) => q.score - p.score);
    const best = fine[0], other = fine.find((m) => Math.abs(m.di - best.di) > 6 || Math.abs(m.dj - best.dj) > 6);
    this.lastMatch = best && { ...best, second: other ? other.score : -2 };
    if (!best || best.score < RELOC_MIN || (other && other.score > best.score - RELOC_CLEAR)) return false;
    const dx = best.di * MAP_CELL, dy = best.dj * MAP_CELL;
    this.T = [this.T[0] + dx, this.T[1] + dy, this.T[2]];
    for (let i = 0; i < this.n; i++) {
      this.wx[3 * i] += dx;
      this.wx[3 * i + 1] += dy;
      const t = this.tri, k = 9 * i;                   // the rays' origins move too: b += A (dx, dy, 0)
      t[k + 6] += t[k] * dx + t[k + 1] * dy;
      t[k + 7] += t[k + 1] * dx + t[k + 3] * dy;
      t[k + 8] += t[k + 2] * dx + t[k + 4] * dy;
    }
    for (const p of this.planes) p.hull = p.hull.map(([x, y]) => [x + dx, y + dy]);
    this.verified = true;
    this.relocs++;
    this.relocShift = Math.hypot(dx, dy);
    return true;
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
    take("hpA", 4 * sw * sh); take("hpB", 4 * sw * sh); take("shift", 16); take("light", 32);
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
      resid: f32("resid", MAX_POINTS), pose: f32("pose", 12), prm: f32("prm", 12), shift: f32("shift", 4), light: f32("light", 7),
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
  // phone's height above the floor when tracking starts (metres); box: [x0, y0, x1, y1], pixels of
  // this image where the floor under the scene is (for the light there). Returns the pose for this
  // frame and the light in it.
  frame(rgba, w, h, f, time, gyro, height = 1.35, box = null) {
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
    const b = (box || [0, 0, 0, 0]).map((v, i) => Math.max(0, Math.min(i % 2 ? h : w, Math.round(v))));
    this.ex.light_stats(this.at.rgba, w, h, b[0], b[1], b[2], b[3], this.at.light);
    const L = this.mem.light;
    const light = { frame: [L[0], L[1], L[2]], floor: L[6] > 20 ? [L[3], L[4], L[5]] : null };
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
    // the floor picture: add to it while the pose is sure; after a loss, find the place in it again
    if (state === "lost" && this.verified && this.mapLooks > 10 && this.opts.relocalize) {
      this.verified = false;
      this.lostAt = time;
      this.coarseReady = false;
    }
    if (!this.verified && ++this.relocTick % 2 === 0) {
      if (this.relocalize(time)) state = state === "tracking" ? "found" : state;
      else if (time - this.lostAt > RELOC_GIVE_UP) { this.clearMap(); this.verified = true; }
    }
    if (this.verified && state === "tracking" && ++this.mapTick % MAP_EVERY === 0) this.mapUpdate();
    [this.prev, this.cur] = [this.cur, this.prev];
    [this.hpPrev, this.hpCur] = [this.hpCur, this.hpPrev];
    this.prevTime = time;
    if (state === "found") state = "tracking";
    if (state === "tracking" || state === "started") {
      this.seen.push({ t: time, C: this.C });
      while (this.seen.length > 2 && this.seen[0].t < time - HISTORY_MS) this.seen.shift();
      if (++this.lagCheck % 10 === 0) this.measureLag();
    } else {
      this.seen = [];
    }
    return { state, ok: state === "tracking" || state === "started", C: this.C.slice(), T: this.T.slice(),
             points: this.n, inliers: this.inliers, ms: performance.now() - t0, lag: this.lag, light,
             verified: this.verified, relocs: this.relocs, mapLooks: this.mapLooks, match: this.lastMatch,
             planes: this.planes.map((p) => ({ z: p.z, hull: p.hull, n: p.n })),
             points3d: this.debug ? Array.from({ length: this.n }, (_, i) => i).filter((i) => this.kind[i] === 2)
               .map((i) => [this.wx[3 * i], this.wx[3 * i + 1], this.wx[3 * i + 2]]) : undefined,
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
      let X = [this.wx[3 * i] - Tp[0], this.wx[3 * i + 1] - Tp[1], this.wx[3 * i + 2] - Tp[2]];
      if (this.kind[i] === 1) X = this.ray(this.C, this.px[2 * i], this.px[2 * i + 1]);   // unknown depth: far
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
      mem.wts[i] = good[i] ? this.weight(i) : 0;
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
    for (let i = 0; i < n; i++) mem.wts[i] = good[i] && mem.resid[i] * f < 3 * INLIER_PX ? this.weight(i) : 0;
    const Cv = transpose(Array.from(mem.pose.subarray(0, 9)));
    const disagree = Math.hypot(...rotvec(mul(transpose(Cp), Cv)));
    this.gyroOff = disagree;
    mem.prm[9] = disagree < AGREE ? PRIOR : WEAK_PRIOR;
    mem.prmU[11] = 6;
    this.ex.solve_pose(n, at.obs, at.xyz, at.wts, at.pose, at.prm, at.resid);
    let inliers = 0;
    for (let i = 0; i < n; i++) if (good[i] && this.kind[i] !== 1 && mem.resid[i] * f < INLIER_PX) inliers++;
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
    // 4. keep the points that fit; a point followed well that doesn't fit the floor may be on
    // something else (a table): it becomes free, and its place is triangulated as the phone moves
    let m = 0, free = 0;
    for (let i = 0; i < n; i++) free += this.kind[i] === 1;
    for (let i = 0; i < n; i++) {
      if (!good[i]) continue;
      const u = mem.dst[2 * i], v = mem.dst[2 * i + 1];
      if (this.kind[i] === 1) {
        this.addRay(i, u, v);
        if (!this.promote(i, u, v)) {
          if (this.rays[i] > 90) continue;               // seen for 3 s without a place: drop it
        }
      } else {
        const e = mem.resid[i] * f;
        const bad = e > INLIER_PX ? this.bad[i] + 1 : 0;
        const doubt = this.kind[i] === 0 && !this.conf[i] && e > PROBATION_PX;
        if (e > 3 * INLIER_PX || bad > 2 || doubt) {
          if (this.kind[i] !== 0 || free >= MAX_FREE) continue;
          this.kind[i] = 1;                              // doesn't fit the floor: free it
          this.tri.fill(0, 9 * i, 9 * i + 9);
          this.addRay(i, u, v, true);
          free++;
        } else {
          this.bad[i] = bad;
          if (this.kind[i] === 2) { this.addRay(i, u, v); this.promote(i, u, v); }
          if (!this.conf[i] && e < INLIER_PX) {           // seen from far enough round, still fitting
            const d = this.ray(this.C, u, v), d0 = this.d0;
            const c = d[0] * d0[3 * i] + d[1] * d0[3 * i + 1] + d[2] * d0[3 * i + 2];
            if (Math.acos(Math.min(1, c)) > CONFIRM_ANGLE) this.conf[i] = 1;
          }
        }
      }
      this.px[2 * i] = u;
      this.px[2 * i + 1] = v;
      this.movePoint(m, i);
      m++;
    }
    this.n = m;
    if (++this.planeTick % 10 === 0) this.detectPlanes();
    return "tracking";
  }

  // Copies point i's state to slot m (m <= i).
  movePoint(m, i) {
    if (m === i) return;
    this.px[2 * m] = this.px[2 * i];
    this.px[2 * m + 1] = this.px[2 * i + 1];
    for (let k = 0; k < 3; k++) {
      this.wx[3 * m + k] = this.wx[3 * i + k];
      this.d0[3 * m + k] = this.d0[3 * i + k];
    }
    for (let k = 0; k < 9; k++) this.tri[9 * m + k] = this.tri[9 * i + k];
    this.bad[m] = this.bad[i];
    this.kind[m] = this.kind[i];
    this.rays[m] = this.rays[i];
    this.conf[m] = this.conf[i];
  }

  // How much point i counts in the pose: free points not at all, triangulated ones half, points on
  // a surface fully once confirmed.
  weight(i) {
    const k = this.kind[i];
    return k === 1 ? 0 : k === 2 ? this.opts.triWeight : this.conf[i] ? 1 : PROBATION;
  }

  // World direction of the ray through pixel (u, v) for camera rotation C.
  ray(C, u, v) {
    const x = (u - this.cx) / this.f, y = (v - this.cy) / this.f, len = Math.hypot(x, y, 1);
    return mulv(C, [x / len, y / len, 1 / len]);
  }

  // Adds the current ray through (u, v) to point i's triangulation: tri holds A = sum of (I - d d^T)
  // (xx, xy, xz, yy, yz, zz) and b = sum of (I - d d^T) o; the place is A^-1 b.
  addRay(i, u, v, first = false) {
    const d = this.ray(this.C, u, v), o = this.T, t = this.tri, k = 9 * i;
    const P = [1 - d[0] * d[0], -d[0] * d[1], -d[0] * d[2], 1 - d[1] * d[1], -d[1] * d[2], 1 - d[2] * d[2]];
    for (let j = 0; j < 6; j++) t[k + j] += P[j];
    t[k + 6] += P[0] * o[0] + P[1] * o[1] + P[2] * o[2];
    t[k + 7] += P[1] * o[0] + P[3] * o[1] + P[4] * o[2];
    t[k + 8] += P[2] * o[0] + P[4] * o[1] + P[5] * o[2];
    if (first) {
      for (let j = 0; j < 3; j++) this.d0[3 * i + j] = d[j];
      this.rays[i] = 1;
    } else if (this.rays[i] < 65535) {
      this.rays[i]++;
    }
    this.lastRay = d;
  }

  // Once point i's rays are far enough apart, its place from them; it then counts like a floor point.
  promote(i, u, v) {
    const d = this.lastRay, d0 = [this.d0[3 * i], this.d0[3 * i + 1], this.d0[3 * i + 2]];
    if (Math.acos(Math.min(1, d[0] * d0[0] + d[1] * d0[1] + d[2] * d0[2])) < TRI_ANGLE || this.rays[i] < 5) return false;
    const t = this.tri, k = 9 * i;
    const a = t[k], b = t[k + 1], c = t[k + 2], e = t[k + 3], g = t[k + 4], h = t[k + 5];
    const det = a * (e * h - g * g) - b * (b * h - g * c) + c * (b * g - e * c);
    if (Math.abs(det) < 1e-9) return false;
    const inv = [e * h - g * g, c * g - b * h, b * g - c * e, a * h - c * c, b * c - a * g, a * e - b * b];
    const r = [t[k + 6], t[k + 7], t[k + 8]];
    const X = [(inv[0] * r[0] + inv[1] * r[1] + inv[2] * r[2]) / det,
               (inv[1] * r[0] + inv[3] * r[1] + inv[4] * r[2]) / det,
               (inv[2] * r[0] + inv[4] * r[1] + inv[5] * r[2]) / det];
    const cam = mulv(transpose(this.C), [X[0] - this.T[0], X[1] - this.T[1], X[2] - this.T[2]]);
    if (cam[2] < 0.2) return false;
    const err = Math.hypot(this.f * cam[0] / cam[2] + this.cx - u, this.f * cam[1] / cam[2] + this.cy - v);
    if (err > 2 * INLIER_PX || X[2] < -this.height - 0.1) return false;
    const above = X[2] + this.height;
    if (this.kind[i] === 1 && above < 0.03) {             // on the floor after all: exactly there
      this.wx[3 * i] = X[0];
      this.wx[3 * i + 1] = X[1];
      this.wx[3 * i + 2] = -this.height;
      this.kind[i] = 0;
      this.bad[i] = 0;
      this.conf[i] = 1;
      return true;
    }
    if (above < 0.08) return false;                     // too close to the floor to tell
    this.wx[3 * i] = X[0];
    this.wx[3 * i + 1] = X[1];
    this.wx[3 * i + 2] = X[2];
    this.kind[i] = 2;
    return true;
  }

  // Surfaces above the floor from the triangulated points (see "Tables" at the top).
  detectPlanes() {
    const pts = [];
    for (let i = 0; i < this.n; i++) {
      if (this.kind[i] !== 2) continue;
      const h = this.wx[3 * i + 2] + this.height;
      if (h > PLANE_LOW && h < PLANE_HIGH) pts.push([this.wx[3 * i], this.wx[3 * i + 1], this.wx[3 * i + 2]]);
    }
    if (pts.length < PLANE_MIN) return;
    pts.sort((p, q) => p[2] - q[2]);
    let best = null;
    for (let i = 0, j = 0; i < pts.length; i++) {
      while (pts[i][2] - pts[j][2] > PLANE_BAND) j++;
      if (!best || i - j + 1 > best[1] - best[0]) best = [j, i + 1];
    }
    const members = pts.slice(best[0], best[1]);
    if (members.length < PLANE_MIN) return;
    const z = members[members.length >> 1][2];
    const near = this.planes.find((p) => Math.abs(p.z - z) < PLANE_BAND &&
      p.hull.some(([x, y]) => members.some((q) => Math.hypot(q[0] - x, q[1] - y) < 0.6)));
    if (near) {
      near.z = (near.z * near.n + z * members.length) / (near.n + members.length);
      near.n += members.length;
      near.hull = convexHull([...near.hull, ...members.map((q) => [q[0], q[1]])]);
    } else {
      this.planes.push({ z, n: members.length, hull: convexHull(members.map((q) => [q[0], q[1]])) });
    }
  }

  // Where the ray through pixel (u, v) meets a surface (the nearest known one above the floor it
  // falls inside, else the floor), or null.
  surfacePoint(u, v, down = MIN_DOWN) {
    const d = this.ray(this.C, u, v);
    let best = null, bestS = Infinity;
    for (const p of this.planes) {
      if (d[2] > -0.05 || p.z > this.T[2] - 0.1) continue;
      const s = (p.z - this.T[2]) / d[2];
      const X = [this.T[0] + s * d[0], this.T[1] + s * d[1], p.z];
      if (s < bestS && insideHull(p.hull, X[0], X[1], 0.05)) { best = X; bestS = s; }
    }
    return best || this.floorPoint(u, v, down);
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
      const X = this.surfacePoint(u, v, 0.8 * MIN_DOWN);
      if (!X) continue;
      const i = this.n++;
      this.px[2 * i] = u;
      this.px[2 * i + 1] = v;
      this.wx[3 * i] = X[0];
      this.wx[3 * i + 1] = X[1];
      this.wx[3 * i + 2] = X[2];
      this.bad[i] = 0;
      this.kind[i] = 0;
      this.conf[i] = 0;
      const d = this.ray(this.C, u, v);
      this.d0[3 * i] = d[0];
      this.d0[3 * i + 1] = d[1];
      this.d0[3 * i + 2] = d[2];
    }
  }
}
