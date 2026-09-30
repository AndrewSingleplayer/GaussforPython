// Splat data engine for the web viewer: decode the download format and sort splats for drawing.
// The work is done by the HA++ module (splatweb.ha compiled to WebAssembly). If a browser refuses
// to run WebAssembly, the same algorithms run in plain JavaScript (JsEngine), and `engine.kind`
// says which one is in use. Used by worker.js in browsers and by the tests in Node.

const HEADER = 64;
const BOX_BYTES = 32;           // struct Box in splatweb.ha
const CAM_BYTES = 60;           // struct SortCamera

export function parseHeader(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== "HSPL") throw new Error("not a splat file (.hspl)");
  const version = dv.getUint32(4, true);
  if (version !== 1) throw new Error(`unsupported .hspl version ${version}`);
  const f = (o) => dv.getFloat32(o, true);
  return {
    count: dv.getUint32(8, true),
    lo: [f(16), f(20), f(24)], size: [f(28), f(32), f(36)], lmin: f(40), lrange: f(44),
    front: [f(48), f(52), f(56)], distance: f(60),
  };
}

// Where the scene's base is, where its middle is and how tall it is (y is up), ignoring the
// outermost 2% of splats: used to stand a scene on the floor in AR at a real size. The middle (cx,
// cz) is that of the bottom slice, the part that touches the floor, so a scene that leans (a head
// sticking out) still stands centred on the spot it is put on.
export function sceneStats(pos, n) {
  const step = Math.max(1, Math.floor(n / 20000));
  const xs = [], ys = [], zs = [];
  for (let i = 0; i < n; i += step) {
    xs.push(pos[3 * i]);
    ys.push(pos[3 * i + 1]);
    zs.push(pos[3 * i + 2]);
  }
  const at = (a, q) => a[Math.min(a.length - 1, Math.floor(q * (a.length - 1)))];
  const sorted = (a) => a.slice().sort((u, v) => u - v);
  const ysSorted = sorted(ys);
  const ground = at(ysSorted, 0.02), top = at(ysSorted, 0.98);
  const band = ground + 0.12 * (top - ground);
  let bx = [], bz = [];
  ys.forEach((y, i) => { if (y >= ground && y <= band) { bx.push(xs[i]); bz.push(zs[i]); } });
  if (bx.length < 30) { bx = xs; bz = zs; }
  const mid = (a) => { const s = sorted(a); return 0.5 * (at(s, 0.1) + at(s, 0.9)); };
  const cx = mid(bx), cz = mid(bz);
  const rs = sorted(xs.map((x, i) => Math.hypot(x - cx, zs[i] - cz)));
  return { ground, top, cx, cz, radius: at(rs, 0.9) };
}

// camera: { rot: [r0 x3, r1 x3, r2 x3] (world -> camera rows: right, down, forward), t: [x, y, z],
//           tanX, tanY, near }
function cameraFloats(cam) {
  return [...cam.rot, ...cam.t, cam.tanX, cam.tanY, cam.near];
}

class WasmEngine {
  constructor(instance) {
    this.kind = "HA++ WebAssembly";
    this.ex = instance.exports;
    this.memory = this.ex.memory;
    this.base = Math.ceil(Number(this.ex.__heap_base.value) / 16) * 16;
    this.n = 0;
  }

  reserve(bytes) {
    const need = this.base + bytes;
    const have = this.memory.buffer.byteLength;
    if (need > have) this.memory.grow(Math.ceil((need - have) / 65536));
  }

  load(buf) {
    const h = parseHeader(buf);
    const n = h.count;
    // memory map: [src 16n][box 32][cam 64][gpu 32n][pos 12n][radius 4n][keys 4n][ids 4n][counts 256K][order 4n]
    const at = {};
    let o = this.base;
    const take = (name, bytes) => { at[name] = o; o += Math.ceil(bytes / 16) * 16; };
    take("src", 16 * n); take("box", BOX_BYTES); take("cam", 64); take("gpu", 32 * n); take("pos", 12 * n);
    take("radius", 4 * n); take("keys", 4 * n); take("ids", 4 * n); take("counts", 4 * 65536); take("order", 4 * n);
    this.reserve(o - this.base);
    this.at = at;
    this.n = n;
    const mem = this.memory.buffer;
    new Uint8Array(mem, at.src, 16 * n).set(new Uint8Array(buf, HEADER, 16 * n));
    new Float32Array(mem, at.box, 8).set([...h.lo, ...h.size, h.lmin, h.lrange]);
    const t0 = performance.now();
    this.ex.decode_splats(at.src, n, at.box, at.gpu, at.pos, at.radius);
    const ms = performance.now() - t0;
    const stats = sceneStats(new Float32Array(this.memory.buffer, at.pos, 3 * n), n);
    return { header: h, gpu: new Uint32Array(this.memory.buffer, at.gpu, 8 * n).slice(), decodeMs: ms, stats };
  }

  sort(cam, budget) {
    const at = this.at;
    const n = Math.min(this.n, budget ?? this.n);
    new Float32Array(this.memory.buffer, at.cam, 15).set(cameraFloats(cam));
    const m = this.ex.sort_splats(at.pos, at.radius, n, at.cam, at.keys, at.ids, at.counts, at.order);
    return new Uint32Array(this.memory.buffer, at.order, m).slice();
  }
}

class JsEngine {
  constructor() {
    this.kind = "JavaScript (WebAssembly unavailable)";
    this.n = 0;
  }

  load(buf) {
    const h = parseHeader(buf);
    const n = h.count;
    const t0 = performance.now();
    const u8 = new Uint8Array(buf, HEADER, 16 * n);
    const u16 = new Uint16Array(buf, HEADER, 8 * n);
    const u32 = new Uint32Array(buf, HEADER, 4 * n);
    const gpu = new Uint32Array(8 * n);
    const gf = new Float32Array(gpu.buffer);
    const pos = new Float32Array(3 * n);
    const radius = new Float32Array(n);
    const sx = h.size[0] / 65535, sy = h.size[1] / 65535, sz = h.size[2] / 65535;
    const ls = h.lrange / 255;
    for (let i = 0; i < n; i++) {
      const px = h.lo[0] + u16[i * 8 + 2] * sx, py = h.lo[1] + u16[i * 8 + 3] * sy, pz = h.lo[2] + u16[i * 8 + 4] * sz;
      const s0 = Math.exp(h.lmin + u8[i * 16 + 10] * ls), s1 = Math.exp(h.lmin + u8[i * 16 + 11] * ls),
            s2 = Math.exp(h.lmin + u8[i * 16 + 12] * ls);
      let x = u8[i * 16 + 13] / 127.5 - 1, y = u8[i * 16 + 14] / 127.5 - 1, z = u8[i * 16 + 15] / 127.5 - 1;
      // quantizing x, y, z can push x^2 + y^2 + z^2 above 1: then w = 0 and the quaternion is
      // scaled back to unit length (as quat_to_mat3 in HA++ does), or the covariance would be off
      const n2 = x * x + y * y + z * z;
      let w = 0;
      if (n2 > 1) { const inv = 1 / Math.sqrt(n2); x *= inv; y *= inv; z *= inv; } else w = Math.sqrt(1 - n2);
      // rotation matrix (rows), as quat_to_mat3 in happ/std/math.ha
      const m00 = 1 - 2 * (y * y + z * z), m01 = 2 * (x * y - w * z), m02 = 2 * (x * z + w * y);
      const m10 = 2 * (x * y + w * z), m11 = 1 - 2 * (x * x + z * z), m12 = 2 * (y * z - w * x);
      const m20 = 2 * (x * z - w * y), m21 = 2 * (y * z + w * x), m22 = 1 - 2 * (x * x + y * y);
      const a0 = s0 * s0, a1 = s1 * s1, a2 = s2 * s2;
      const big = Math.max(s0, s1, s2), scale = big * big, inv = 1 / scale;
      const cxx = (m00 * m00 * a0 + m01 * m01 * a1 + m02 * m02 * a2) * inv;
      const cxy = (m00 * m10 * a0 + m01 * m11 * a1 + m02 * m12 * a2) * inv;
      const cxz = (m00 * m20 * a0 + m01 * m21 * a1 + m02 * m22 * a2) * inv;
      const cyy = (m10 * m10 * a0 + m11 * m11 * a1 + m12 * m12 * a2) * inv;
      const cyz = (m10 * m20 * a0 + m11 * m21 * a1 + m12 * m22 * a2) * inv;
      const czz = (m20 * m20 * a0 + m21 * m21 * a1 + m22 * m22 * a2) * inv;
      const o = i * 8;
      gf[o] = px; gf[o + 1] = py; gf[o + 2] = pz;
      gpu[o + 3] = u32[i * 4];
      gf[o + 4] = scale;
      gpu[o + 5] = half2(cxx, cxy); gpu[o + 6] = half2(cxz, cyy); gpu[o + 7] = half2(cyz, czz);
      pos[i * 3] = px; pos[i * 3 + 1] = py; pos[i * 3 + 2] = pz;
      radius[i] = 3 * big;
    }
    const ms = performance.now() - t0;
    this.n = n;
    this.pos = pos;
    this.radius = radius;
    this.depth = new Float32Array(n);
    this.keys = new Uint32Array(n);
    this.ids = new Uint32Array(n);
    this.counts = new Uint32Array(65536);
    return { header: h, gpu, decodeMs: ms, stats: sceneStats(pos, n) };
  }

  sort(cam, budget) {
    const n = Math.min(this.n, budget ?? this.n);
    const [r00, r01, r02, r10, r11, r12, r20, r21, r22] = cam.rot;
    const [tx, ty, tz] = cam.t;
    const kx = Math.sqrt(1 + cam.tanX * cam.tanX), ky = Math.sqrt(1 + cam.tanY * cam.tanY);
    const { pos, radius, depth, ids, keys, counts } = this;
    let m = 0, zmin = 3e38, zmax = -3e38;
    for (let i = 0; i < n; i++) {
      const px = pos[3 * i], py = pos[3 * i + 1], pz = pos[3 * i + 2];
      const z = r20 * px + r21 * py + r22 * pz + tz;
      const r = radius[i];
      const x = r00 * px + r01 * py + r02 * pz + tx;
      const y = r10 * px + r11 * py + r12 * pz + ty;
      if (z > cam.near && Math.abs(x) - cam.tanX * z < r * kx && Math.abs(y) - cam.tanY * z < r * ky) {
        depth[m] = z; ids[m] = i; m++;
        if (z < zmin) zmin = z;
        if (z > zmax) zmax = z;
      }
    }
    if (m === 0) return new Uint32Array(0);
    counts.fill(0);
    const scale = 65535 / Math.max(zmax - zmin, 1e-6);
    for (let j = 0; j < m; j++) {
      const key = ((zmax - depth[j]) * scale) >>> 0;
      keys[j] = key;
      counts[key]++;
    }
    let sum = 0;
    for (let k = 0; k < 65536; k++) { const v = counts[k]; counts[k] = sum; sum += v; }
    const order = new Uint32Array(m);
    for (let j = 0; j < m; j++) order[counts[keys[j]]++] = ids[j];
    return order;
  }
}

// float -> IEEE half bits (round to nearest even), two per u32 like pack_half2
const f32 = new Float32Array(1);
const u32v = new Uint32Array(f32.buffer);
function half(v) {
  f32[0] = v;
  const x = u32v[0];
  const sign = (x >>> 16) & 0x8000;
  const abs = x & 0x7fffffff;
  if (abs > 0x7f800000) return sign | 0x7e00;
  if (abs >= 0x477ff000) return sign | 0x7c00;
  if (abs < 0x38800000) {                        // subnormal half
    f32[0] = Math.abs(v) + 0.5;
    return sign | (u32v[0] - 0x3f000000);
  }
  const t = abs + 0x0fff + ((abs >>> 13) & 1);
  return sign | ((t >>> 13) - 0x1c000);
}
function half2(a, b) { return (half(a) | (half(b) << 16)) >>> 0; }

export async function createEngine(wasmSource, { forceJs = false } = {}) {
  if (!forceJs) {
    try {
      let bytes = wasmSource;
      if (!(bytes instanceof ArrayBuffer) && !ArrayBuffer.isView(bytes)) {
        const r = await fetch(wasmSource);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        bytes = await r.arrayBuffer();
      }
      const { instance } = await WebAssembly.instantiate(bytes, { env: {} });
      return new WasmEngine(instance);
    } catch (e) {
      const js = new JsEngine();
      js.why = String(e && e.message || e);
      return js;
    }
  }
  return new JsEngine();
}
