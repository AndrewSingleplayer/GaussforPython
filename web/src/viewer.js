// HA++ Splats: a Gaussian splat viewer for phones, Safari on iPhone included.
//
// Drawing (WebGL2): every splat is one quad along the axes of its projected ellipse, cut where the
// Gaussian falls below 1/255 of the splat's opacity, blended back to front with premultiplied alpha.
// The GPU's own blending does the compositing (on iPhones it happens in on-chip tile memory).
// Sorting and culling run in a Web Worker, in the HA++ module compiled to WebAssembly.

import { LookAroundAR } from "./ar.js";

const $ = (id) => document.getElementById(id);
const AR_URL = "https://andrewsingleplayer.github.io/GaussforPython/";
const canvas = $("view");
const hud = { fps: $("fps"), frame: $("frame"), sort: $("sort"), drawn: $("drawn"), res: $("res"), engine: $("engine"),
              status: $("status"), quality: $("quality"), bar: $("bar"), scene: $("scene-name"), track: $("track-info") };

// ------------------------------------------------------------------ shaders
const VS = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D u_data;          // 2 texels per splat: position, colour | scale, covariance/scale (f16 x6)
uniform vec3 u_r0, u_r1, u_r2, u_t; // world -> camera (x right, y down, z forward)
uniform vec2 u_focal, u_size, u_lim;
uniform float u_near;
// AR light matching, applied here to each splat as it is drawn (the scene data is never changed):
// u_light: gain for the colours (room brightness and colour cast); u_floor: (scene y of the floor,
// metres per scene unit, 1 in AR / 0 otherwise); u_floorTint: the floor's colour cast.
uniform vec3 u_light, u_floorTint;
uniform vec3 u_floor;
layout(location = 0) in vec2 a_corner;
layout(location = 1) in uint a_index;
out vec4 v_color;
out vec2 v_uv;                      // position inside the splat, in standard deviations
void main() {
  ivec2 tc = ivec2(int((a_index & 2047u) << 1), int(a_index >> 11));
  uvec4 d0 = texelFetch(u_data, tc, 0);
  uvec4 d1 = texelFetch(u_data, tc + ivec2(1, 0), 0);
  vec3 p = uintBitsToFloat(d0.xyz);
  vec3 c = vec3(dot(u_r0, p), dot(u_r1, p), dot(u_r2, p)) + u_t;
  vec4 color = vec4(uvec4(d0.w, d0.w >> 8, d0.w >> 16, d0.w >> 24) & 255u) / 255.0;   // RGBA8
  if (c.z <= u_near || color.a < 0.004) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  float s = uintBitsToFloat(d1.x);
  vec2 h0 = unpackHalf2x16(d1.y), h1 = unpackHalf2x16(d1.z), h2 = unpackHalf2x16(d1.w);
  mat3 S = mat3(h0.x, h0.y, h1.x, h0.y, h1.y, h2.x, h1.x, h2.x, h2.y) * s;
  // EWA splatting (as in 3DGS and gaussian/splat.ha): 2D covariance = J W S W^T J^T
  float iz = 1.0 / c.z;
  vec2 xy = clamp(c.xy * iz, -u_lim, u_lim);
  vec3 t0 = u_focal.x * iz * (u_r0 - xy.x * u_r2);
  vec3 t1 = u_focal.y * iz * (u_r1 - xy.y * u_r2);
  vec3 st0 = S * t0;
  float a = dot(t0, st0) + 0.3;
  float b = dot(t1, st0);
  float cc = dot(t1, S * t1) + 0.3;
  float mid = 0.5 * (a + cc);
  float rad = length(vec2(0.5 * (a - cc), b));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.01);
  vec2 e1 = abs(b) > 1e-9 ? normalize(vec2(b, l1 - a)) : (a >= cc ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 e2 = vec2(-e1.y, e1.x);
  float k = sqrt(2.0 * log(255.0 * color.a));     // beyond k sigmas the splat adds less than 1/255
  float maxr = 1.5 * max(u_size.x, u_size.y);
  float r1 = min(k * sqrt(l1), maxr), r2 = min(k * sqrt(l2), maxr);
  vec2 center = u_focal * c.xy * iz + 0.5 * u_size;
  vec2 px = center + a_corner.x * r1 * e1 + a_corner.y * r2 * e2;
  gl_Position = vec4(px.x / u_size.x * 2.0 - 1.0, 1.0 - px.y / u_size.y * 2.0, 0.0, 1.0);
  v_uv = a_corner * vec2(r1 * inversesqrt(l1), r2 * inversesqrt(l2));
  float hgt = (p.y - u_floor.x) * u_floor.y;          // height above the floor, metres
  float ao = mix(1.0, mix(0.55, 1.0, smoothstep(0.0, 0.08, hgt)), u_floor.z);   // darker where it touches
  float bounce = u_floor.z * 0.35 * (1.0 - smoothstep(0.0, 0.3, hgt));        // lit by the floor's colour
  v_color = vec4(min(color.rgb * u_light * ao * mix(vec3(1.0), u_floorTint, bounce), vec3(1.0)), color.a);
}`;

const FS = `#version 300 es
precision mediump float;
in vec4 v_color;
in vec2 v_uv;
out vec4 frag;
void main() {
  float alpha = min(0.99, v_color.a * exp(-0.5 * dot(v_uv, v_uv)));
  if (alpha < 0.004) discard;
  frag = vec4(v_color.rgb * alpha, alpha);         // premultiplied, drawn back to front
}`;

// A disc lying on the floor (world z up): the placement ring and the contact shadow under the scene.
// Clip coordinates keep w = depth, so the GPU clips the parts behind the camera.
const FLOOR_VS = `#version 300 es
precision highp float;
uniform vec3 u_r0, u_r1, u_r2, u_t;   // world -> camera
uniform vec3 u_center;
uniform float u_radius;
uniform vec2 u_focal, u_size;
layout(location = 0) in vec2 a_corner;
out vec2 v_uv;
void main() {
  vec3 w = u_center + vec3(a_corner * u_radius, 0.0);
  vec3 c = vec3(dot(u_r0, w), dot(u_r1, w), dot(u_r2, w)) + u_t;
  gl_Position = vec4(2.0 * u_focal.x * c.x / u_size.x, -2.0 * u_focal.y * c.y / u_size.y, 0.0, c.z);
  v_uv = a_corner;
}`;

const FLOOR_FS = `#version 300 es
precision mediump float;
uniform int u_kind;                   // 0: placement ring, 1: contact shadow
in vec2 v_uv;
out vec4 frag;
void main() {
  float r = length(v_uv);
  if (u_kind == 0) {
    float ring = smoothstep(0.72, 0.78, r) * (1.0 - smoothstep(0.94, 1.0, r));
    float dot0 = 1.0 - smoothstep(0.10, 0.14, r);
    float a = max(ring, dot0) * 0.9;
    frag = vec4(vec3(a), a);
  } else {
    float a = 0.5 * exp(-4.5 * r * r) * (1.0 - smoothstep(0.85, 1.0, r));   // darkest at the base
    frag = vec4(0.0, 0.0, 0.0, a);
  }
}`;

// The camera image in AR, drawn from the frame the floor tracker measured (see ar.js), cropped to
// fill the screen like object-fit: cover.
const CAM_VS = `#version 300 es
precision highp float;
uniform vec2 u_scale;                 // part of the image that shows, per axis
layout(location = 0) in vec2 a_corner;
out vec2 v_uv;
void main() {
  v_uv = 0.5 + vec2(a_corner.x, -a_corner.y) * 0.5 * u_scale;
  gl_Position = vec4(a_corner, 0.0, 1.0);
}`;

const CAM_FS = `#version 300 es
precision mediump float;
uniform sampler2D u_image;
in vec2 v_uv;
out vec4 frag;
void main() {
  frag = vec4(texture(u_image, v_uv).rgb, 1.0);
}`;

// ------------------------------------------------------------------ WebGL setup
const gl = canvas.getContext("webgl2", { antialias: false, alpha: true, premultipliedAlpha: true,
                                         powerPreference: "high-performance", depth: false, stencil: false });
function fail(message) {
  hud.status.textContent = message;
  hud.status.hidden = false;
  throw new Error(message);
}
if (!gl) fail("This browser has no WebGL2. Update iOS (15 or newer) or try another browser.");

function shader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) fail("Shader error: " + gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) fail("Shader link error: " + gl.getProgramInfoLog(prog));
gl.useProgram(prog);
const U = {};
for (const name of ["u_data", "u_r0", "u_r1", "u_r2", "u_t", "u_focal", "u_size", "u_lim", "u_near",
                    "u_light", "u_floor", "u_floorTint"]) {
  U[name] = gl.getUniformLocation(prog, name);
}
const floorProg = gl.createProgram();
gl.attachShader(floorProg, shader(gl.VERTEX_SHADER, FLOOR_VS));
gl.attachShader(floorProg, shader(gl.FRAGMENT_SHADER, FLOOR_FS));
gl.linkProgram(floorProg);
if (!gl.getProgramParameter(floorProg, gl.LINK_STATUS)) fail("Shader link error: " + gl.getProgramInfoLog(floorProg));
const FU = {};
for (const name of ["u_r0", "u_r1", "u_r2", "u_t", "u_center", "u_radius", "u_focal", "u_size", "u_kind"]) {
  FU[name] = gl.getUniformLocation(floorProg, name);
}
const camProg = gl.createProgram();
gl.attachShader(camProg, shader(gl.VERTEX_SHADER, CAM_VS));
gl.attachShader(camProg, shader(gl.FRAGMENT_SHADER, CAM_FS));
gl.linkProgram(camProg);
if (!gl.getProgramParameter(camProg, gl.LINK_STATUS)) fail("Shader link error: " + gl.getProgramInfoLog(camProg));
const CU = { scale: gl.getUniformLocation(camProg, "u_scale"), image: gl.getUniformLocation(camProg, "u_image") };
const camTex = [0, 1].map(() => {
  const t = gl.createTexture();
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.activeTexture(gl.TEXTURE0);
  return t;
});
const camSize = [[0, 0], [0, 0]];

// Copies the camera's current frame into texture slot i (called by ar.js with each tracked frame).
function uploadCameraFrame(i) {
  const v = $("camera");
  if (!v.videoWidth) return;
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, camTex[i]);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, v);
  gl.activeTexture(gl.TEXTURE0);
  camSize[i] = [v.videoWidth, v.videoHeight];
}

function drawCamera(i, w, h) {
  const [vw, vh] = camSize[i];
  if (!vw) return false;
  const k = Math.max(w / vw, h / vh);
  gl.useProgram(camProg);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, camTex[i]);
  gl.uniform1i(CU.image, 1);
  gl.uniform2f(CU.scale, w / (k * vw), h / (k * vh));
  gl.bindVertexArray(floorVao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.activeTexture(gl.TEXTURE0);
  gl.useProgram(prog);
  return true;
}

const floorVao = gl.createVertexArray();
gl.bindVertexArray(floorVao);
const floorBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, floorBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

function drawFloor(world, center, radius, kind, fx, fy, w, h) {
  gl.useProgram(floorProg);
  gl.uniform3fv(FU.u_r0, world.rows.slice(0, 3));
  gl.uniform3fv(FU.u_r1, world.rows.slice(3, 6));
  gl.uniform3fv(FU.u_r2, world.rows.slice(6, 9));
  gl.uniform3fv(FU.u_t, world.t);
  gl.uniform3fv(FU.u_center, center);
  gl.uniform1f(FU.u_radius, radius);
  gl.uniform2f(FU.u_focal, fx, fy);
  gl.uniform2f(FU.u_size, w, h);
  gl.uniform1i(FU.u_kind, kind);
  gl.bindVertexArray(floorVao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.useProgram(prog);
}

const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const cornerBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
const orderBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, orderBuf);
gl.enableVertexAttribArray(1);
gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
gl.vertexAttribDivisor(1, 1);
const tex = gl.createTexture();
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, tex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
gl.uniform1i(U.u_data, 0);
gl.disable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
const renderer = (() => {
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
})();

// ------------------------------------------------------------------ worker + scenes
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let scene = null;               // { n, header }
let drawCount = 0;
let sortInFlight = false;
let sortedFor = null;           // camera key of the last sort request
let lastSortMs = 0;
let loadId = 0;
let engineName = "";
let dirty = true;                // something changed since the last drawn frame
const ar = new LookAroundAR($("camera"));
let lightMatch = true;                   // AR: draw the scene in the room's light (or as captured)
ar.frameSink = uploadCameraFrame;
if (new URLSearchParams(location.search).has("debug")) {                  // for tests
  window.ar = ar;
  window.viewerScene = () => scene;
}

worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "loaded" && m.id === loadId) {
    engineName = m.engine;
    hud.engine.textContent = m.engine;
    hud.engine.title = m.why || "";
    const n = m.header.count;
    const rows = Math.ceil(n / 2048);
    const data = new Uint32Array(4096 * 4 * rows);
    data.set(m.gpu);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, 4096, rows, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, data);
    scene = { n, header: m.header, decodeMs: m.decodeMs, stats: m.stats };
    ar.sceneChanged(m.header);
    drawCount = 0;
    sortedFor = null;
    sortInFlight = false;
    resetView();
    setStatus("");
    hud.bar.style.width = "0%";
  } else if (m.type === "sorted" && m.id === loadId) {
    gl.bindBuffer(gl.ARRAY_BUFFER, orderBuf);
    gl.bufferData(gl.ARRAY_BUFFER, m.order, gl.DYNAMIC_DRAW);
    drawCount = m.order.length;
    lastSortMs = m.ms;
    sortInFlight = false;
    dirty = true;
  } else if (m.type === "error") {
    setStatus("Could not read this scene: " + m.message);
  }
};

function setStatus(text) {
  hud.status.textContent = text;
  hud.status.hidden = !text;
}

async function loadScene(entry) {
  const id = ++loadId;
  scene = null;
  drawCount = 0;
  hud.scene.textContent = entry.title;
  document.querySelectorAll(".scene").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.name === entry.name)));
  setStatus(`Loading ${entry.title} (${(entry.bytes / 1e6).toFixed(1)} MB)`);
  try {
    const r = await fetch(entry.file);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const total = entry.bytes;
    const reader = r.body.getReader();
    const file = new Uint8Array(entry.bytes);
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (got + value.length > file.length) throw new Error("scene file is larger than expected");
      file.set(value, got);
      got += value.length;
      if (id !== loadId) return;
      hud.bar.style.width = `${Math.min(100, (got / total) * 100).toFixed(1)}%`;
    }
    // scene files are WebAssembly data modules: the .hspl bytes start at entry.offset
    const offset = entry.offset || 0;
    const buf = file.slice(offset, offset + (entry.length || got - offset));
    setStatus(`Decoding ${entry.splats.toLocaleString()} splats`);
    worker.postMessage({ type: "load", id, buffer: buf.buffer }, [buf.buffer]);
  } catch (e) {
    if (id === loadId) setStatus(`Could not download ${entry.title}: ${e.message}`);
  }
}

// ------------------------------------------------------------------ camera
const cam = { target: [0, 0, 0], yaw: 0, pitch: 0.3, dist: 3, fovy: 50 * Math.PI / 180 };
let lastInput = performance.now();
let autoRotate = true;

function resetView() {
  if (!scene) return;
  const h = scene.header;
  const f = h.front;
  const dir = [f[0], f[1] + 0.39, f[2]];
  const len = Math.hypot(...dir);
  cam.target = [0, 0, 0];
  cam.yaw = Math.atan2(dir[0], dir[2]);
  cam.pitch = Math.asin(dir[1] / len);
  // the packed view distance frames the scene in a 50 degree field of view; a portrait phone
  // screen is narrower than that, so step back until the scene fits across it
  const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
  const tanY = Math.tan(cam.fovy / 2), tanX = tanY * aspect;
  cam.dist = h.distance * len * Math.max(1, tanY / tanX);
  lastInput = performance.now();
  dirty = true;
}

function viewRows() {
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const d = [cp * Math.sin(cam.yaw), sp, cp * Math.cos(cam.yaw)];
  const eye = [cam.target[0] + d[0] * cam.dist, cam.target[1] + d[1] * cam.dist, cam.target[2] + d[2] * cam.dist];
  const fwd = [-d[0], -d[1], -d[2]];
  let right = [fwd[1] * 0 - fwd[2] * 1, fwd[2] * 0 - fwd[0] * 0, fwd[0] * 1 - fwd[1] * 0];   // fwd x up(0,1,0)
  const rl = Math.hypot(...right) || 1;
  right = right.map((v) => v / rl);
  const down = [fwd[1] * right[2] - fwd[2] * right[1], fwd[2] * right[0] - fwd[0] * right[2],
                fwd[0] * right[1] - fwd[1] * right[0]];
  const t = [-(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]),
             -(down[0] * eye[0] + down[1] * eye[1] + down[2] * eye[2]),
             -(fwd[0] * eye[0] + fwd[1] * eye[1] + fwd[2] * eye[2])];
  return { r0: right, r1: down, r2: fwd, t };
}

// ------------------------------------------------------------------ input: drag to orbit, pinch to zoom, two fingers to pan
const pointers = new Map();
let tap = null;                         // a possible tap: one finger that barely moves
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  tap = pointers.size === 1 ? { x: e.clientX, y: e.clientY, t: performance.now(), moved: 0 } : null;
  lastInput = performance.now();
});
canvas.addEventListener("pointermove", (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y;
  if (tap) tap.moved += Math.hypot(dx, dy);
  if (ar.on && pointers.size === 1) {
    if (!tap || tap.moved > 12) ar.turn += dx * 0.01;   // AR: drag turns the scene where it stands
  } else if (pointers.size === 1) {
    cam.yaw -= dx * 0.006;
    cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy * 0.006));
  } else if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const other = a === p ? b : a;
    const before = Math.hypot(p.x - other.x, p.y - other.y);
    const after = Math.hypot(e.clientX - other.x, e.clientY - other.y);
    if (ar.on) {                            // AR: two fingers slide it along the floor, pinch resizes, twist turns
      if (before > 0 && after > 0) ar.zoom = Math.max(0.2, Math.min(5, ar.zoom * after / before));
      ar.turn -= Math.atan2(e.clientY - other.y, e.clientX - other.x) - Math.atan2(p.y - other.y, p.x - other.x);
      const r = canvas.getBoundingClientRect();
      const k = canvas.width / r.width;
      const f = ar.focal(canvas.width, canvas.clientWidth, canvas.clientHeight);
      const mid = (x, y) => [((x + other.x) / 2 - r.left) * k, ((y + other.y) / 2 - r.top) * k];
      const [x0, y0] = mid(p.x, p.y), [x1, y1] = mid(e.clientX, e.clientY);
      ar.slide(x0, y0, x1, y1, canvas.width, canvas.height, f);
      p.x = e.clientX;
      p.y = e.clientY;
      return;
    }
    if (before > 0 && after > 0) cam.dist = Math.max(0.05, Math.min(200, cam.dist * before / after));
    const v = viewRows();
    const k = cam.dist * 0.0012;
    for (let i = 0; i < 3; i++) cam.target[i] -= (v.r0[i] * dx + v.r1[i] * dy) * k * 0.5;
  }
  p.x = e.clientX;
  p.y = e.clientY;
  lastInput = performance.now();
  dirty = true;
});
const up = (e) => {
  pointers.delete(e.pointerId);
  lastInput = performance.now();
  if (ar.on && scene && tap && tap.moved <= 12 && performance.now() - tap.t < 500 && e.type === "pointerup") {
    const r = canvas.getBoundingClientRect();
    const k = canvas.width / r.width;
    const f = ar.focal(canvas.width, canvas.clientWidth, canvas.clientHeight);
    const x = (e.clientX - r.left) * k, y = (e.clientY - r.top) * k;
    if (!ar.place(x, y, canvas.width, canvas.height, f, scene.header)) arHint("Tap on the floor (point the phone down a bit).");
    dirty = true;
  }
  tap = null;
};
canvas.addEventListener("pointerup", up);
canvas.addEventListener("pointercancel", up);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  cam.dist = Math.max(0.05, Math.min(200, cam.dist * Math.exp(e.deltaY * 0.0012)));
  lastInput = performance.now();
  dirty = true;
}, { passive: false });
canvas.addEventListener("dblclick", () => resetView());

// ------------------------------------------------------------------ quality: resolution and splat budget
const dpr = Math.min(window.devicePixelRatio || 1, 3);
const quality = { mode: "auto", scale: Math.min(dpr, 1.5), budget: 1 };
const LIMITS = { minScale: 0.5, maxScale: Math.min(dpr, 2), minBudget: 0.2 };
let fpsWindow = { start: performance.now(), frames: 0, fps: 60, frameMs: 16.7 };

const MODES = { auto: "Auto", sharp: "Sharp", fast: "Fast" };
function setMode(mode) {
  quality.mode = mode;
  if (mode === "sharp") { quality.scale = LIMITS.maxScale; quality.budget = 1; }
  if (mode === "fast") { quality.scale = Math.max(LIMITS.minScale, Math.min(1, dpr)); quality.budget = 0.35; }
  if (mode === "auto") { quality.scale = Math.min(dpr, 1.5); quality.budget = 1; }
  $("mode").textContent = `Quality: ${MODES[mode]}`;
  sortedFor = null;
  dirty = true;
}

// The display or the browser may cap the frame rate (Low Power Mode caps Safari at 30 fps). Lowering
// the quality can't beat a cap, so each step down is a trial: if the fps doesn't rise within two
// seconds, the step is undone and the fps it stays at is taken as the cap.
let trial = null;
let cap = 0;
function adapt(fps) {
  if (quality.mode !== "auto" || !scene) return;
  const now = performance.now();
  if (trial && now - trial.time > 2000) {
    if (fps < trial.fps * 1.1) {
      quality.scale = trial.scale;
      quality.budget = trial.budget;
      cap = Math.max(cap, trial.fps);
      sortedFor = null;
      dirty = true;
    }
    trial = null;
    return;
  }
  if (trial) return;
  const low = cap ? 0.85 * cap : 42, high = cap ? 0.95 * cap : 57;
  if (fps < low) {                          // too slow: fewer pixels first, then fewer splats
    trial = { fps, scale: quality.scale, budget: quality.budget, time: now };
    if (quality.scale > Math.max(LIMITS.minScale, 0.75 * Math.min(dpr, 1.5))) quality.scale = Math.max(LIMITS.minScale, quality.scale - 0.15);
    else if (quality.budget > LIMITS.minBudget) quality.budget = Math.max(LIMITS.minBudget, quality.budget * 0.8);
    else quality.scale = Math.max(LIMITS.minScale, quality.scale - 0.1);
    sortedFor = null;
    dirty = true;
  } else if (fps > high) {                 // headroom: all splats first, then sharper
    if (quality.budget < 1) { quality.budget = Math.min(1, quality.budget * 1.15); sortedFor = null; }
    else if (quality.scale < LIMITS.maxScale) quality.scale = Math.min(LIMITS.maxScale, quality.scale + 0.1);
    dirty = true;
  }
}

// ------------------------------------------------------------------ frame loop
function resize() {
  const w = Math.max(1, Math.round(canvas.clientWidth * quality.scale));
  const h = Math.max(1, Math.round(canvas.clientHeight * quality.scale));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    dirty = true;
  }
  return [w, h];
}

function clearColor() {
  const c = getComputedStyle(document.documentElement).getPropertyValue("--canvas").trim();
  const m = c.match(/^#?([0-9a-f]{6})$/i);
  if (!m) return [0.93, 0.94, 0.95];
  const v = parseInt(m[1], 16);
  return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255];
}
let bg = clearColor();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { bg = clearColor(); dirty = true; });

let prev = performance.now();
let hudTimer = 0;
let lastDrawn = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - prev) / 1000);
  prev = now;
  if (autoRotate && !ar.on && pointers.size === 0 && now - lastInput > 2500) { cam.yaw += dt * 0.25; dirty = true; }
  if (ar.on) dirty = true;                   // the phone is always moving a little
  const [w, h] = resize();
  if (!dirty) {                              // nothing changed: don't redraw (saves the battery)
    updateHud(now, w, h);
    requestAnimationFrame(frame);
    return;
  }
  dirty = false;
  lastDrawn = now;
  gl.viewport(0, 0, w, h);
  if (ar.on) gl.clearColor(0, 0, 0, 0);     // transparent: the camera image shows through
  else gl.clearColor(bg[0], bg[1], bg[2], 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (ar.on && ar.shownSlot >= 0) drawCamera(ar.shownSlot, w, h);    // the frame the pose is for
  const arView = ar.on && scene ? ar.view(scene.stats, w, canvas.clientWidth, canvas.clientHeight) : null;
  if (arView) {
    const f = arView.focal;
    if (arView.reticle) drawFloor(arView.world, arView.reticle, arView.reticleRadius, 0, f, f, w, h);
    if (arView.shadow) drawFloor(arView.world, arView.shadow.center, arView.shadow.radius, 1, f, f, w, h);
    const st = ar.trackState();
    if (arView.draw) {
      arHint(st === "tracking" ? "Walk around it · drag to turn · two fingers to slide and resize"
           : st === "found" ? "Found its place again"
           : st === "recovering" ? "Finding its place… show the floor you saw before"
           : st === "rotation" ? "Drag to turn it · pinch to resize · tap the floor to move it"
           : "Lost the floor: point the phone at it and move slowly");
    } else {
      arHint(!arView.reticle ? "Point the phone at the floor"
           : st === "searching" ? "Tap to put it on the floor · finding floor texture…"
           : "Tap to put it on the floor");
    }
  } else if (ar.on) {
    arHint(scene ? "Waiting for the motion sensors…" : "Loading the scene…");
  }
  if (scene && (!ar.on || (arView && arView.draw))) {
    const v = arView ? arView.draw : viewRows();
    const vs = arView ? arView.sort : v;
    const fy = arView ? arView.focal : 0.5 * h / Math.tan(cam.fovy / 2);
    const fx = fy;
    const tanX = 0.5 * w / fx, tanY = 0.5 * h / fy;
    const near = arView ? arView.sort.near : cam.dist * 0.01;
    const budget = Math.max(1, Math.round(scene.n * quality.budget));
    const key = [...vs.r0, ...vs.r1, ...vs.r2, ...vs.t, tanX, tanY, budget].map((x) => x.toFixed(4)).join(",");
    if (!sortInFlight && key !== sortedFor) {
      sortInFlight = true;
      sortedFor = key;
      worker.postMessage({ type: "sort", id: loadId, budget,
                           camera: { rot: [...vs.r0, ...vs.r1, ...vs.r2], t: vs.t, tanX, tanY, near } });
    }
    if (drawCount) {
      gl.uniform3fv(U.u_r0, v.r0);
      gl.uniform3fv(U.u_r1, v.r1);
      gl.uniform3fv(U.u_r2, v.r2);
      gl.uniform3fv(U.u_t, v.t);
      gl.uniform2f(U.u_focal, fx, fy);
      gl.uniform2f(U.u_size, w, h);
      gl.uniform2f(U.u_lim, 1.3 * tanX, 1.3 * tanY);
      gl.uniform1f(U.u_near, arView ? 0.03 : near);
      const lit = arView && lightMatch ? arView.light : null;
      gl.uniform3fv(U.u_light, lit ? lit.gain : [1, 1, 1]);
      gl.uniform3fv(U.u_floorTint, lit ? lit.floorTint : [1, 1, 1]);
      gl.uniform3f(U.u_floor, scene.stats.ground, arView ? arView.scale : 1, lit ? 1 : 0);
      gl.bindVertexArray(vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, drawCount);
    }
  }
  fpsWindow.frames++;
  if (now - fpsWindow.start > 1000) {
    fpsWindow.fps = fpsWindow.frames * 1000 / (now - fpsWindow.start);
    fpsWindow.frameMs = (now - fpsWindow.start) / fpsWindow.frames;
    fpsWindow.start = now;
    fpsWindow.frames = 0;
    adapt(fpsWindow.fps);
  }
  updateHud(now, w, h);
  requestAnimationFrame(frame);
}

function updateHud(now, w, h) {
  if (now - lastDrawn > 1500) {              // idle: no frames to measure
    fpsWindow.start = now;
    fpsWindow.frames = 0;
  }
  if (now - hudTimer > 250) {
    hudTimer = now;
    const idle = now - lastDrawn > 1500;
    hud.fps.textContent = idle ? "idle" : fpsWindow.fps.toFixed(0);
    hud.frame.textContent = idle ? "–" : fpsWindow.frameMs.toFixed(1);
    hud.sort.textContent = lastSortMs ? lastSortMs.toFixed(1) : "–";
    hud.drawn.textContent = scene ? `${drawCount.toLocaleString()} / ${scene.n.toLocaleString()}` : "–";
    hud.res.textContent = `${w}×${h}`;
    const t = ar.on && ar.track;
    hud.track.hidden = !ar.on;
    if (ar.on) {
      hud.track.textContent = ar.trackError ? "floor tracking off (rotation only)"
        : t ? `floor: ${t.inliers || t.points} points, ${t.ms.toFixed(1)} ms · camera ${Math.round(1000 / (ar.frameMs || 33))} fps, ${Math.round(t.lag)} ms late · light ${lightMatch ? ar.lighting().exposure.toFixed(2) + "×" : "original"}`
        : "floor: starting";
    }
    hud.quality.textContent = quality.mode === "auto"
      ? `${Math.round(quality.budget * 100)}% of splats · ${quality.scale.toFixed(2)}× resolution`
      : quality.mode === "sharp" ? "all splats · full resolution" : "35% of splats · low resolution";
  }
}

// ------------------------------------------------------------------ AR
function arHint(text) {
  const el = $("ar-hint");
  if (el.textContent !== text) el.textContent = text;
  el.hidden = !text;
}

function arNote(text, link) {
  $("ar-where").textContent = text;
  const a = $("ar-link");
  a.hidden = !link;
  if (link) { a.href = link; a.textContent = link.replace("https://", ""); }
  $("ar-note").hidden = false;
}

async function toggleAR() {
  if (ar.on) {
    ar.stop();
    arHint("");
    document.body.classList.remove("ar");
    $("ar").textContent = "AR";
    $("ar").setAttribute("aria-pressed", "false");
    dirty = true;
    return;
  }
  const why = LookAroundAR.blocked();
  if (why === "frame") {
    arNote("This page is shown inside another site (here: claude.ai), and that blocks the camera. " +
           "Open the AR version on its own:", AR_URL);
    return;
  }
  if (why === "insecure") { arNote("The camera needs a secure (https) page.", AR_URL); return; }
  if (why) { arNote("This browser can't give a web page the camera and the motion sensors. Use Safari on an iPhone."); return; }
  if (!scene) return;
  try {
    await ar.start();
  } catch (e) {
    ar.stop();
    arNote(`AR couldn't start: ${e.message || e}. In Settings > Safari you can allow Camera and Motion & Orientation access.`);
    return;
  }
  $("ar-note").hidden = true;
  document.body.classList.add("ar");
  $("ar").textContent = "Exit AR";
  $("ar").setAttribute("aria-pressed", "true");
  dirty = true;
}

// ------------------------------------------------------------------ UI
async function start() {
  let list = [];
  try {
    list = await (await fetch("./scenes.json")).json();
  } catch (e) {
    setStatus("Could not load the scene list.");
    return;
  }
  const rail = $("scenes");
  for (const entry of list) {
    const b = document.createElement("button");
    b.className = "scene";
    b.dataset.name = entry.name;
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<span class="scene-title"></span><span class="scene-meta"></span>`;
    b.querySelector(".scene-title").textContent = entry.title;
    b.querySelector(".scene-meta").textContent = `${(entry.splats / 1000).toFixed(0)}k · ${(entry.bytes / 1e6).toFixed(1)} MB`;
    b.addEventListener("click", () => loadScene(entry));
    rail.appendChild(b);
  }
  $("mode").addEventListener("click", () => {
    const next = { auto: "sharp", sharp: "fast", fast: "auto" }[quality.mode];
    setMode(next);
  });
  $("spin").addEventListener("click", () => {
    autoRotate = !autoRotate;
    $("spin").setAttribute("aria-pressed", String(autoRotate));
    dirty = true;
  });
  $("reset").addEventListener("click", resetView);
  $("ar").addEventListener("click", toggleAR);
  $("light").addEventListener("click", () => {
    lightMatch = !lightMatch;
    $("light").textContent = lightMatch ? "Light: room" : "Light: original";
    $("light").setAttribute("aria-pressed", String(lightMatch));
  });
  $("ar-close").addEventListener("click", () => { $("ar-note").hidden = true; });
  $("gpu").textContent = renderer.length > 40 ? renderer.slice(0, 38) + "…" : renderer;
  $("gpu").title = renderer;
  setMode("auto");
  const first = list.find((e) => e.name === (location.hash.slice(1) || "unicorn")) || list[0];
  if (new URLSearchParams(location.search).has("still")) {     // for screenshots: no turntable
    autoRotate = false;
    $("spin").setAttribute("aria-pressed", "false");
  }
  if (first) loadScene(first);
  requestAnimationFrame(frame);
}
start();
