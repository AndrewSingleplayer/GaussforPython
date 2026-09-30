// AR without WebXR (Safari on iPhone has no WebXR): the floor is found and followed by the page.
//
// What a web page can use on an iPhone is enough for this:
//   - the camera image (getUserMedia), shown behind the transparent WebGL canvas;
//   - the phone's orientation (DeviceOrientation: fused gyroscope, accelerometer and compass),
//     which gives the camera's rotation and, with it, which way is down.
// The floor is the horizontal plane `height` metres below the phone when AR starts. A ring shows
// where the middle of the screen meets the floor; a tap puts the scene there, standing on the floor
// at a real size. Floor tracking (tracker.mjs, with the HA++ module track.wasm, in a worker)
// follows points on the floor in the camera image, so the scene stays on its spot when you walk
// around it, not just when you turn. Without it (no WebAssembly), rotation is still tracked.
//
// World frame: x east, y north, z up, metres; the camera starts at the origin; the floor is
// z = -height. Scenes are y up (web/pack.py).
//
// Timing: the tracker's pose is for the camera frame it was given, which is already one frame old
// when the answer comes back (the screen shows the next one by then). So the scene is drawn with
// the pose brought forward to the frame on screen: the gyroscope's turning between the two frames
// (read with the camera delay the tracker measured), and the tracked velocity for the position.

import { orthonormalize } from "./tracker.mjs";

const D2R = Math.PI / 180;
// iPhone main (wide) camera: 26 mm equivalent focal length -> about 67 degrees across the long
// side of the image. Safari's getUserMedia uses this camera for facingMode "environment".
const FOV_LONG = 67 * D2R;
const PHONE_HEIGHT = 1.35;       // metres from the floor to a phone held in front of you
const SCENE_HEIGHT = 0.7;        // metres: how tall a scene stands when placed (pinch changes it)
const TRACK_LONG = 360;          // long side of the image the floor tracker works on, pixels

function mul(a, b) {               // 3x3, row-major
  const r = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  }
  return r;
}
const mulv = (a, v) => [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
                        a[6] * v[0] + a[7] * v[1] + a[8] * v[2]];
const transpose = (a) => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
const rotZ = (t) => [Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t), 0, 0, 0, 1];
const M_SCENE = [1, 0, 0, 0, 0, -1, 0, 1, 0];     // scene (x, y, z) -> world (x, -z, y): y up -> z up

// Earth-from-device rotation from DeviceOrientation angles (W3C: intrinsic Z-X'-Y'', degrees).
// Device axes: x to the right of the screen (portrait), y to the top, z out of the screen.
export function deviceRotation(alpha, beta, gamma) {
  const a = alpha * D2R, b = beta * D2R, g = gamma * D2R;
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cg = Math.cos(g), sg = Math.sin(g);
  return [ca * cg - sa * sb * sg, -sa * cb, ca * sg + sa * sb * cg,
          sa * cg + ca * sb * sg, ca * cb, sa * sg - ca * sb * cg,
          -cb * sg, sb, cb * cg];
}

// World-from-camera rotation. Camera axes as in the renderer: x right, y down, z forward (out of
// the back of the phone, where the camera looks). `screenAngle` is the screen's rotation in degrees.
export function cameraToWorld(R, screenAngle) {
  const Rs = mul(R, rotZ(-screenAngle * D2R));        // axes of the screen as the user holds it
  return mul(Rs, [1, 0, 0, 0, -1, 0, 0, 0, -1]);
}

// Where a ray from the camera at T meets the floor (z = -height), or null when it points too high
// or too far.
export function floorHit(C, dirCam, height, maxDist = 8, T = [0, 0, 0]) {
  const d = mulv(C, dirCam);
  if (d[2] > -0.05) return null;
  const t = (-height - T[2]) / d[2];
  if (t <= 0 || t * Math.hypot(d[0], d[1]) > maxDist) return null;
  return [T[0] + t * d[0], T[1] + t * d[1], -height];
}

// Rotation of a scene standing at floor point P: upright, its front facing the camera at T, plus a
// user turn about the vertical.
export function standOnFloor(P, front, turn, T = [0, 0, 0]) {
  const fw = mulv(M_SCENE, front);
  const want = Math.atan2(T[1] - P[1], T[0] - P[0]);  // from the scene toward the camera
  const have = Math.atan2(fw[1], fw[0]);
  return mul(rotZ(want - have + turn), M_SCENE);
}

export class LookAroundAR {
  constructor(video) {
    this.video = video;
    this.on = false;
    this.R = null;
    this.height = PHONE_HEIGHT;
    this.track = null;               // the floor tracker's last answer
    this.trackError = "";
    this.readings = [];
    this.history = [];               // motion sensor readings of the last 1.5 s: { t, C }
    this.shownTime = 0;              // time of the camera frame on screen
    this.velocity = [0, 0, 0];       // metres per ms, from the tracker
    this.reset();
  }

  reset() {
    this.placed = null;              // floor point where the scene stands
    this.S = null;                   // its rotation (fixed when placed)
    this.turn = 0;                   // extra turn about the vertical (drag)
    this.zoom = 1;                   // size factor (pinch)
  }

  // Why AR can't start here, or null.
  static blocked() {
    if (window.self !== window.top) return "frame";
    if (!window.isSecureContext) return "insecure";
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return "camera";
    if (typeof DeviceOrientationEvent === "undefined") return "motion";
    return null;
  }

  // Call from a tap: iOS only asks for motion access inside a user gesture.
  async start() {
    if (typeof DeviceOrientationEvent.requestPermission === "function") {
      const answer = await DeviceOrientationEvent.requestPermission();
      if (answer !== "granted") throw new Error("Motion access was not allowed. Allow it to use AR.");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 },
                             frameRate: { ideal: 60 } } });   // 60: half the time between frames, less delay
    this.video.srcObject = this.stream;
    this.video.hidden = false;
    await this.video.play();
    this.readings = [];              // motion sensor readings not yet sent to the tracker
    this.onOrient = (e) => {
      if (e.alpha === null || e.beta === null || e.gamma === null) return;
      this.R = deviceRotation(e.alpha, e.beta, e.gamma);
      const reading = { t: e.timeStamp || performance.now(), C: cameraToWorld(this.R, this.screenAngle()) };
      this.readings.push(reading);
      if (this.readings.length > 120) this.readings.shift();
      this.history.push(reading);
      while (this.history.length > 2 && this.history[0].t < reading.t - 1500) this.history.shift();
    };
    window.addEventListener("deviceorientation", this.onOrient);
    this.on = true;
    this.reset();
    this.startTracking();
  }

  stop() {
    this.on = false;
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    window.removeEventListener("deviceorientation", this.onOrient);
    this.video.srcObject = null;
    this.video.hidden = true;
    this.R = null;
    this.track = null;
    if (this.worker) this.worker.postMessage({ type: "reset" });
    this.reset();
  }

  // ---------------------------------------------------------------- floor tracking
  startTracking() {
    this.track = null;
    this.busy = false;
    if (!this.worker) {
      try {
        this.worker = new Worker(new URL("./track-worker.js", import.meta.url), { type: "module" });
      } catch (e) {
        this.trackError = String(e && e.message || e);
        return;
      }
      this.worker.onmessage = (ev) => this.onTrack(ev.data);
      this.worker.onerror = (e) => { this.trackError = e.message || "the tracking worker failed"; this.busy = false; };
    }
    this.worker.postMessage({ type: "reset" });
    this.grab = this.grab || document.createElement("canvas");
    this.grabCtx = this.grabCtx || this.grab.getContext("2d", { willReadFrequently: true });
    const next = () => {
      if (!this.on) return;
      const frame = (now) => {
        if (this.shownTime) this.frameMs = 0.9 * (this.frameMs || now - this.shownTime) + 0.1 * (now - this.shownTime);
        this.shownTime = now;
        this.sendFrame(now);
        next();
      };
      if (this.video.requestVideoFrameCallback) this.video.requestVideoFrameCallback(frame);
      else setTimeout(() => frame(performance.now()), 33);
    };
    next();
  }

  // Hands the camera frame now showing to the tracker, with the sensor readings since the last one.
  sendFrame(time) {
    if (!this.on || this.busy || this.trackError || !this.readings.length) return;
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!vw || !vh) return;
    const k = TRACK_LONG / Math.max(vw, vh);
    const w = Math.round(vw * k), h = Math.round(vh * k);
    if (this.grab.width !== w || this.grab.height !== h) {
      this.grab.width = w;
      this.grab.height = h;
    }
    this.grabCtx.drawImage(this.video, 0, 0, w, h);
    const img = this.grabCtx.getImageData(0, 0, w, h);
    const f = 0.5 * Math.max(w, h) / Math.tan(FOV_LONG / 2);
    const gyro = this.readings;
    this.readings = [];
    this.busy = true;
    this.worker.postMessage({ type: "frame", rgba: img.data.buffer, w, h, f, time, gyro, height: this.height },
                            [img.data.buffer]);
  }

  onTrack(m) {
    this.busy = false;
    if (m.type === "error") { this.trackError = m.message; return; }
    if (m.type !== "pose" || !this.on || !m.C) return;
    const prev = this.track;
    if (prev && prev.state === "tracking" && m.state === "tracking" && m.time > prev.time) {
      const dt = m.time - prev.time;
      this.velocity = this.velocity.map((v, i) => 0.5 * v + 0.5 * (m.T[i] - prev.T[i]) / dt);
    } else if (m.state !== "tracking") {
      this.velocity = [0, 0, 0];
    }
    this.track = m;
  }

  // The gyroscope's camera rotation at time t (ms), between the two nearest readings.
  gyroAt(t) {
    const g = this.history;
    if (!g.length) return null;
    if (t <= g[0].t) return g[0].C;
    if (t >= g[g.length - 1].t) return g[g.length - 1].C;
    let i = g.length - 1;
    while (i > 0 && g[i - 1].t > t) i--;
    const a = g[i - 1], b = g[i];
    const k = (t - a.t) / Math.max(1e-6, b.t - a.t);
    return orthonormalize(a.C.map((x, j) => x + k * (b.C[j] - x)));
  }

  // How far (ms) the frame on screen is ahead of the frame the tracker answered for.
  ahead() {
    return this.track ? Math.max(0, Math.min(150, this.shownTime - this.track.time)) : 0;
  }

  // Tracking state for the hints and the readout: "tracking", "searching" (not enough floor
  // texture yet), "lost", "rotation" (no tracker: rotation only).
  trackState() {
    if (this.trackError) return "rotation";
    if (!this.track) return "searching";
    return this.track.state === "started" ? "tracking" : this.track.state;
  }

  // ---------------------------------------------------------------- camera
  screenAngle() {
    if (screen.orientation && typeof screen.orientation.angle === "number") return screen.orientation.angle;
    return typeof window.orientation === "number" ? window.orientation : 0;
  }

  // Focal length in canvas pixels that matches the camera image as shown (object-fit: cover).
  focal(canvasW, cssW, cssH) {
    const vw = this.video.videoWidth || 720, vh = this.video.videoHeight || 1280;
    const k = Math.max(cssW / vw, cssH / vh);
    const fVideo = 0.5 * Math.max(vw, vh) / Math.tan(FOV_LONG / 2);
    return fVideo * k * (canvasW / cssW);
  }

  // Camera -> world rotation: the tracker's (for the camera frame on screen) or the gyroscope's.
  camera() {
    if (this.track) {
      const lag = this.track.lag ?? 60;
      const g0 = this.gyroAt(this.track.time - lag), g1 = this.gyroAt(this.track.time + this.ahead() - lag);
      if (!g0 || !g1) return this.track.C;
      return orthonormalize(mul(this.track.C, mul(transpose(g0), g1)));
    }
    return this.R ? cameraToWorld(this.R, this.screenAngle()) : null;
  }

  position() {
    if (!this.track) return [0, 0, 0];
    const dt = this.ahead();
    return this.track.T.map((x, i) => x + this.velocity[i] * dt);
  }

  // Put the scene where the tap at canvas pixel (x, y) meets the floor. Returns false if it doesn't.
  place(x, y, w, h, f, header) {
    const C = this.camera();
    if (!C) return false;
    const T = this.position();
    const len = Math.hypot((x - w / 2) / f, (y - h / 2) / f, 1);
    const P = floorHit(C, [(x - w / 2) / f / len, (y - h / 2) / f / len, 1 / len], this.height, 8, T);
    if (!P) return false;
    this.placed = P;
    this.S = standOnFloor(P, header.front, 0, T);
    this.turn = 0;
    return true;
  }

  // Another scene: it stands where the last one stood, facing the camera.
  sceneChanged(header) {
    if (this.placed) {
      this.S = standOnFloor(this.placed, header.front, 0, this.position());
      this.turn = 0;
    }
  }

  // Scale from scene units to metres.
  scale(stats) {
    return this.zoom * SCENE_HEIGHT / Math.max(1e-6, stats.top - stats.ground);
  }

  // What to draw this frame:
  //   focal; world: world -> camera (for the floor ring and the shadow); reticle, reticleRadius: the
  //   floor point in the middle of the screen before placing, and the size of the floor the scene
  //   will cover; draw: scene -> camera in metres; sort: the same in scene units (with near);
  //   shadow: {center, radius} on the floor.
  view(stats, canvasW, cssW, cssH) {
    const C = this.camera();
    if (!C) return null;
    const T = this.position();
    const f = this.focal(canvasW, cssW, cssH);
    const W2C = transpose(C);
    const out = { focal: f, world: { rows: W2C, t: mulv(W2C, T).map((v) => -v) }, reticle: null };
    const s = this.scale(stats);
    const footprint = 1.1 * s * stats.radius;          // radius of the floor the scene covers
    if (!this.placed) {
      out.reticle = floorHit(C, [0, 0, 1], this.height, 8, T);
      out.reticleRadius = Math.max(0.1, Math.min(0.6, footprint));
      return out;
    }
    const S = mul(rotZ(this.turn), this.S);
    const rot = mul(W2C, S);                            // scene -> camera rotation
    const base = mulv(S, [stats.cx, stats.ground, stats.cz]);   // the middle of the scene's base, turned
    const P = this.placed;
    const t = mulv(W2C, [P[0] - s * base[0] - T[0], P[1] - s * base[1] - T[1], P[2] - s * base[2] - T[2]]);
    const rows = (k) => rot.slice(3 * k, 3 * k + 3);
    out.draw = { r0: rows(0).map((v) => v * s), r1: rows(1).map((v) => v * s), r2: rows(2).map((v) => v * s), t };
    out.sort = { r0: rows(0), r1: rows(1), r2: rows(2), t: t.map((v) => v / s), near: 0.03 / s };
    out.shadow = { center: P, radius: 1.3 * footprint };
    return out;
  }
}
