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
// Camera image and scene in step: the page draws the camera image itself (WebGL), and it draws the
// very frame the tracker measured, with that frame's pose (like ARKit apps do). Safari's own
// camera preview (the <video>) can show a newer frame than the one a pose is for; any guess about
// how much newer shows as lag or jiggle. So each frame handed to the tracker is also copied into a
// texture (frameSink), and it is shown when the tracker's answer for it comes back. The <video>
// shows only until then, or when there is no tracker.
//
// Light: a capture carries the light of the room it was filmed in, usually bright and neutral, so
// in a dim or warm room it looks pasted in. Like ARKit's light estimation, the tracker measures the
// camera frame's brightness and colour cast, and the colour of the floor under the scene; the
// scene is drawn darker or brighter, tinted the same way, darkened where it touches the floor and
// lit a little by the floor's colour near it (viewer.js).
const LIGHT_REF = 0.42;          // mean brightness (0-1) of a normally exposed camera frame

import { insideHull } from "./tracker.mjs";

const D2R = Math.PI / 180;
// iPhone main (wide) camera: 26 mm equivalent focal length -> about 67 degrees across the long
// side of the image. Safari's getUserMedia uses this camera for facingMode "environment".
const FOV_LONG = 67 * D2R;
const PHONE_HEIGHT = 1.35;       // metres from the floor to a phone held in front of you
const SCENE_HEIGHT = 0.7;        // metres: how tall a scene stands when placed on the floor
const TABLE_HEIGHT = 0.3;        // and on a table (pinch changes both)
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
    this.motionEvents = 0;
    this.startedAt = 0;
    this.frameSink = null;           // (slot) => copies the video's current frame into texture slot
    this.shownSlot = -1;             // camera texture to draw (-1: none yet, the <video> shows)
    this.pendingSlot = -1;           // texture holding the frame the tracker is working on
    this.lastFrame = 0;
    this.light = null;               // smoothed { frame: [r, g, b], floor: [r, g, b] }, 0-1
    this.footprint = 0.3;            // metres: radius of the floor the scene covers
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
    this.motionEvents = 0;
    this.startedAt = performance.now();
    this.frameError = "";
    this.onOrient = (e) => {
      if (e.alpha === null || e.beta === null || e.gamma === null) return;
      this.motionEvents++;
      this.R = deviceRotation(e.alpha, e.beta, e.gamma);
      this.readings.push({ t: e.timeStamp || performance.now(), C: cameraToWorld(this.R, this.screenAngle()) });
      if (this.readings.length > 120) this.readings.shift();
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
    this.shownSlot = this.pendingSlot = -1;
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
        if (this.lastFrame) this.frameMs = 0.9 * (this.frameMs || now - this.lastFrame) + 0.1 * (now - this.lastFrame);
        this.lastFrame = now;
        if (this.trackError && this.frameSink) {          // no tracker: show every frame as it comes
          this.frameSink(0);
          this.shownSlot = 0;
        }
        try {                          // one bad frame mustn't stop the frames that follow
          this.sendFrame(now);
        } catch (e) {
          this.frameError = String((e && e.message) || e);
          this.busy = false;
        }
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
    const box = this.floorBox(w, h, f);
    if (this.frameSink) {                                  // keep this frame to show with its pose
      this.pendingSlot = this.shownSlot === 0 ? 1 : 0;
      this.frameSink(this.pendingSlot);
    }
    this.worker.postMessage({ type: "frame", rgba: img.data.buffer, w, h, f, time, gyro, height: this.height, box },
                            [img.data.buffer]);
  }

  onTrack(m) {
    this.busy = false;
    if (m.type === "error") { this.trackError = m.message; return; }
    if (m.type !== "pose" || !this.on) return;
    if (this.pendingSlot >= 0) this.shownSlot = this.pendingSlot;   // the frame this answer is for
    this.pendingSlot = -1;
    if (m.C) {
      if (this.track && m.relocs > (this.track.relocs || 0)) this.foundAt = performance.now();
      this.track = m;
    }
    if (m.light) {
      const k = this.light ? 0.08 : 1;                      // about 0.4 s to follow a change of light
      const mix = (a, b) => a.map((v, i) => v + k * (b[i] / 255 - v));
      const zero = [0, 0, 0];
      this.light = { frame: mix(this.light ? this.light.frame : zero, m.light.frame),
                     floor: m.light.floor ? mix(this.light && this.light.floor || zero, m.light.floor)
                                          : this.light && this.light.floor };
    }
  }

  // The box of the tracking image where the floor under the scene is, or null.
  floorBox(w, h, f) {
    if (!this.placed || !this.track) return null;
    const C = this.camera(), T = this.position(), P = this.placed;
    const c = mulv(transpose(C), [P[0] - T[0], P[1] - T[1], P[2] - T[2]]);
    if (c[2] < 0.2) return null;
    const u = f * c[0] / c[2] + (w - 1) / 2, v = f * c[1] / c[2] + (h - 1) / 2;
    const r = Math.max(4, f * 1.4 * this.footprint / c[2]);
    return [u - r, v - 0.6 * r, u + r, v + 0.6 * r];
  }

  // How to draw the scene to match the room's light: gain (r, g, b) for its colours, and the floor's
  // colour cast for the light it bounces near the floor.
  lighting() {
    const l = this.light;
    if (!l) return { gain: [1, 1, 1], floorTint: [1, 1, 1], exposure: 1 };
    const luma = (c) => Math.max(1e-3, 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]);
    const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
    const L = luma(l.frame);
    const exposure = clamp((L / LIGHT_REF) ** 0.75, 0.45, 1.2);
    const gain = l.frame.map((c) => exposure * clamp(1 + 0.5 * (c / L - 1), 0.8, 1.25));
    const floorTint = l.floor ? l.floor.map((c) => clamp(c / luma(l.floor), 0.6, 1.5)) : [1, 1, 1];
    return { gain, floorTint, exposure };
  }

  // Tracking state for the hints and the readout: "tracking", "searching" (not enough floor
  // texture yet), "lost", "recovering" (tracking again, but not yet sure where: looking for the
  // place in the floor memory), "found" (just found it again), "rotation" (no tracker).
  trackState() {
    if (this.trackError) return "rotation";
    if (!this.track) return "searching";
    if (this.foundAt && performance.now() - this.foundAt < 1500) return "found";
    if (this.track.verified === false) return "recovering";
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

  // Camera -> world rotation: the tracker's (for the camera frame drawn) or the gyroscope's.
  camera() {
    if (this.track) return this.track.C;
    return this.R ? cameraToWorld(this.R, this.screenAngle()) : null;
  }

  position() {
    return this.track ? this.track.T : [0, 0, 0];
  }

  // Surfaces the tracker found above the floor (tables): [{ z, hull }].
  planes() {
    return (this.track && this.track.planes) || [];
  }

  // Where a camera ray meets a surface: the nearest table it falls on, else the floor.
  // Returns { P, table } or null.
  surfaceHit(C, dirCam, T, maxDist = 8) {
    const d = mulv(C, dirCam);
    let best = null, bestS = Infinity;
    for (const p of this.planes()) {
      if (d[2] > -0.05 || p.z > T[2] - 0.1) continue;
      const s = (p.z - T[2]) / d[2];
      const X = [T[0] + s * d[0], T[1] + s * d[1], p.z];
      if (s > 0 && s < bestS && insideHull(p.hull, X[0], X[1], 0.03)) { best = X; bestS = s; }
    }
    if (best) return { P: best, table: true };
    const P = floorHit(C, dirCam, this.height, maxDist, T);
    return P ? { P, table: false } : null;
  }

  dirAt(x, y, w, h, f) {
    const len = Math.hypot((x - w / 2) / f, (y - h / 2) / f, 1);
    return [(x - w / 2) / f / len, (y - h / 2) / f / len, 1 / len];
  }

  // Where the ray through canvas pixel (x, y) meets a surface (a table or the floor), or null.
  floorAt(x, y, w, h, f, maxDist = 8) {
    const C = this.camera();
    if (!C) return null;
    const hit = this.surfaceHit(C, this.dirAt(x, y, w, h, f), this.position(), maxDist);
    return hit && hit.P;
  }

  // Put the scene on the floor: the first time on the ring (the middle of the screen), wherever the
  // tap was; afterwards where the tap was. Returns false if that isn't floor.
  place(x, y, w, h, f, header) {
    if (!this.placed) { x = w / 2; y = h / 2; }
    const P = this.floorAt(x, y, w, h, f);
    if (!P) return false;
    const T = this.position();
    this.onTable = P[2] > -this.height + 0.1;
    this.placed = P;
    this.S = standOnFloor(P, header.front, 0, T);
    this.turn = 0;
    return true;
  }

  // Two fingers moved from (x0, y0) to (x1, y1) (canvas pixels): the scene slides along the floor
  // by as much as the floor under the fingers did.
  slide(x0, y0, x1, y1, w, h, f) {
    if (!this.placed) return;
    const C = this.camera(), T = this.position(), z = this.placed[2];
    const on = (x, y) => floorHit(C, this.dirAt(x, y, w, h, f), -z, 30, T);    // the surface it stands on
    const a = on(x0, y0), b = on(x1, y1);
    if (!a || !b) return;
    this.placed = [this.placed[0] + b[0] - a[0], this.placed[1] + b[1] - a[1], this.placed[2]];
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
    return this.zoom * (this.onTable ? TABLE_HEIGHT : SCENE_HEIGHT) / Math.max(1e-6, stats.top - stats.ground);
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
    const hit = this.placed ? null : this.surfaceHit(C, [0, 0, 1], T);
    if (!this.placed) this.onTable = !!(hit && hit.table);   // the ring shows the size it will have there
    const s = this.scale(stats);
    const footprint = 1.1 * s * stats.radius;          // radius of the floor the scene covers
    this.footprint = footprint;
    if (!this.placed) {
      out.reticle = hit && hit.P;
      out.reticleOnTable = !!(hit && hit.table);
      out.reticleRadius = Math.max(0.1, Math.min(0.6, footprint));
      out.planes = this.planes();
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
    out.scale = s;
    out.light = this.lighting();
    return out;
  }
}
