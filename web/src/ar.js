// AR without WebXR (Safari on iPhone has no WebXR), with the floor found from gravity.
//
// What a web page can use on an iPhone is enough for this:
//   - the camera image (getUserMedia), shown behind the transparent WebGL canvas;
//   - the phone's orientation (DeviceOrientation: fused gyroscope, accelerometer and compass),
//     which gives the camera's rotation and, with it, which way is down.
// The floor is the horizontal plane `height` metres below the phone. A ring shows where the middle
// of the screen meets the floor; a tap puts the scene there, standing on the floor at a real size.
// Rotation is tracked; walking is not yet (floor tracking with the camera image is the next step,
// see research/04-phones-and-webar.md), so the scene keeps its place while you turn, not while
// you walk.
//
// World frame: the orientation frame (x east, y north, z up), camera at the origin, metres.
// Scenes are y up (web/pack.py).

const D2R = Math.PI / 180;
// iPhone main (wide) camera: 26 mm equivalent focal length -> about 67 degrees across the long
// side of the image. Safari's getUserMedia uses this camera for facingMode "environment".
const FOV_LONG = 67 * D2R;
const PHONE_HEIGHT = 1.35;       // metres from the floor to a phone held in front of you
const SCENE_HEIGHT = 0.7;        // metres: how tall a scene stands when placed (pinch changes it)

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

// Where a camera ray meets the floor (z = -height), or null when it points too high or too far.
export function floorHit(C, dirCam, height, maxDist = 8) {
  const d = mulv(C, dirCam);
  if (d[2] > -0.05) return null;
  const t = height / -d[2];
  if (t * Math.hypot(d[0], d[1]) > maxDist) return null;
  return [t * d[0], t * d[1], -height];
}

// Rotation of a scene standing at floor point P: upright, its front facing the camera (at the
// origin), plus a user turn about the vertical.
export function standOnFloor(P, front, turn) {
  const fw = mulv(M_SCENE, front);
  const want = Math.atan2(-P[1], -P[0]);              // from the scene toward the camera
  const have = Math.atan2(fw[1], fw[0]);
  return mul(rotZ(want - have + turn), M_SCENE);
}

export class LookAroundAR {
  constructor(video) {
    this.video = video;
    this.on = false;
    this.R = null;
    this.height = PHONE_HEIGHT;
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
      audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } } });
    this.video.srcObject = this.stream;
    this.video.hidden = false;
    await this.video.play();
    this.onOrient = (e) => {
      if (e.alpha !== null && e.beta !== null && e.gamma !== null) this.R = deviceRotation(e.alpha, e.beta, e.gamma);
    };
    window.addEventListener("deviceorientation", this.onOrient);
    this.on = true;
    this.reset();
  }

  stop() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    window.removeEventListener("deviceorientation", this.onOrient);
    this.video.srcObject = null;
    this.video.hidden = true;
    this.on = false;
    this.R = null;
    this.reset();
  }

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

  camera() {
    return this.R ? cameraToWorld(this.R, this.screenAngle()) : null;
  }

  // Put the scene where the tap at canvas pixel (x, y) meets the floor. Returns false if it doesn't.
  place(x, y, w, h, f, header) {
    const C = this.camera();
    if (!C) return false;
    const len = Math.hypot((x - w / 2) / f, (y - h / 2) / f, 1);
    const P = floorHit(C, [(x - w / 2) / f / len, (y - h / 2) / f / len, 1 / len], this.height);
    if (!P) return false;
    this.placed = P;
    this.S = standOnFloor(P, header.front, 0);
    this.turn = 0;
    return true;
  }

  // Another scene: it stands where the last one stood, facing the camera.
  sceneChanged(header) {
    if (this.placed) {
      this.S = standOnFloor(this.placed, header.front, 0);
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
    const f = this.focal(canvasW, cssW, cssH);
    const W2C = transpose(C);
    const out = { focal: f, world: { rows: W2C, t: [0, 0, 0] }, reticle: null };
    const s = this.scale(stats);
    const footprint = 1.1 * s * stats.radius;          // radius of the floor the scene covers
    if (!this.placed) {
      out.reticle = floorHit(C, [0, 0, 1], this.height);
      out.reticleRadius = Math.max(0.1, Math.min(0.6, footprint));
      return out;
    }
    const S = mul(rotZ(this.turn), this.S);
    const rot = mul(W2C, S);                            // scene -> camera rotation
    const base = mulv(S, [stats.cx, stats.ground, stats.cz]);   // the middle of the scene's base, turned
    const P = this.placed;
    const t = mulv(W2C, [P[0] - s * base[0], P[1] - s * base[1], P[2] - s * base[2]]);
    const rows = (k) => rot.slice(3 * k, 3 * k + 3);
    out.draw = { r0: rows(0).map((v) => v * s), r1: rows(1).map((v) => v * s), r2: rows(2).map((v) => v * s), t };
    out.sort = { r0: rows(0), r1: rows(1), r2: rows(2), t: t.map((v) => v / s), near: 0.03 / s };
    out.shadow = { center: P, radius: 1.3 * footprint };
    return out;
  }
}
