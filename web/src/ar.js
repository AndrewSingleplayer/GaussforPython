// Look-around AR without WebXR (Safari on iPhone has no WebXR).
//
// What a web page can use on an iPhone is enough for this:
//   - the camera image (getUserMedia), shown behind the transparent WebGL canvas;
//   - the phone's orientation (DeviceOrientation: the fused gyroscope/accelerometer/compass angles),
//     which turns the virtual camera, so the scene stays put in the room when you turn the phone.
// This tracks rotation only (3 degrees of freedom). Walking doesn't change your distance to the
// scene yet; that needs visual tracking of the floor (the next step, see web/README.md).

const D2R = Math.PI / 180;
// iPhone main (wide) camera: 26 mm equivalent focal length -> about 67 degrees across the long
// side of the image. Safari's getUserMedia uses this camera for facingMode "environment".
const FOV_LONG = 67 * D2R;

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

// Horizontal direction the camera looks in (unit 2D vector in the world's x-y plane).
export function heading(C) {
  const fh = [C[2], C[5]];                             // camera forward = third column of C
  const len = Math.hypot(fh[0], fh[1]);
  return len > 1e-3 ? [fh[0] / len, fh[1] / len] : [0, 1];
}

// Where the scene goes: `dist` along the horizontal direction `fh`, a bit below eye level, turned
// so its front faces the viewer. Returns world-from-scene as p_world = T + S p_scene. The world is
// the orientation frame (z up); scenes are y up.
export function placement(fh, front, dist, turn = 0) {
  const T = [fh[0] * dist, fh[1] * dist, -0.15 * dist];     // about 8 degrees below eye level
  const M = [1, 0, 0, 0, 0, -1, 0, 1, 0];              // scene (x, y, z) -> world (x, -z, y)
  const fw = mulv(M, front);
  const want = Math.atan2(-fh[1], -fh[0]);             // from the scene toward the camera
  const have = Math.atan2(fw[1], fw[0]);
  return { T, S: mul(rotZ(want - have + turn), M) };
}

export class LookAroundAR {
  constructor(video) {
    this.video = video;
    this.on = false;
    this.R = null;
    this.fh = null;                                    // where the scene was placed (set on the first view)
    this.turn = 0;                                     // extra turn of the scene (drag)
    this.zoom = 1;                                     // distance factor (pinch)
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
    this.fh = null;
    this.turn = 0;
    this.zoom = 1;
  }

  stop() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    window.removeEventListener("deviceorientation", this.onOrient);
    this.video.srcObject = null;
    this.video.hidden = true;
    this.on = false;
    this.R = null;
    this.fh = null;
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

  // View for scene coordinates: rows of scene -> camera, translation, focal length. Null until the
  // first orientation reading arrives.
  view(header, canvasW, cssW, cssH) {
    if (!this.R) return null;
    const C = cameraToWorld(this.R, this.screenAngle());
    const f = this.focal(canvasW, cssW, cssH);
    if (!this.fh) {                                    // first reading: the scene goes where the phone looks, once
      this.fh = heading(C);
      const tanX = 0.5 * canvasW / f;                  // fit the scene across the screen, as the viewer does
      this.base = 1.3 * header.distance * Math.max(1, Math.tan(25 * D2R) / tanX);
    }
    const place = placement(this.fh, header.front, this.base * this.zoom, this.turn);
    const W2C = transpose(C);
    const rot = mul(W2C, place.S);
    return { r0: rot.slice(0, 3), r1: rot.slice(3, 6), r2: rot.slice(6, 9), t: mulv(W2C, place.T), focal: f };
  }
}
