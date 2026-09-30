// Web Worker: floor tracking off the main thread, in the HA++ module track.wasm (see tracker.mjs).
import { FloorTracker } from "./tracker.mjs";

let tracker = null;
let failed = "";
const ready = FloorTracker.create(new URL("./track.wasm", import.meta.url))
  .then((t) => { tracker = t; })
  .catch((e) => { failed = String(e && e.message || e); });

self.onmessage = async (ev) => {
  const m = ev.data;
  await ready;
  if (!tracker) {
    self.postMessage({ type: "error", message: "floor tracking is unavailable: " + failed });
    return;
  }
  if (m.type === "reset") {
    tracker.reset();
  } else if (m.type === "frame") {
    const r = tracker.frame(new Uint8Array(m.rgba), m.w, m.h, m.f, m.time, m.gyro, m.height);
    self.postMessage({ type: "pose", state: r.state, C: r.C, T: r.T, points: r.points, inliers: r.inliers,
                       ms: r.ms, lag: r.lag });
  }
};
