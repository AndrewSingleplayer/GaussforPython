// Runs the floor tracker on raw grey frames (for tests/track_sim.py).
//   node tests/track_run.mjs frames.bin meta.json track.wasm out.json
import fs from "node:fs";
import { FloorTracker } from "../web/src/tracker.mjs";

const [framesPath, metaPath, wasmPath, outPath] = process.argv.slice(2);
const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
const frames = fs.readFileSync(framesPath);
const { w, h, f, n } = meta;
const tracker = await FloorTracker.create(fs.readFileSync(wasmPath));
const rgba = new Uint8Array(4 * w * h);
const out = [];
for (let k = 0; k < n; k++) {
  const g = frames.subarray(k * w * h, (k + 1) * w * h);
  for (let i = 0; i < w * h; i++) {
    rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = g[i];
    rgba[4 * i + 3] = 255;
  }
  const r = tracker.frame(rgba, w, h, f, meta.times[k], meta.gyro[k], meta.height);
  out.push({ state: r.state, C: r.C, T: r.T, Craw: r.Craw, Traw: r.Traw, points: r.points, inliers: r.inliers,
             ms: r.ms, lag: r.lag, followed: r.followed, fix: r.shiftFix });
}
fs.writeFileSync(outPath, JSON.stringify(out));
