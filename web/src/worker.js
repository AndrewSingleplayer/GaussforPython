// Web Worker: decodes scenes and sorts splats off the main thread, so drawing never waits.
// The work runs in the HA++ WebAssembly module (splatweb.wasm) through engine.mjs.
import { createEngine } from "./engine.mjs";

const ready = createEngine(new URL("./splatweb.wasm", import.meta.url));
let engine = null;

self.onmessage = async (ev) => {
  const msg = ev.data;
  engine = engine || await ready;
  if (msg.type === "load") {
    try {
      const r = engine.load(msg.buffer);
      self.postMessage({ type: "loaded", id: msg.id, header: r.header, gpu: r.gpu, decodeMs: r.decodeMs,
                         stats: r.stats, engine: engine.kind, why: engine.why || "" }, [r.gpu.buffer]);
    } catch (e) {
      self.postMessage({ type: "error", id: msg.id, message: String(e && e.message || e) });
    }
  } else if (msg.type === "sort") {
    const t0 = performance.now();
    const order = engine.sort(msg.camera, msg.budget);
    self.postMessage({ type: "sorted", id: msg.id, order, ms: performance.now() - t0 }, [order.buffer]);
  }
};
