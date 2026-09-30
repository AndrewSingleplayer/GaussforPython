"""The web viewer's data path, end to end: pack (web/pack.py) -> decode and sort in the browser
engine (web/splatweb.ha compiled to WebAssembly, and the JavaScript fallback in
web/src/engine.mjs), run in Node and checked against NumPy.

Uses a generated scene, so it needs no download. Needs node and wasm-ld.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "web"), os.path.join(ROOT, "gaussian")):
    sys.path.insert(0, p)

from happ.driver import build  # noqa: E402
from happ.targets import Toolchain  # noqa: E402

HAVE = bool(shutil.which("node") and Toolchain().find("wasm-ld"))

NODE = r"""
import fs from "node:fs";
import { createEngine } from "ENGINE";
const dir = process.argv[2];
const file = fs.readFileSync(dir + "/scene.hspl");
const buf = () => file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
const cams = JSON.parse(fs.readFileSync(dir + "/cams.json"));
const out = {};
for (const [name, engine] of [["wasm", await createEngine(fs.readFileSync(dir + "/splatweb.wasm"))],
                              ["js", await createEngine(null, { forceJs: true })]]) {
  const r = engine.load(buf());
  fs.writeFileSync(`${dir}/${name}_gpu.bin`, new Uint8Array(r.gpu.buffer));
  out[name] = { kind: engine.kind, orders: [] };
  cams.forEach((cam, k) => {
    const order = engine.sort(cam, cam.budget);
    fs.writeFileSync(`${dir}/${name}_order${k}.bin`, new Uint8Array(order.buffer));
    out[name].orders.push(order.length);
  });
}
console.log(JSON.stringify(out));
"""


def camera(eye, target, tan_x, tan_y, near, budget=None):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    cam = {"rot": R.ravel().tolist(), "t": (-R @ eye).tolist(), "tanX": tan_x, "tanY": tan_y, "near": near}
    if budget:
        cam["budget"] = budget
    return cam


def decode_numpy(data):
    """Reference decoder for the .hspl format, in float64."""
    count = int.from_bytes(data[8:12], "little")
    lo = np.frombuffer(data, "<f4", 3, 16).astype(np.float64)
    size = np.frombuffer(data, "<f4", 3, 28).astype(np.float64)
    lmin, lrange = (float(v) for v in np.frombuffer(data, "<f4", 2, 40))
    rec = np.frombuffer(data, np.dtype([("rgba", "u1", 4), ("pos", "<u2", 3), ("scale", "u1", 3),
                                        ("rot", "u1", 3)]), count, 64)
    pos = lo + rec["pos"].astype(np.float64) * (size / 65535.0)
    s = np.exp(lmin + rec["scale"].astype(np.float64) * (lrange / 255.0))
    x, y, z = (rec["rot"].astype(np.float64) / 127.5 - 1.0).T
    w = np.sqrt(np.maximum(0.0, 1 - x * x - y * y - z * z))
    norm = np.sqrt(w * w + x * x + y * y + z * z)            # > 1 when quantization pushed |xyz| past 1
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                  np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                  np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    M = R * s[:, None, :]
    cov = M @ np.transpose(M, (0, 2, 1))
    return pos, cov, rec["rgba"].view("<u4").ravel(), 3 * s.max(1)


@unittest.skipUnless(HAVE, "needs node and wasm-ld")
class WebEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pack
        import render
        import scenes
        cls.tmp = tempfile.mkdtemp(prefix="happ-web-")
        wasm = build(os.path.join(ROOT, "web", "splatweb.ha"), ["web-wasm32"], cls.tmp, quiet=True,
                     bridges=False)["web-wasm32"]
        shutil.copyfile(wasm, os.path.join(cls.tmp, "splatweb.wasm"))
        raw = render.synthetic_scene(20000, seed=3)
        raw[:50, 10] = -9.0                                  # invisible splats: dropped by the packer
        s = scenes.Scene(raw[:, 0:3], raw[:, 3:6], raw[:, 6:10], raw[:, 10], raw[:, None, 11:14])
        cls.data = pack.pack_scene(s, np.zeros(3), 4.0, (0.0, 1.0, 0.0))
        with open(os.path.join(cls.tmp, "scene.hspl"), "wb") as f:
            f.write(cls.data)
        cls.n = int.from_bytes(cls.data[8:12], "little")
        cls.cams = [camera([0.3, 1.8, 3.2], [0, 0, 0], 0.4, 0.5, 0.05),
                    camera([0.0, 0.3, 1.1], [0.5, 0.0, 0.0], 0.3, 0.6, 0.05),          # inside the scene
                    camera([2.5, 0.8, -2.0], [0, 0, 0], 0.45, 0.45, 0.05, budget=6000)]
        with open(os.path.join(cls.tmp, "cams.json"), "w") as f:
            json.dump(cls.cams, f)
        script = os.path.join(cls.tmp, "run.mjs")
        with open(script, "w") as f:
            f.write(NODE.replace("ENGINE", "file://" + os.path.join(ROOT, "web", "src", "engine.mjs")))
        r = subprocess.run(["node", script, cls.tmp], capture_output=True, text=True, check=True)
        cls.summary = json.loads(r.stdout)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def gpu(self, engine):
        g = np.fromfile(os.path.join(self.tmp, f"{engine}_gpu.bin"), np.uint32).reshape(self.n, 8)
        pos = g[:, 0:3].copy().view(np.float32)
        scale = g[:, 4].copy().view(np.float32)
        halves = g[:, 5:8].copy().view(np.float16).astype(np.float64).reshape(self.n, 6)
        return pos, g[:, 3], scale, halves

    def order(self, engine, k):
        return np.fromfile(os.path.join(self.tmp, f"{engine}_order{k}.bin"), np.uint32)

    def test_the_webassembly_engine_ran(self):
        self.assertEqual(self.summary["wasm"]["kind"], "HA++ WebAssembly")
        self.assertEqual(self.n, 20000 - 50)

    def test_decode_matches_numpy(self):
        pos_ref, cov_ref, rgba_ref, _ = decode_numpy(self.data)
        for engine in ("wasm", "js"):
            with self.subTest(engine=engine):
                pos, rgba, scale, c = self.gpu(engine)
                np.testing.assert_allclose(pos, pos_ref, rtol=0, atol=2e-6)
                np.testing.assert_array_equal(rgba, rgba_ref)
                full = c * scale[:, None].astype(np.float64)
                ref = np.stack([cov_ref[:, 0, 0], cov_ref[:, 0, 1], cov_ref[:, 0, 2], cov_ref[:, 1, 1],
                                cov_ref[:, 1, 2], cov_ref[:, 2, 2]], 1)
                big = np.abs(ref).max(1, keepdims=True)
                # the covariance is stored as f16 after dividing by the splat's largest variance:
                # every entry is exact to f16 precision relative to that (about 5e-4)
                self.assertLess(np.max(np.abs(full - ref) / big), 1e-3)

    def test_sort_is_back_to_front_and_culls(self):
        pos_ref, _, _, radius = decode_numpy(self.data)
        for k, cam in enumerate(self.cams):
            R = np.array(cam["rot"]).reshape(3, 3)
            pc = pos_ref @ R.T + np.array(cam["t"])
            n = cam.get("budget", self.n)
            kx, ky = np.sqrt(1 + cam["tanX"] ** 2), np.sqrt(1 + cam["tanY"] ** 2)
            margin_x = np.abs(pc[:, 0]) - cam["tanX"] * pc[:, 2] - radius * kx
            margin_y = np.abs(pc[:, 1]) - cam["tanY"] * pc[:, 2] - radius * ky
            visible = (pc[:, 2] > cam["near"]) & (margin_x < 0) & (margin_y < 0)
            visible[n:] = False
            unsure = (np.abs(margin_x) < 1e-4) | (np.abs(margin_y) < 1e-4) | (np.abs(pc[:, 2] - cam["near"]) < 1e-4)
            for engine in ("wasm", "js"):
                with self.subTest(camera=k, engine=engine):
                    order = self.order(engine, k)
                    self.assertEqual(len(np.unique(order)), len(order))
                    got = np.zeros(self.n, bool)
                    got[order] = True
                    self.assertFalse(np.any((got != visible) & ~unsure), "visible set differs from the model")
                    z = pc[order, 2]
                    step = (z.max() - z.min()) / 65535 * 1.01
                    self.assertTrue(np.all(np.diff(z) <= step), "not sorted back to front")
            self.assertGreater(visible.sum(), 100)

    def test_scene_container_is_valid_webassembly(self):
        import build as web_build
        payload = self.data[:5000]
        wrapped, offset = web_build.wasm_container(payload)
        self.assertEqual(wrapped[offset:], payload)
        path = os.path.join(self.tmp, "c.wasm")
        with open(path, "wb") as f:
            f.write(wrapped)
        js = ("const b = require('fs').readFileSync(process.argv[1]);"
              "const ok = WebAssembly.validate(b);"
              "const m = new WebAssembly.Instance(new WebAssembly.Module(b)).exports.memory;"
              "console.log(ok, Buffer.from(m.buffer, 0, 4).toString());")
        r = subprocess.run(["node", "-e", js, path], capture_output=True, text=True, check=True)
        self.assertEqual(r.stdout.split(), ["true", "HSPL"])


if __name__ == "__main__":
    unittest.main()
