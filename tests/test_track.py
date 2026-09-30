"""Floor tracking for the web AR mode (web/track.ha + web/src/tracker.mjs), on simulated walks.

Each test renders what a phone camera sees while its holder moves around a spot on the floor
(tests/track_sim.py), runs the tracker in Node on the frames with the WebAssembly module, and checks
the anchor error: how far from the real floor spot a point placed at the start is drawn, in pixels
of the 202 x 360 tracking image (about 2.4 screen points each on an iPhone).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import track_sim  # noqa: E402

HAVE_TOOLS = shutil.which("node") and shutil.which("wasm-ld")


@unittest.skipUnless(HAVE_TOOLS, "needs node and wasm-ld")
class FloorTracking(unittest.TestCase):
    def check(self, res, median, worst, lost=0):
        self.assertLessEqual(res["lost_frames"], lost, res)
        self.assertLess(res["anchor_px_median"], median, res)
        self.assertLess(res["anchor_px_max"], worst, res)

    def test_walk_around_a_spot(self):
        # 150 degrees around the spot, 4 m of walking, hand shake and steps
        res = track_sim.simulate(seconds=8, arc=150)
        self.check(res, 1.0, 3.0)
        self.assertLess(res["position_err_end_m"], 0.05)

    def test_quick_turns_with_a_late_camera(self):
        # two 70-degree turns at up to 275 degrees/s while the camera image arrives 100 ms after the
        # motion sensors (the tracker starts assuming 60 ms and has to measure it)
        res = track_sim.simulate(turns=70, latency=0.1)
        self.check(res, 1.5, 10.0)
        self.assertAlmostEqual(res["lag_ms_end"], 100, delta=15)

    def test_wood_floor_rolling_shutter_boxes(self):
        # stripes (few corners), rows read out over 20 ms, boxes standing on the floor
        res = track_sim.simulate(floor="planks", readout_ms=20, turns=40, boxes=True)
        self.check(res, 3.0, 10.0)

    def test_light_stats(self):
        # the light measurement (web/track.ha) on a frame with a warm left half and a known box
        w, h = 64, 48
        with tempfile.TemporaryDirectory() as tmp:
            wasm = track_sim.build_wasm(tmp)
            script = f"""
                import fs from "node:fs";
                const {{ instance }} = await WebAssembly.instantiate(fs.readFileSync({json.dumps(wasm)}), {{}});
                const ex = instance.exports, base = Math.ceil(Number(ex.__heap_base.value) / 16) * 16;
                ex.memory.grow(4);
                const px = new Uint8Array(ex.memory.buffer, base, {4 * w * h});
                for (let y = 0; y < {h}; y++) for (let x = 0; x < {w}; x++) {{
                    const i = 4 * (y * {w} + x);
                    px.set(x < {w // 2} ? [200, 150, 100, 255] : [100, 100, 100, 255], i);
                }}
                const out = base + {4 * w * h};
                ex.light_stats(base, {w}, {h}, 0, 0, 16, 16, out);
                console.log(JSON.stringify(Array.from(new Float32Array(ex.memory.buffer, out, 7))));
            """
            res = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
        r = json.loads(res.stdout)
        self.assertEqual([round(v, 3) for v in r[:3]], [150, 125, 100])     # half warm, half grey
        self.assertEqual([round(v, 3) for v in r[3:6]], [200, 150, 100])    # the box is in the warm half
        self.assertEqual(r[6], 64)                                          # every second pixel of 16 x 16

    def test_wrong_height_and_lens(self):
        # the phone is 1.15 m above the floor, not 1.35, and the lens sees 72 degrees, not 67:
        # the scene is drawn a little bigger than planned but stays on its spot
        res = track_sim.simulate(true_height=1.15, true_fov=72)
        self.check(res, 3.0, 6.0)


if __name__ == "__main__":
    unittest.main()
