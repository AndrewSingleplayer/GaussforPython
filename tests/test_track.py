"""Floor tracking for the web AR mode (web/track.ha + web/src/tracker.mjs), on simulated walks.

Each test renders what a phone camera sees while its holder moves around a spot on the floor
(tests/track_sim.py), runs the tracker in Node on the frames with the WebAssembly module, and checks
the anchor error: how far from the real floor spot a point placed at the start is drawn, in pixels
of the 202 x 360 tracking image (about 2.4 screen points each on an iPhone).
"""
import os
import shutil
import sys
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

    def test_wrong_height_and_lens(self):
        # the phone is 1.15 m above the floor, not 1.35, and the lens sees 72 degrees, not 67:
        # the scene is drawn a little bigger than planned but stays on its spot
        res = track_sim.simulate(true_height=1.15, true_fov=72)
        self.check(res, 3.0, 6.0)


if __name__ == "__main__":
    unittest.main()
