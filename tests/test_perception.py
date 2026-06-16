"""Unit tests for tools/depth_perception.py free-space scoring.

The fast path (free_space) is exercised on hand-built depth arrays so the
wall-vs-open-floor discrimination is deterministic and model-free. The heuristic
fallback is checked on a real repo frame. A slow end-to-end regression against
the actual Depth Anything model is opt-in (CAR_ROBOT_RUN_MODEL_TESTS=1) so the
default suite stays fast.

Run:  CAR_ROBOT_NO_VENV_REEXEC=1 .venv/bin/python -m unittest discover -s tests
"""
import os
import sys
import unittest

os.environ.setdefault("CAR_ROBOT_NO_VENV_REEXEC", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "tools"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import depth_perception as dp  # noqa: E402


def synthetic_depth(near_zone=None, h=100, w=90, far=10.0, near=100.0):
    """A depth map (larger == nearer, Depth Anything convention).

    Baseline = far. The immediate-floor reference band is always near. If
    near_zone is given (slice of columns), that slab of the look-ahead band is
    also near == an obstacle as close as our own floor in that zone.
    """
    d = np.full((h, w), far, dtype=np.float32)
    d[int(h * 0.82):int(h * 0.98), int(w * 0.33):int(w * 0.67)] = near  # floor reference
    if near_zone is not None:
        a, b = near_zone
        d[int(h * 0.55):int(h * 0.78), a:b] = near                      # obstacle ahead
    return d


class FreeSpaceTests(unittest.TestCase):
    def test_output_shape_and_range(self):
        out = dp.free_space(synthetic_depth())
        self.assertEqual(set(out), {"left", "center", "right"})
        for v in out.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_open_floor_reads_open(self):
        out = dp.free_space(synthetic_depth())  # nothing near ahead
        self.assertGreater(out["center"], 0.9)
        self.assertGreater(out["left"], 0.9)
        self.assertGreater(out["right"], 0.9)

    def test_obstacle_dead_ahead_blocks_only_center(self):
        out = dp.free_space(synthetic_depth(near_zone=(30, 60)))  # center third near
        self.assertLess(out["center"], 0.2)
        self.assertGreater(out["left"], 0.8)
        self.assertGreater(out["right"], 0.8)

    def test_open_center_beats_blocked_center(self):
        opened = dp.free_space(synthetic_depth())["center"]
        blocked = dp.free_space(synthetic_depth(near_zone=(30, 60)))["center"]
        self.assertGreater(opened, blocked)

    def test_debug_view_matches_free_space(self):
        d = synthetic_depth(near_zone=(30, 60))
        dbg = dp.free_space_debug(d)
        fs = dp.free_space(d)
        for zone in ("left", "center", "right"):
            self.assertAlmostEqual(dbg["zones"][zone]["openness"], fs[zone], places=3)


class HeuristicTests(unittest.TestCase):
    def test_heuristic_on_repo_frame(self):
        path = os.path.join(_REPO, "open_floor.jpg")
        if not os.path.exists(path):
            self.skipTest("open_floor.jpg not present")
        import reflex_drive as rd
        out = rd.heuristic_free_space(Image.open(path).convert("RGB"))
        self.assertEqual(set(out), {"left", "center", "right"})
        for v in out.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


class DepthModelRegressionTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("CAR_ROBOT_RUN_MODEL_TESTS") == "1",
        "slow; set CAR_ROBOT_RUN_MODEL_TESTS=1 to run the real depth-model regression",
    )
    def test_open_floor_more_open_than_near_wall(self):
        model_dir = os.path.join(_REPO, "models", "da2-small")
        if not os.path.isdir(model_dir):
            self.skipTest("models/da2-small not downloaded")
        try:
            import torch  # noqa: F401
        except Exception as e:
            self.skipTest(f"torch unavailable: {e}")
        floor_p = os.path.join(_REPO, "open_floor.jpg")
        wall_p = os.path.join(_REPO, "wall25.jpg")
        if not (os.path.exists(floor_p) and os.path.exists(wall_p)):
            self.skipTest("reference frames missing")
        dp.load(model_id=model_dir)
        floor = dp.free_space(dp.depth(Image.open(floor_p).convert("RGB")))["center"]
        wall = dp.free_space(dp.depth(Image.open(wall_p).convert("RGB")))["center"]
        # documents the known behaviour: a near wall reads less open than open floor
        self.assertGreaterEqual(floor, wall)


if __name__ == "__main__":
    unittest.main(verbosity=2)
