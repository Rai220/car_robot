"""Unit tests for the layer-1 controller decision logic in tools/reflex_drive.py.

These are pure-function tests — no car, no network, no ML model — so they run in
milliseconds and pin down the *intent* of the fast loop:
  * steering points TOWARD the more open side (script convention; firmware sign
    is a separate hardware calibration, see the left-right-inverted-firmware note);
  * the VLM goal is clamped/normalized and can only bias, never command wheels;
  * forward_intent hold/retreat and caution behave as documented.

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

import reflex_drive as rd  # noqa: E402


def make_args(**overrides):
    args = rd.build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def per(left, center, right, source="depth"):
    return rd.Perception(t=0.0, seq=0, source=source, left=left, center=center,
                         right=right, latency=0.0)


class ClampGoalTests(unittest.TestCase):
    def test_empty_falls_back_to_safe_defaults(self):
        g = rd.clamp_goal({}, "gemini")
        self.assertEqual(g.mode, "explore")
        self.assertEqual(g.route_bias, "center")
        self.assertEqual(g.forward_intent, "advance")
        self.assertFalse(g.target_visible)
        self.assertEqual(g.target_bearing, "none")
        self.assertEqual(g.target_size, "none")
        self.assertFalse(g.arrived)
        self.assertEqual(g.confidence, 0.0)

    def test_bearing_overrides_route_bias(self):
        self.assertEqual(rd.clamp_goal({"target_bearing": "far_left", "route_bias": "right"}, "x").route_bias, "left")
        self.assertEqual(rd.clamp_goal({"target_bearing": "right", "route_bias": "left"}, "x").route_bias, "right")

    def test_confidence_clamped_and_reason_truncated(self):
        self.assertEqual(rd.clamp_goal({"confidence": 5}, "x").confidence, 1.0)
        self.assertEqual(rd.clamp_goal({"confidence": "nope"}, "x").confidence, 0.0)
        self.assertLessEqual(len(rd.clamp_goal({"reason": "z" * 500}, "x").reason), 140)

    def test_unknown_enums_default(self):
        g = rd.clamp_goal({"mode": "fly", "route_bias": "up", "forward_intent": "warp",
                           "target_bearing": "behind", "target_size": "huge"}, "x")
        self.assertEqual((g.mode, g.route_bias, g.forward_intent, g.target_bearing, g.target_size),
                         ("explore", "center", "advance", "none", "none"))


class BiasSignTests(unittest.TestCase):
    """The whole-stack convention: positive steer / +trim == toward the RIGHT."""

    def test_route_bias_value(self):
        self.assertLess(rd.route_bias_value("left"), 0)
        self.assertGreater(rd.route_bias_value("right"), 0)
        self.assertEqual(rd.route_bias_value("center"), 0)

    def test_visible_target_bearing_sign(self):
        mk = lambda b: rd.Goal(target_visible=True, target_bearing=b)
        self.assertEqual(rd.goal_bias_value(mk("far_left")), -1.0)
        self.assertLess(rd.goal_bias_value(mk("left")), 0)
        self.assertGreater(rd.goal_bias_value(mk("right")), 0)
        self.assertEqual(rd.goal_bias_value(mk("far_right")), 1.0)
        self.assertEqual(rd.goal_bias_value(mk("center")), 0.0)


class CommandFromStateTests(unittest.TestCase):
    def test_open_corridor_drives_straight(self):
        args = make_args()
        kind, speed, trim = rd.command_from_state(per(0.6, 0.7, 0.6), rd.Goal(), rd.MapState(), args)
        self.assertEqual(kind, "arc")
        self.assertGreaterEqual(speed, args.min_drive_speed)
        self.assertEqual(trim, 0)  # symmetric openness -> no steer

    def test_steers_toward_the_open_side(self):
        args = make_args()
        # right much more open than left -> steer right (+trim)
        _, _, trim_r = rd.command_from_state(per(0.2, 0.6, 0.8), rd.Goal(), rd.MapState(), args)
        self.assertGreater(trim_r, 0)
        # mirror image -> steer left (-trim)
        _, _, trim_l = rd.command_from_state(per(0.8, 0.6, 0.2), rd.Goal(), rd.MapState(), args)
        self.assertLess(trim_l, 0)
        self.assertEqual(trim_r, -trim_l)

    def test_tight_center_reduces_speed(self):
        args = make_args(speed=3, min_drive_speed=1, approach_speed=1)
        _, fast, _ = rd.command_from_state(per(0.6, 0.7, 0.6), rd.Goal(), rd.MapState(), args)
        _, slow, _ = rd.command_from_state(per(0.6, 0.40, 0.6), rd.Goal(), rd.MapState(), args)
        self.assertEqual(fast, 3)
        self.assertLess(slow, fast)

    def test_hold_intent_stops_without_retreat(self):
        args = make_args()
        kind, speed, _ = rd.command_from_state(per(0.6, 0.7, 0.6), rd.Goal(forward_intent="hold"), rd.MapState(), args)
        self.assertEqual(kind, "arc")
        self.assertEqual(speed, 0)

    def test_retreat_intent(self):
        args = make_args()
        kind, speed, trim = rd.command_from_state(per(0.6, 0.7, 0.6), rd.Goal(forward_intent="retreat"), rd.MapState(), args)
        self.assertEqual((kind, speed, trim), ("retreat", 0, 0))

    def test_caution_caps_speed(self):
        args = make_args(speed=3)
        _, speed, _ = rd.command_from_state(per(0.6, 0.7, 0.6), rd.Goal(caution=True), rd.MapState(), args)
        self.assertLessEqual(speed, args.caution_speed)

    def test_centered_target_steering_is_tightly_clamped(self):
        args = make_args()
        g = rd.Goal(target_visible=True, target_bearing="center", target_size="small")
        _, _, trim = rd.command_from_state(per(0.1, 0.6, 0.9), g, rd.MapState(), args)
        self.assertLessEqual(abs(trim), args.target_center_max_trim)

    def test_far_target_steers_toward_it_within_limit(self):
        args = make_args()
        g = rd.Goal(target_visible=True, target_bearing="far_left", target_size="small")
        _, _, trim = rd.command_from_state(per(0.5, 0.6, 0.5), g, rd.MapState(), args)
        self.assertLess(trim, 0)                       # toward the target on the left
        self.assertGreaterEqual(trim, -args.target_far_max_trim)


class GoalParsingTests(unittest.TestCase):
    def test_parse_json_object_from_noise(self):
        self.assertEqual(rd.parse_json_object('blah {"a": 1, "b": 2} trailing'), {"a": 1, "b": 2})

    def test_parse_json_object_single_quotes(self):
        self.assertEqual(rd.parse_json_object("{'mode': 'explore'}"), {"mode": "explore"})

    def test_parse_json_object_none_on_garbage(self):
        self.assertIsNone(rd.parse_json_object("no json at all"))


class EffectiveGoalTests(unittest.TestCase):
    def test_local_only_passthrough(self):
        g = rd.Goal(t=0.0, source="local-only", reason="keep me")
        self.assertIs(rd.effective_goal(g, stale_sec=12.0), g)

    def test_stale_goal_falls_back(self):
        g = rd.Goal(t=0.0, source="gemini")  # t<=0 -> stale
        out = rd.effective_goal(g, stale_sec=12.0)
        self.assertIn("stale", out.reason.lower())

    def test_fresh_goal_kept(self):
        import time
        g = rd.Goal(t=time.time(), source="gemini", reason="fresh")
        self.assertIs(rd.effective_goal(g, stale_sec=12.0), g)


class ExplorerMemoryTests(unittest.TestCase):
    def test_bias_prefers_more_open_frontier(self):
        args = make_args()
        mem = rd.ExplorerMemory(args.map_cell_size)
        # right clearly most open on a fresh map -> committed bias should be right
        mem.update("arc", args.speed, 0, per(0.2, 0.5, 0.9), rd.Goal(), args)
        self.assertEqual(mem.state().route_bias, "right")


if __name__ == "__main__":
    unittest.main(verbosity=2)
