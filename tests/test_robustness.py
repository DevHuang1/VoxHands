import unittest
from voxhands.mujoco_workcell import DinnerTableWorkcell, PrimitiveResult
from voxhands.assets.scene_dinner import DinnerConfig
from voxhands.robustness import (
    robust_eval,
    _evaluate_seed,
    _silverware_targets,
    _goal_plan,
    _nearest,
    canonical_goals,
)


def _oracle_factory(seed: int) -> DinnerTableWorkcell:
    return DinnerTableWorkcell(DinnerConfig(seed=seed), oracle=True)


class RobustnessTests(unittest.TestCase):
    def test_oracle_two_seeds_all_pass(self) -> None:
        result = robust_eval(_oracle_factory, seeds=2, oracle=True)
        self.assertTrue(result["all_ok"])
        self.assertEqual(result["seeds_ok"], 2)
        for run in result["runs"]:
            self.assertEqual(len(run["steps"]), 16)
            self.assertTrue(run["ok"])
            names = [s["name"] for s in run["steps"]]
            self.assertIn("open_drawer", names)
            self.assertIn("pour", names)
            self.assertIn("handoff_spoon", names)
            self.assertIn("check_final_state", names)

    def test_vision_two_seeds_pass_or_honest_failures(self) -> None:
        def vision_factory(seed: int) -> DinnerTableWorkcell:
            return DinnerTableWorkcell(DinnerConfig(seed=seed), oracle=False)

        result = robust_eval(vision_factory, seeds=2, oracle=False)
        self.assertTrue(result["all_ok"])
        self.assertEqual(result["seeds_ok"], 2)
        for run in result["runs"]:
            cam_steps = [s for s in run["steps"] if s["name"] == "camera_loop"]
            self.assertEqual(len(cam_steps), 1)
            self.assertTrue(cam_steps[0]["ok"])

    def test_silverware_targets_inside_drawer(self) -> None:
        env = _oracle_factory(0)
        env.open_drawer("left")
        spoon, fork = _silverware_targets(env)
        self.assertNotEqual(spoon, fork)
        draw = env._drawer_bounds()
        for cx, cy in (spoon, fork):
            self.assertGreaterEqual(cx, draw["x0"])
            self.assertLessEqual(cx, draw["x1"])
            self.assertGreaterEqual(cy, draw["y0"])
            self.assertLessEqual(cy, draw["y1"])

    def test_runner_custom_driver(self) -> None:
        def driver(env: DinnerTableWorkcell) -> PrimitiveResult:
            return PrimitiveResult("custom", True, message="stub ok")

        result = robust_eval(_oracle_factory, seeds=1, oracle=True, runner=driver)
        self.assertTrue(result["all_ok"])
        self.assertEqual(result["runs"][0]["ok"], True)

    def test_canonical_goals_alias_exists(self) -> None:
        self.assertIs(canonical_goals, _goal_plan)

    def test_nearest_simple(self) -> None:
        pts = [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)]
        self.assertEqual(_nearest(pts, 0.0, 0.0), (0.0, 0.0))
        self.assertEqual(_nearest(pts, 0.6, 0.6), (0.5, 0.5))
        self.assertEqual(_nearest([], 0.0, 0.0), None)


if __name__ == "__main__":
    unittest.main()
