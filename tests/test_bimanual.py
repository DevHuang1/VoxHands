import unittest

from voxhands.assets.scene_dinner import ARM_NAMES
from voxhands.bimanual import BimanualController, PlanStep
from voxhands.mujoco_workcell import TARGET_POSITIONS

CANONICAL_PRIMITIVES = [
    "open_drawer",
    "grasp",
    "place",
    "grasp",
    "place",
    "grasp",
    "grasp",
    "pour",
    "place",
    "handoff",
    "grasp",
    "place",
    "grasp",
    "place",
    "check_final_state",
]


class BimanualPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = BimanualController(oracle=True)
        self.plan = self.controller.plan_sequence()

    def test_sequence_is_proper_ordered_plan(self) -> None:
        self.assertIsInstance(self.plan, list)
        self.assertTrue(self.plan)
        self.assertTrue(all(isinstance(step, PlanStep) for step in self.plan))
        self.assertEqual(
            [step.primitive for step in self.plan],
            CANONICAL_PRIMITIVES,
        )
        self.assertIn("open_drawer", [step.primitive for step in self.plan])
        self.assertIn("pour", [step.primitive for step in self.plan])
        self.assertIn("handoff", [step.primitive for step in self.plan])
        self.assertEqual(self.plan[-1].primitive, "check_final_state")

    def test_all_placement_targets_valid(self) -> None:
        for step in self.plan:
            if step.primitive != "place":
                continue
            target = step.args[1]
            if isinstance(target, str):
                self.assertIn(target, TARGET_POSITIONS)
            else:
                x, y = target
                crescent = {
                    "left": {"x": (0.22, 0.58), "y": (0.30, 1.12)},
                    "right": {"x": (0.42, 0.78), "y": (0.30, 1.12)},
                }[step.arm]
                self.assertGreaterEqual(x, crescent["x"][0])
                self.assertLessEqual(x, crescent["x"][1])
                self.assertGreaterEqual(y, crescent["y"][0])
                self.assertLessEqual(y, crescent["y"][1])

    def test_handoff_and_pour_args_correct(self) -> None:
        handoff = next(step for step in self.plan if step.primitive == "handoff")
        self.assertEqual(handoff.args, ("left", "right", "bottle"))
        self.assertIn(handoff.args[0], ARM_NAMES)
        self.assertIn(handoff.args[1], ARM_NAMES)
        pour = next(step for step in self.plan if step.primitive == "pour")
        self.assertEqual(pour.args, ("left", "right", "bottle", "cup"))
        self.assertIn(pour.args[0], ARM_NAMES)
        self.assertIn(pour.args[1], ARM_NAMES)
        self.assertEqual(pour.args[0], pour.arm)
        self.assertEqual(pour.args, ("left", "right", "bottle", "cup"))

    def _parallel_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        index = 1
        while index < len(self.plan):
            if self.plan[index - 1].parallel_ok and self.plan[index].parallel_ok:
                pairs.append((index - 1, index))
                index += 2
            else:
                index += 1
        return pairs

    def test_parallel_ok_flags_on_non_colliding_steps(self) -> None:
        flagged = {
            step.primitive for step in self.plan if step.parallel_ok
        }
        self.assertIn("grasp", flagged)
        pairs = self._parallel_pairs()
        self.assertTrue(pairs)
        for lhs, rhs in pairs:
            self.assertNotEqual(
                self._zone_type(self.plan[lhs]),
                self._zone_type(self.plan[rhs]),
            )
        blocked = {
            (step.primitive, step.arm)
            for step in self.plan
            if not step.parallel_ok
        }
        self.assertIn(("open_drawer", "left"), blocked)
        self.assertIn(("pour", "left"), blocked)
        self.assertIn(("handoff", "left"), blocked)
        self.assertIn(("check_final_state", "left"), blocked)

    def test_parallel_zone_types_collision(self) -> None:
        left_zone_steps = [
            step
            for step in self.plan
            if self._zone_type(step) == "left"
        ]
        right_zone_steps = [
            step
            for step in self.plan
            if self._zone_type(step) == "right"
        ]
        self.assertTrue(left_zone_steps)
        self.assertTrue(right_zone_steps)

    @staticmethod
    def _zone_type(step: PlanStep) -> str | None:
        from voxhands.bimanual import BimanualController as C

        return C._zone_type(step)

    def test_refresh_goals_assigns_reachable_arms(self) -> None:
        refreshed = self.controller.refresh_goals()
        self.assertEqual(set(refreshed), set(TARGET_POSITIONS))
        self.assertEqual(refreshed["left_place"]["arm"], "left")
        self.assertEqual(refreshed["right_place"]["arm"], "right")
        self.assertIn(refreshed["center_place"]["arm"], ARM_NAMES)
        self.assertFalse(refreshed["left_place"]["snapped_away"])
        self.assertFalse(refreshed["right_place"]["snapped_away"])

    def test_focused_sub_plans_have_correct_args(self) -> None:
        pour_plan = self.controller.pour_plan()
        self.assertEqual(
            [step.primitive for step in pour_plan],
            ["grasp", "grasp", "pour"],
        )
        self.assertEqual(pour_plan[-1].args, ("left", "right", "bottle", "cup"))
        handoff_plan = self.controller.handoff_plan()
        self.assertEqual(
            [step.primitive for step in handoff_plan],
            ["grasp", "handoff"],
        )
        self.assertEqual(handoff_plan[-1].args, ("right", "left", "bottle"))
        drawer_plan = self.controller.drawer_plan()
        self.assertEqual(
            [step.primitive for step in drawer_plan],
            ["open_drawer", "close_drawer"],
        )

    def test_oracle_full_sequence_succeeds(self) -> None:
        results = self.controller.run(self.plan)
        self.assertEqual(len(results), len(self.plan))
        self.assertTrue(all(result.success for result in results))
        self.assertTrue(results[-1].success)
        self.assertTrue(self.controller.workcell.counters["pour"] > 0)
        self.assertTrue(self.controller.workcell.counters["handoff"] > 0)


if __name__ == "__main__":
    unittest.main()