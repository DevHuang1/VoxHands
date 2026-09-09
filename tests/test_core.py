import time
import unittest

from voxhands.planner import build_plan
from voxhands.safety import validate_plan
from voxhands.simulation import TableSettingSimulation


class PlannerTests(unittest.TestCase):
    def test_table_command_creates_parallel_pair(self) -> None:
        plan = build_plan(
            "Set the table for two. Put the blue plate on the left and the cup on the right. Avoid the red zone."
        )
        self.assertEqual(plan.intent, "table_setting")
        self.assertEqual([action.object_id for action in plan.actions], ["blue_plate", "cup"])
        self.assertEqual([action.arm for action in plan.actions], ["left", "right"])
        self.assertEqual(validate_plan(plan), [])

    def test_unknown_command_is_blocked(self) -> None:
        plan = build_plan("Please inspect the weather")
        issues = validate_plan(plan)
        self.assertTrue(issues)


class SimulationTests(unittest.TestCase):
    def test_command_reaches_complete_state(self) -> None:
        simulation = TableSettingSimulation()
        simulation.submit_command("Place the blue plate on the left and the cup on the right.")
        deadline = time.time() + 6
        while simulation.snapshot()["status"] == "running" and time.time() < deadline:
            time.sleep(0.05)
        state = simulation.snapshot()
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["metrics"]["successful_runs"], 1)
        self.assertEqual(state["objects"]["blue_plate"]["x"], state["targets"]["left_place"]["x"])
        self.assertEqual(state["objects"]["cup"]["x"], state["targets"]["right_place"]["x"])

    def test_unknown_command_does_not_start_worker(self) -> None:
        simulation = TableSettingSimulation()
        state = simulation.submit_command("Please inspect the weather")
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["plan"]["status"], "blocked")
        self.assertTrue(state["plan"]["safety_issues"])


if __name__ == "__main__":
    unittest.main()
