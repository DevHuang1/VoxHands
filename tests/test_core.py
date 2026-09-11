import time
import unittest

from voxhands.planner import build_plan
from voxhands.safety import validate_plan
from voxhands.simulation import TableSettingSimulation, minimum_jerk


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

    def test_unknown_object_command_does_not_fall_back_to_table_plan(self) -> None:
        plan = build_plan("Pick up the red one and place it on the left.")
        self.assertEqual(plan.actions, [])
        issues = validate_plan(plan)
        self.assertIn("no-go safety barrier", " ".join(issues).lower())


class SimulationTests(unittest.TestCase):
    def test_minimum_jerk_has_bounded_monotonic_progress(self) -> None:
        samples = [minimum_jerk(index / 20) for index in range(21)]
        self.assertEqual(samples[0], 0)
        self.assertEqual(samples[-1], 1)
        self.assertEqual(samples, sorted(samples))

    def test_snapshot_exposes_3d_and_physics_contract(self) -> None:
        simulation = TableSettingSimulation()
        state = simulation.snapshot()
        self.assertEqual(state["physics"]["engine"], "rapier3d-browser")
        self.assertEqual(state["physics"]["timestep_hz"], 60)
        self.assertIsNone(state["physics"]["grasp_mode"])
        self.assertFalse(state["physics"]["left_pad_contact"])
        self.assertIn("z", state["objects"]["cup"])
        self.assertIn("pose", state["arms"]["left"])
        self.assertEqual(state["arms"]["left"]["gripper"], "open")

    def test_physics_telemetry_cannot_change_server_status(self) -> None:
        simulation = TableSettingSimulation()
        simulation.record_physics_event({"kind": "collision", "pair": "test barrier"})
        simulation.record_physics_event({"kind": "telemetry", "grasp_lock": True, "drop_lock": True, "settle_steps": 12, "settle_velocity": 0.0, "grasp_mode": "contact", "left_pad_contact": True, "right_pad_contact": True, "grasp_distance_mm": 4.2, "grasp_attempts": 4, "smoothness_max_step_mm": 18.5})
        state = simulation.snapshot()
        self.assertEqual(state["status"], "idle")
        self.assertEqual(state["physics"]["collision_count"], 1)
        self.assertTrue(state["physics"]["grasp_lock"])
        self.assertTrue(state["physics"]["drop_lock"])
        self.assertEqual(state["physics"]["settle_steps"], 12)
        self.assertEqual(state["physics"]["settle_velocity"], 0.0)
        self.assertEqual(state["physics"]["grasp_mode"], "contact")
        self.assertTrue(state["physics"]["left_pad_contact"])
        self.assertTrue(state["physics"]["right_pad_contact"])
        self.assertEqual(state["physics"]["grasp_distance_mm"], 4.2)
        self.assertEqual(state["physics"]["grasp_attempts"], 4)
        self.assertEqual(state["physics"]["smoothness_max_step_mm"], 18.5)

    def test_pause_resume_and_stop_controls_are_truthful(self) -> None:
        simulation = TableSettingSimulation()
        simulation.submit_command("Place the blue plate on the left.")
        time.sleep(0.12)
        paused = simulation.pause()
        paused_revision = paused["motion"]["revision"]
        self.assertEqual(paused["status"], "paused")
        time.sleep(0.08)
        self.assertEqual(simulation.snapshot()["motion"]["revision"], paused_revision)
        resumed = simulation.resume()
        self.assertEqual(resumed["status"], "running")
        stopped = simulation.stop()
        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(stopped["plan"]["status"], "stopped")
        self.assertGreaterEqual(stopped["metrics"]["recovery_count"], 1)

    def test_command_reaches_complete_state(self) -> None:
        simulation = TableSettingSimulation()
        simulation.submit_command("Place the blue plate on the left and the cup on the right.")
        deadline = time.time() + 6
        revisions = []
        blue_x_values = []
        while simulation.snapshot()["status"] == "running" and time.time() < deadline:
            snapshot = simulation.snapshot()
            revisions.append(snapshot["motion"]["revision"])
            blue_x_values.append(snapshot["objects"]["blue_plate"]["x"])
            time.sleep(0.05)
        state = simulation.snapshot()
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["metrics"]["successful_runs"], 1)
        self.assertEqual(state["objects"]["blue_plate"]["x"], state["targets"]["left_place"]["x"])
        self.assertEqual(state["objects"]["cup"]["x"], state["targets"]["right_place"]["x"])
        self.assertEqual(state["objects"]["blue_plate"]["pose"]["x"], state["targets"]["left_place"]["x"])
        self.assertTrue(state["objects"]["blue_plate"]["settled"])
        self.assertEqual(state["motion"]["elapsed_ms"], state["motion"]["duration_ms"])
        self.assertEqual(state["motion"]["sample_hz"], 60)
        self.assertGreater(len(revisions), 2)
        self.assertEqual(revisions, sorted(revisions))
        self.assertTrue(all(0.22 <= value <= 0.35 for value in blue_x_values))

    def test_release_settle_and_return_are_distinct_phases(self) -> None:
        simulation = TableSettingSimulation()
        simulation.submit_command("Place the blue plate on the left and the cup on the right.")
        seen: list[str] = []
        release_state = None
        deadline = time.time() + 7
        while time.time() < deadline:
            state = simulation.snapshot()
            phase = state["arms"]["left"]["phase"]
            if phase not in seen:
                seen.append(phase)
            if phase == "releasing" and release_state is None:
                release_state = state
            if state["status"] == "complete":
                break
            time.sleep(0.035)
        state = simulation.snapshot()
        self.assertEqual(state["status"], "complete")
        self.assertTrue(all(phase in seen for phase in ("placing", "releasing", "settling", "returning")))
        self.assertIsNotNone(release_state)
        self.assertEqual(release_state["arms"]["left"]["phase"], "releasing")
        self.assertEqual(release_state["objects"]["blue_plate"]["holder"], "left")
        self.assertEqual(state["arms"]["left"]["phase"], "parked")
        self.assertEqual(state["arms"]["left"]["pose"]["x"], 0.24)
        self.assertTrue(state["objects"]["blue_plate"]["settled"])
        self.assertIsNotNone(state["motion"]["release_revision"])

    def test_grasp_confirmation_precedes_carry_and_release_is_precise(self) -> None:
        simulation = TableSettingSimulation()
        simulation.submit_command("Place the blue plate on the left and the cup on the right.")
        grasp_state = None
        carry_state = None
        release_state = None
        deadline = time.time() + 7
        while time.time() < deadline:
            state = simulation.snapshot()
            arm = state["arms"]["left"]
            if arm["phase"] == "gripping" and arm["grasp_confirmed"] and grasp_state is None:
                grasp_state = state
            if arm["phase"] == "carrying" and carry_state is None:
                carry_state = state
            if arm["phase"] == "releasing" and release_state is None:
                release_state = state
            if state["status"] == "complete":
                break
            time.sleep(0.025)
        state = simulation.snapshot()
        self.assertIsNotNone(grasp_state)
        self.assertIsNotNone(carry_state)
        self.assertIsNotNone(release_state)
        self.assertEqual(grasp_state["objects"]["blue_plate"]["holder"], "left")
        self.assertTrue(grasp_state["arms"]["left"]["grasp_confirmed"])
        self.assertAlmostEqual(grasp_state["arms"]["left"]["grip_aperture"], 0.6305, places=4)
        self.assertEqual(carry_state["arms"]["left"]["holding"], "blue_plate")
        self.assertEqual(release_state["objects"]["blue_plate"]["rotation"], 0)
        self.assertLessEqual(release_state["objects"]["blue_plate"]["target_error_cm"], 1.0)
        self.assertTrue(state["objects"]["blue_plate"]["target_locked"])
        self.assertEqual(state["objects"]["blue_plate"]["target_error_cm"], 0.0)

    def test_unknown_command_does_not_start_worker(self) -> None:
        simulation = TableSettingSimulation()
        state = simulation.submit_command("Please inspect the weather")
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["plan"]["status"], "blocked")
        self.assertTrue(state["plan"]["safety_issues"])


if __name__ == "__main__":
    unittest.main()
