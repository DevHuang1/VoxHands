import json
import time
import unittest
from unittest.mock import patch

from voxhands.agent import WorkcellToolDispatcher, run_agent, unsupported_target_issue
from voxhands.simulation import TableSettingSimulation


class FakeToolGroq:
    available = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def tool_complete(self, messages, tools):
        self.calls.append((messages, tools))
        return next(self.responses)


def tool_response(name, arguments):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
            }
        }]
    }


class AgentToolTests(unittest.TestCase):
    def test_ground_target_is_rejected_by_server_tool_boundary(self):
        simulation = TableSettingSimulation()
        issue = unsupported_target_issue("move the blue plate to the ground")
        self.assertIn("Unsupported destination", issue or "")
        dispatcher = WorkcellToolDispatcher(simulation, "move the blue plate to the ground", blocked_issue=issue)
        result = dispatcher.call("execute_placement", {"object_id": "blue_plate", "target_id": "left_place"})
        self.assertIn("error", result)

    def test_unsupported_ground_request_never_reinterprets_target(self):
        simulation = TableSettingSimulation()
        client = FakeToolGroq([
            tool_response("execute_placement", {"object_id": "blue_plate", "target_id": "left_place"}),
            {"choices": [{"message": {"role": "assistant", "content": "I cannot place objects on the ground.", "tool_calls": []}}]},
        ])
        state = run_agent("Move the blue plate to the ground.", simulation, client)
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["plan"]["actions"], [])
        self.assertIn("Unsupported destination", state["plan"]["safety_issues"][0])
        self.assertEqual(state["motion"]["phase"], "parked")

    def test_manual_arm_and_gripper_controls_are_bounded(self):
        simulation = TableSettingSimulation()
        self.assertIn("error", simulation.move_arm("left", 1.2, 0.3, 0.2))
        state = simulation.move_arm("left", 0.30, 0.30, 0.20, duration_ms=100)
        self.assertEqual(state["arms"]["left"]["status"], "manual")
        time.sleep(0.2)
        self.assertAlmostEqual(simulation.snapshot()["arms"]["left"]["x"], 0.30, places=2)
        closed = simulation.set_gripper("left", "close", aperture=0.10)
        self.assertEqual(closed["arms"]["left"]["gripper"], "closed")

    def test_tool_loop_executes_authoritative_placement(self):
        with patch("voxhands.simulation.RUN_DURATION_MS", 120):
            simulation = TableSettingSimulation()
            client = FakeToolGroq([
                tool_response("execute_placement", {"object_id": "blue_plate", "target_id": "left_place"}),
                {"choices": [{"message": {"role": "assistant", "content": "The blue plate is moving to the left place.", "tool_calls": []}}]},
            ])
            state = run_agent("Move the blue plate to the left.", simulation, client)
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["plan"]["llm_provider"], "groq")
            deadline = time.time() + 3
            while simulation.snapshot()["status"] == "running" and time.time() < deadline:
                time.sleep(0.02)
            final = simulation.snapshot()
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["objects"]["blue_plate"]["target_locked"], True)
            self.assertEqual(final["transcript"]["reply"], "The blue plate is moving to the left place.")
            self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
