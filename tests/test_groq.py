import time
import unittest
from unittest.mock import patch

from voxhands.groq import GroqClient, ai_build_plan, plan_from_ai_payload
from voxhands.models import Action, Plan
from voxhands.safety import validate_plan
from voxhands.simulation import TableSettingSimulation
from voxhands.styles import normalize_gesture, normalize_style


class FakeGroq(GroqClient):
    def __init__(self, payload=None, exc=None):
        super().__init__(api_key="test-key")
        self._payload = payload or {}
        self._exc = exc

    def parse_command(self, text):
        if self._exc:
            raise self._exc
        return self._payload


class GroqPlannerTests(unittest.TestCase):
    def test_current_models_are_json_controller_defaults(self):
        client = GroqClient(api_key="test-key")
        self.assertEqual(client.model, "openai/gpt-oss-120b")
        self.assertEqual(client.fallback_model, "openai/gpt-oss-20b")

    def test_parse_command_tries_fallback_model(self):
        class FallbackClient(GroqClient):
            def __init__(self):
                super().__init__(api_key="test-key", model="primary", fallback_model="fallback")
                self.models = []

            def _complete(self, messages, model=None, **kwargs):
                self.models.append(model)
                if model == "primary":
                    raise RuntimeError("primary unavailable")
                return '{"intent":"conversation","actions":[],"reply":"ok"}'

        client = FallbackClient()
        self.assertEqual(client.parse_command("hello"), {"intent": "conversation", "actions": [], "reply": "ok"})
        self.assertEqual(client.models, ["primary", "fallback"])

    def test_ai_payload_builds_styled_plan(self):
        payload = {
            "intent": "object_placement",
            "actions": [{"object_id": "cup", "target_id": "right_place", "style": "rapid", "speed": 1.7, "gesture": "dance", "pause_ms": 200, "condition": "only if the red zone is clear"}],
            "constraints": ["avoid_red_zone", "no_arm_collision"],
            "recognized_objects": ["Cup"],
            "suggestions": [],
            "reply": "Fast cup placement, then a dance.",
        }
        plan, reply = ai_build_plan("put the cup right quickly and dance", FakeGroq(payload))
        self.assertEqual(plan.intent, "object_placement")
        self.assertEqual(plan.mode, "command")
        self.assertEqual(plan.llm_provider, "groq")
        self.assertEqual(validate_plan(plan), [])
        action = plan.actions[0]
        self.assertEqual(action.object_id, "cup")
        self.assertEqual(action.target_id, "right_place")
        self.assertEqual(action.style, "rapid")
        self.assertEqual(action.speed, 1.7)
        self.assertEqual(action.gesture, "dance")
        self.assertEqual(action.pause_ms, 200)
        self.assertEqual(action.condition, "only if the red zone is clear")
        self.assertEqual(reply, "Fast cup placement, then a dance.")
        self.assertEqual(plan.to_dict()["actions"][0]["style"], "rapid")

    def test_conversation_intent_returns_reply_only(self):
        payload = {"intent": "conversation", "actions": [], "recognized_objects": [], "suggestions": [], "reply": "I can move the plate, cup, fork, or spoon."}
        plan, reply = ai_build_plan("hello there", FakeGroq(payload))
        self.assertEqual(plan.mode, "conversation")
        self.assertEqual(plan.actions, [])
        self.assertIn("plate", reply)
        sim = TableSettingSimulation()
        state = sim.reply("hello there", reply, plan.llm_provider, plan.suggestions)
        self.assertEqual(state["status"], "idle")
        self.assertEqual(state["transcript"]["reply"], reply)

    def test_exception_falls_back_to_keyword_planner(self):
        plan, reply = ai_build_plan("Place the cup on the right.", FakeGroq(exc=RuntimeError("boom")))
        self.assertEqual(plan.llm_provider, "keyword")
        self.assertEqual(validate_plan(plan), [])
        self.assertEqual(reply, "")

    def test_no_key_falls_back_to_keyword_planner(self):
        with patch.dict("voxhands.groq.os.environ", {"GROQ_API_KEY": ""}):
            plan, _ = ai_build_plan("Place the cup on the right.", GroqClient(api_key=""))
        self.assertEqual(plan.llm_provider, "keyword")
        self.assertEqual(plan.actions[0].target_id, "right_place")

    def test_unknown_reference_creates_safety_issue(self):
        payload = {"intent": "object_placement", "actions": [{"object_id": "nope", "target_id": "right_place"}], "reply": "x"}
        plan, _ = ai_build_plan("move the nope", FakeGroq(payload))
        self.assertTrue(validate_plan(plan))
        self.assertEqual(plan.actions, [])

    def test_extreme_ai_values_are_sanitized(self):
        payload = {
            "intent": "object_placement",
            "actions": [{"object_id": "cup", "target_id": "right_place", "speed": 99, "pause_ms": -100, "gesture": "jump", "style": "nonsense"}],
            "reply": "x",
        }
        plan, _ = ai_build_plan("place the cup right", FakeGroq(payload))
        self.assertEqual(validate_plan(plan), [])
        action = plan.actions[0]
        self.assertEqual(action.speed, 3.0)
        self.assertEqual(action.pause_ms, 0)
        self.assertEqual(action.gesture, "none")
        self.assertEqual(action.style, "standard")

    def test_regex_planner_accepts_style_override(self):
        with patch.dict("voxhands.groq.os.environ", {"GROQ_API_KEY": ""}):
            plan, _ = ai_build_plan("Move the cup to the right.", GroqClient(api_key=""), style="gentle", gesture="wave")
        self.assertEqual(plan.llm_provider, "keyword")
        self.assertEqual(plan.actions[0].style, "gentle")
        self.assertEqual(plan.actions[0].gesture, "wave")

    def test_style_aliases_normalize(self):
        self.assertEqual(normalize_style("gently"), "gentle")
        self.assertEqual(normalize_style("ZIGZAG"), "wavy")
        self.assertEqual(normalize_gesture("hi"), "wave")
        self.assertEqual(normalize_gesture("none"), "none")


class StyledSimulationTests(unittest.TestCase):
    def test_speed_scales_duration(self):
        sim = TableSettingSimulation()
        slow_acts = [Action("a1", "left", "blue_plate", "Blue plate", "left_place", "Left place", duration_ms=5000, style="gentle", speed=0.55)]
        slow_plan = Plan("p1", "gently place the plate", "object_placement", slow_acts, ["avoid_red_zone", "no_arm_collision"])
        slow = sim.submit_command(slow_plan.raw_text, plan=slow_plan)
        sim.stop()
        sim.reset()
        fast_acts = [Action("a2", "left", "blue_plate", "Blue plate", "left_place", "Left place", duration_ms=5000, style="rapid", speed=1.7)]
        fast_plan = Plan("p2", "quickly place the plate", "object_placement", fast_acts, ["avoid_red_zone", "no_arm_collision"])
        fast = sim.submit_command(fast_plan.raw_text, plan=fast_plan)
        sim.stop()
        self.assertGreater(slow["plan"]["actions"][0]["duration_ms"], fast["plan"]["actions"][0]["duration_ms"])
        self.assertGreater(slow["motion"]["duration_ms"], 5000)
        self.assertLess(fast["motion"]["duration_ms"], 5000)

    def test_gesture_and_hold_extend_timeline_and_execute(self):
        with patch("voxhands.simulation.RUN_DURATION_MS", 300):
            acts = [Action("a1", "left", "blue_plate", "Blue plate", "left_place", "Left place", duration_ms=300, style="playful", gesture="wave", pause_ms=150)]
            plan = Plan("p1", "place the plate and wave", "object_placement", acts, ["avoid_red_zone", "no_arm_collision"])
            sim = TableSettingSimulation()
            state = sim.submit_command(plan.raw_text, plan=plan, reply="Done — wave!")
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["motion"]["gesture"], "wave")
            self.assertGreaterEqual(state["motion"]["duration_ms"], 300 + 150 + 2600)
            deadline = time.monotonic() + 6
            saw_wave = False
            while sim.snapshot()["status"] == "running" and time.monotonic() < deadline:
                if sim.snapshot()["arms"]["left"]["phase"] == "wave":
                    saw_wave = True
                time.sleep(.01)
            final = sim.snapshot()
            self.assertEqual(final["status"], "complete")
            self.assertTrue(saw_wave)
            self.assertEqual(final["objects"]["blue_plate"]["x"], 0.35)

    def test_hold_phase_shows_in_events(self):
        with patch("voxhands.simulation.RUN_DURATION_MS", 200):
            acts = [Action("a1", "left", "blue_plate", "Blue plate", "left_place", "Left place", duration_ms=200, pause_ms=300)]
            plan = Plan("p1", "place and hold", "object_placement", acts, ["avoid_red_zone", "no_arm_collision"])
            sim = TableSettingSimulation()
            state = sim.submit_command(plan.raw_text, plan=plan)
            self.assertEqual(state["motion"]["hold_ms"], 300)
            self.assertGreaterEqual(state["motion"]["duration_ms"], 500)
            self.assertIn("Holding", " ".join(event["message"] for event in state["events"]))
            sim.stop()


if __name__ == "__main__":
    unittest.main()
