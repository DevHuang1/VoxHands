import json
import threading
import time
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error
from voxhands.planner import build_plan
from voxhands.safety import validate_plan
from voxhands.simulation import TableSettingSimulation
from voxhands.server import VoxHandsServer

class Improvements(unittest.TestCase):
    def test_missing_destination_has_actionable_choices(self):
        plan=build_plan("move spoon")
        issues=validate_plan(plan)
        self.assertEqual(plan.recognized_objects,["Spoon"])
        self.assertEqual(len(issues),1)
        self.assertIn("Where should Spoon go",issues[0])
        self.assertEqual(len(plan.suggestions),3)
        for suggestion in plan.suggestions:
            self.assertEqual(validate_plan(build_plan(suggestion["text"])),[])

    def test_compound_clarification_preserves_known_destination(self):
        plan=build_plan("Move the plate left and move the spoon")
        self.assertEqual(len(plan.suggestions),2)
        for suggestion in plan.suggestions:
            candidate=build_plan(suggestion["text"])
            self.assertEqual(validate_plan(candidate),[])
            self.assertEqual(candidate.actions[0].target_id,"left_place")

    def test_safety_phrase_is_not_movement_negation(self):
        self.assertEqual(validate_plan(build_plan("Place the cup right. Do not collide.")),[])

    def test_spoon_into_cup_is_a_container_action(self):
        plan = build_plan("Put the spoon into the cup.")
        self.assertEqual(validate_plan(plan), [])
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].object_id, "spoon")
        self.assertEqual(plan.actions[0].target_id, "cup_interior")

    def test_spoon_into_cup_finishes_inside_live_container(self):
        with patch("voxhands.simulation.RUN_DURATION_MS", 400):
            sim = TableSettingSimulation()
            state = sim.submit_command("Put the spoon into the cup.")
            self.assertEqual(state["status"], "running")
            deadline = time.monotonic() + 3
            while sim.snapshot()["status"] == "running" and time.monotonic() < deadline:
                time.sleep(.01)
            state = sim.snapshot()
            spoon = state["objects"]["spoon"]
            cup = state["objects"]["cup"]
            self.assertEqual(state["status"], "complete")
            self.assertEqual(spoon["container"], "cup")
            self.assertAlmostEqual(spoon["x"], cup["x"], places=4)
            self.assertAlmostEqual(spoon["y"], cup["y"], places=4)
            self.assertGreater(spoon["z"], cup["z"])

    def test_occupied_destination_is_blocked(self):
        sim=TableSettingSimulation()
        sim.objects["blue_plate"].update(x=.35,y=.42)
        state=sim.submit_command("Move spoon left")
        self.assertEqual(state["status"],"blocked")
        self.assertIn("occupied", " ".join(state["plan"]["safety_issues"]))

    def test_clarification_does_not_change_execution_rate(self):
        sim=TableSettingSimulation()
        sim.metrics.update(successful_runs=1,started_runs=1,success_rate=100)
        state=sim.submit_command("move spoon")
        self.assertEqual(state["metrics"]["success_rate"],100)
        self.assertEqual(state["metrics"]["started_runs"],1)

    def test_destinations(self):
        for text, expected in [("Move the blue plate to the right.", "right_place"), ("Place the cup in the center.", "center_place")]:
            plan = build_plan(text)
            self.assertEqual(validate_plan(plan), [])
            self.assertEqual(plan.actions[0].target_id, expected)

    def test_reversed_pair_and_order(self):
        plan = build_plan("Put the cup on the left and the plate on the right.")
        self.assertEqual(validate_plan(plan), [])
        self.assertEqual([(a.object_id,a.target_id) for a in plan.actions], [("cup","left_place"),("blue_plate","right_place")])

    def test_ambiguous_or_partial_commands_block(self):
        for text in ["Move the cup.", "Do not move the cup to the right.", "Place the cup on the left or right.", "Put the cup on the right and the bowl on the left.", "Move the cupboard to the right."]:
            self.assertTrue(validate_plan(build_plan(text)), text)

    def test_home_calibration(self):
        sim = TableSettingSimulation()
        self.assertEqual(sim.submit_command("Home both arms.")["status"], "idle")
        state = sim.submit_command("Show calibration status.")
        self.assertIn("no camera", state["events"][-1]["message"])

    def test_busy_and_stopped_guard(self):
        sim = TableSettingSimulation()
        first = sim.submit_command("Place the cup on the right.")
        self.assertIn("error", sim.submit_command("Place the plate on the left."))
        self.assertEqual(sim.snapshot()["plan"]["id"],first["plan"]["id"])
        sim.stop()
        self.assertIn("error",sim.submit_command("Place the cup on the right."))
        sim.reset()
        self.assertEqual(sim.snapshot()["status"], "idle")

    def test_crossed_start_positions_do_not_run_parallel(self):
        sim=TableSettingSimulation()
        sim.objects["blue_plate"].update(x=.70,y=.69)
        sim.objects["cup"].update(x=.30,y=.69)
        state=sim.submit_command("Set the table for two.")
        self.assertEqual(state["status"],"running")
        offsets=list(sim._offsets.values())
        self.assertGreater(offsets[1],offsets[0])
        sim.stop()

    def test_collision_run_identity(self):
        sim = TableSettingSimulation()
        state = sim.submit_command("Place the cup on the right.")
        sim.record_physics_event({"kind":"collision","run_id":"old"})
        self.assertEqual(sim.snapshot()["status"],"running")
        sim.record_physics_event({"kind":"collision","run_id":state["plan"]["id"]})
        self.assertEqual(sim.snapshot()["status"],"stopped")

    def test_queued_moves_and_reset_cancellation(self):
        sim = TableSettingSimulation()
        state = sim.submit_command("Place the plate on the left and the fork in the center.")
        self.assertEqual(state["status"],"running")
        self.assertGreaterEqual(state["motion"]["duration_ms"],10000)
        time.sleep(.1)
        self.assertEqual(sim.snapshot()["plan"]["actions"][1]["status"],"queued")
        sim.reset()
        time.sleep(.05)
        self.assertEqual(sim.snapshot()["status"],"idle")
        self.assertEqual(sim.snapshot()["motion"]["revision"],0)

    def test_three_objects_complete_sequentially_and_repeat(self):
        with patch("voxhands.simulation.RUN_DURATION_MS", 400):
            sim=TableSettingSimulation()
            for attempt in range(2):
                state=sim.submit_command("Place the plate on the left, the cup on the right and the fork in the center.")
                self.assertEqual(state["status"],"running")
                deadline=time.monotonic()+4
                while sim.snapshot()["status"]=="running" and time.monotonic()<deadline:
                    state=sim.snapshot()
                    for side in ("left","right"):
                        active=[a for a in state["plan"]["actions"] if a["arm"]==side and a["status"] not in {"queued","placed"}]
                        self.assertLessEqual(len(active),1)
                    time.sleep(.01)
                self.assertEqual(sim.snapshot()["status"],"complete")
                self.assertTrue(all(a["status"]=="placed" for a in sim.snapshot()["plan"]["actions"]))

    def test_barrier_detour(self):
        sim=TableSettingSimulation()
        state=sim.submit_command("Place the plate on the right.")
        self.assertEqual(state["status"],"running")
        route=sim._routes[state["plan"]["actions"][0]["id"]]
        self.assertGreater(len(route),2)
        for step in range(501):
            x,y=sim._route_pose(route,step/500)
            self.assertFalse(.42<=x<=.58 and .51<=y<=.81)
        sim.stop()

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=VoxHandsServer(("127.0.0.1",0))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.base=f"http://127.0.0.1:{cls.server.server_port}"
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
    def test_assets(self):
        for path in ["/", "/api/health", "/vendor/three.module.js", "/vendor/rapier.es.js"]:
            with urllib.request.urlopen(self.base+path) as response:
                self.assertEqual(response.status,200)
    def test_invalid_json_and_types(self):
        for data in [b"{",b"[]",b'{"text":42}',b'{"text":""}']:
            request=urllib.request.Request(self.base+"/api/command",data=data,headers={"Content-Type":"application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code,400)
    def test_vendor_traversal(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.base+"/vendor/../server.py")
        self.assertEqual(caught.exception.code,404)

    def test_plan_preview_is_read_only(self):
        before = self.server.simulation.snapshot()
        request = urllib.request.Request(self.base+"/api/plan", data=b'{"text":"Place the cup on the right."}', headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read())
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["plan"]["actions"][0]["target_id"], "right_place")
        after = self.server.simulation.snapshot()
        self.assertEqual(after["status"], before["status"])
        self.assertEqual(after["metrics"], before["metrics"])

    def test_non_finite_telemetry_is_sanitized(self):
        sim = TableSettingSimulation()
        state = sim.record_physics_event({"kind":"telemetry", "settle_velocity":float("nan"), "grasp_distance_mm":float("inf"), "smoothness_max_step_mm":float("-inf")})
        self.assertEqual(state["physics"]["settle_velocity"], 0.0)
        self.assertIsNone(state["physics"]["grasp_distance_mm"])
        self.assertEqual(state["physics"]["smoothness_max_step_mm"], 0.0)
