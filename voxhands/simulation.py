from __future__ import annotations

import copy
import threading
import time
from typing import Any

from .integrations import runtime_status
from .models import Plan, utc_now
from .planner import TARGETS, build_plan
from .safety import validate_plan


OBJECT_HOME = {
    "blue_plate": {"label": "Blue plate", "color": "#62a6ff", "x": 0.22, "y": 0.69},
    "cup": {"label": "Cup", "color": "#f5c86b", "x": 0.72, "y": 0.69},
    "fork": {"label": "Fork", "color": "#dce7f5", "x": 0.40, "y": 0.77},
    "spoon": {"label": "Spoon", "color": "#cad7e8", "x": 0.58, "y": 0.77},
}

TARGET_POSITIONS = {
    "left_place": {"label": "Left place", "x": 0.35, "y": 0.42},
    "right_place": {"label": "Right place", "x": 0.65, "y": 0.42},
    "center_place": {"label": "Center place", "x": 0.50, "y": 0.42},
}

ARM_HOME = {
    "left": {"x": 0.24, "y": 0.16, "angle": -25},
    "right": {"x": 0.76, "y": 0.16, "angle": 25},
}


class TableSettingSimulation:
    """Deterministic MVP backend that mirrors the eventual MuJoCo control contract."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._run_token = 0
        self.runtime = runtime_status()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._run_token += 1
            self._worker = None
            self.status = "idle"
            self.plan: Plan | None = None
            self.transcript = {"text": "", "status": "waiting", "source": "demo transcript"}
            self.objects = copy.deepcopy(OBJECT_HOME)
            self.events: list[dict[str, Any]] = []
            self.metrics = {
                "commands": 0,
                "successful_runs": 0,
                "recovery_count": 0,
                "plan_latency_ms": 42,
                "last_run_ms": None,
                "sim_fps": 60,
                "success_rate": 0,
            }
            self._append_event("system", "VoxHands simulator ready. Both hands are parked.")

    def _append_event(self, kind: str, message: str) -> None:
        self.events.append({"time": utc_now(), "kind": kind, "message": message})
        self.events = self.events[-40:]

    def submit_command(self, raw_text: str) -> dict[str, Any]:
        plan = build_plan(raw_text)
        issues = validate_plan(plan)
        with self._lock:
            self._run_token += 1
            token = self._run_token
            self.metrics["commands"] += 1
            self.transcript = {"text": plan.raw_text, "status": "final", "source": "demo transcript"}
            self.plan = plan
            plan.safety_issues = issues
            if issues:
                self.status = "blocked"
                plan.status = "blocked"
                self._append_event("safety", "Command blocked: " + " ".join(issues))
                return self.snapshot()

            self.status = "running"
            plan.status = "executing"
            self._append_event("voice", f'Heard: “{plan.raw_text}”')
            self._append_event("planner", f"Plan {plan.id} validated with {len(plan.actions)} action(s).")
            self._append_event("safety", "Safety gate passed: red zone and arm collision constraints enabled.")
            self._worker = threading.Thread(target=self._execute, args=(token, plan), daemon=True)
            self._worker.start()
            return self.snapshot()

    def _execute(self, token: int, plan: Plan) -> None:
        started = time.perf_counter()
        with self._lock:
            starts = {
                action.id: {
                    "x": self.objects[action.object_id]["x"],
                    "y": self.objects[action.object_id]["y"],
                }
                for action in plan.actions
            }
            for action in plan.actions:
                action.status = "reaching"

        with self._lock:
            self._append_event("vision", "Camera observation locked: tabletop objects identified.")
            self._append_event("control", "Left and right arm trajectories scheduled in parallel where safe.")

        # Small, frequent updates make the arm and object motion visible in the browser
        # while preserving a short hackathon demo loop.
        steps = 60
        for step in range(1, steps + 1):
            time.sleep(0.065)
            with self._lock:
                if token != self._run_token:
                    return
                progress = step / steps * 100
                for action in plan.actions:
                    action.progress = progress
                    if step == steps:
                        action.status = "placed"
                    elif progress < 25:
                        action.status = "reaching"
                    elif progress < 42:
                        action.status = "gripping"
                    elif progress < 84:
                        action.status = "carrying"
                    else:
                        action.status = "placing"

                    target = TARGET_POSITIONS[action.target_id]
                    start = starts[action.id]
                    if progress < 42:
                        object_x, object_y = start["x"], start["y"]
                    else:
                        carry_ratio = min((progress - 42) / 58, 1)
                        object_x = start["x"] + (target["x"] - start["x"]) * carry_ratio
                        object_y = start["y"] + (target["y"] - start["y"]) * carry_ratio
                    self.objects[action.object_id]["x"] = object_x
                    self.objects[action.object_id]["y"] = object_y
                    self.objects[action.object_id]["motion"] = action.status

                if step in {15, 25, 38, 50}:
                    phase = plan.actions[0].status if plan.actions else "idle"
                    self._append_event("control", f"{phase.title()} phase · trajectories {round(progress)}% complete.")

        with self._lock:
            if token != self._run_token:
                return
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            plan.status = "complete"
            self.status = "complete"
            self.metrics["successful_runs"] += 1
            self.metrics["last_run_ms"] = elapsed_ms
            self.metrics["success_rate"] = round(self.metrics["successful_runs"] / self.metrics["commands"] * 100)
            self._append_event("success", f"Task complete in {elapsed_ms} ms. Both hands returned to safe idle.")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "transcript": copy.deepcopy(self.transcript),
                "plan": self.plan.to_dict() if self.plan else None,
                "objects": copy.deepcopy(self.objects),
                "targets": copy.deepcopy(TARGET_POSITIONS),
                "red_zone": {"x": 0.44, "y": 0.53, "width": 0.12, "height": 0.26},
                "events": copy.deepcopy(self.events),
                "metrics": copy.deepcopy(self.metrics),
                "runtime": copy.deepcopy(self.runtime),
                "arms": {
                    "left": self._arm_view("left"),
                    "right": self._arm_view("right"),
                },
            }

    def _arm_status(self, arm: str) -> str:
        if not self.plan or self.status == "idle":
            return "parked"
        statuses = [a.status for a in self.plan.actions if a.arm == arm]
        if self.status == "complete":
            return "parked"
        if "moving" in statuses:
            return "moving"
        if statuses:
            return statuses[-1]
        return "standby"

    def _arm_view(self, arm: str) -> dict[str, Any]:
        home = ARM_HOME[arm]
        parked = {
            "status": self._arm_status(arm),
            "phase": "parked",
            "x": home["x"],
            "y": home["y"],
            "angle": home["angle"],
            "progress": 0,
        }
        if not self.plan or self.status != "running":
            return parked

        action = next(
            (candidate for candidate in self.plan.actions if candidate.arm == arm and candidate.status != "placed"),
            None,
        )
        if action is None:
            return parked

        object_position = self.objects[action.object_id]
        target = TARGET_POSITIONS[action.target_id]
        progress = action.progress
        if progress < 25:
            ratio = progress / 25
            hand_x = home["x"] + (object_position["x"] - home["x"]) * ratio
            hand_y = 0.24 + (object_position["y"] - 0.24) * ratio
        elif progress < 84:
            hand_x = object_position["x"]
            hand_y = max(0.25, object_position["y"] - 0.12)
        else:
            ratio = min((progress - 84) / 16, 1)
            hand_x = object_position["x"] + (target["x"] - object_position["x"]) * ratio
            hand_y = max(0.25, target["y"] - 0.12)

        direction = -1 if arm == "left" else 1
        angle = direction * (18 + abs(hand_x - home["x"]) * 42)
        return {
            "status": self._arm_status(arm),
            "phase": action.status,
            "x": round(hand_x, 4),
            "y": round(hand_y, 4),
            "angle": round(angle, 2),
            "progress": round(progress, 1),
            "object_id": action.object_id,
        }
