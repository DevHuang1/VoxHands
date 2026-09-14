from __future__ import annotations

import copy
import math
import threading
import time
from typing import Any

from .integrations import runtime_status
from .models import Plan, utc_now
from .planner import TARGETS, build_plan
from .safety import validate_plan


SIM_HZ = 60
SIM_DT = 1 / SIM_HZ
RUN_DURATION_MS = 5000
PLACEMENT_TOLERANCE_M = 0.01
GRIP_CONFIRM_PROGRESS = 34
PHASE_THRESHOLDS = (
    (25, "reaching"),
    (42, "gripping"),
    (82, "carrying"),
    (90, "placing"),
    (92, "releasing"),
    (96, "settling"),
    (100, "returning"),
)

OBJECT_HOME = {
    "blue_plate": {"label": "Blue plate", "color": "#62a6ff", "x": 0.22, "y": 0.69, "z": 0.08, "rotation": 0, "holder": None, "settled": True, "target_error_cm": None, "target_locked": False, "release_pose": None, "settle_velocity": 0.0, "settle_locked": False, "grasp_confirmed": False},
    "cup": {"label": "Cup", "color": "#f5c86b", "x": 0.72, "y": 0.69, "z": 0.12, "rotation": 0, "holder": None, "settled": True, "target_error_cm": None, "target_locked": False, "release_pose": None, "settle_velocity": 0.0, "settle_locked": False, "grasp_confirmed": False},
    "fork": {"label": "Fork", "color": "#dce7f5", "x": 0.40, "y": 0.77, "z": 0.06, "rotation": 0, "holder": None, "settled": True, "target_error_cm": None, "target_locked": False, "release_pose": None, "settle_velocity": 0.0, "settle_locked": False, "grasp_confirmed": False},
    "spoon": {"label": "Spoon", "color": "#cad7e8", "x": 0.58, "y": 0.77, "z": 0.06, "rotation": 0, "holder": None, "settled": True, "target_error_cm": None, "target_locked": False, "release_pose": None, "settle_velocity": 0.0, "settle_locked": False, "grasp_confirmed": False},
}

GRASP_PROFILES = {
    # These heights place the visual jaw pads (30 cm below the wrist in the
    # browser workcell) on the exact object center before a grasp is confirmed.
    "blue_plate": {"jaw_width": 1.16, "grasp_width": 1.16, "half_height": 0.055, "hand_z": 0.139, "grasp_target": "plate_center"},
    "cup": {"jaw_width": 0.54, "grasp_width": 0.54, "half_height": 0.20, "hand_z": 0.241, "grasp_target": "cup_center"},
    "fork": {"jaw_width": 0.72, "grasp_width": 0.07, "half_height": 0.035, "hand_z": 0.125, "grasp_target": "handle_center"},
    "spoon": {"jaw_width": 0.72, "grasp_width": 0.07, "half_height": 0.035, "hand_z": 0.125, "grasp_target": "handle_center"},
}

TARGET_POSITIONS = {
    "left_place": {"label": "Left place", "x": 0.35, "y": 0.42},
    "right_place": {"label": "Right place", "x": 0.65, "y": 0.42},
    "center_place": {"label": "Center place", "x": 0.50, "y": 0.42},
    "cup_interior": {"label": "Inside cup", "x": 0.72, "y": 0.69},
}

ARM_HOME = {
    "left": {"x": 0.24, "y": 0.16, "z": 0.72, "angle": -25},
    "right": {"x": 0.76, "y": 0.16, "z": 0.72, "angle": 25},
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def minimum_jerk(progress: float) -> float:
    """Quintic easing with zero velocity and acceleration at both endpoints."""
    t = _clamp(progress, 0.0, 1.0)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _blend(start: float, end: float, progress: float) -> float:
    return start + (end - start) * minimum_jerk(progress)


def _grip_aperture(object_id: str, closed: bool) -> float:
    """Return the jaw center offset from the object geometry in workcell units."""
    profile = GRASP_PROFILES.get(object_id, {})
    grasp_width = float(profile.get("grasp_width", profile.get("jaw_width", 0.30)))
    pad_half_width = 0.0525
    if closed:
        return round(max(pad_half_width + 0.006, grasp_width / 2 + pad_half_width - 0.002), 5)
    return round(grasp_width / 2 + pad_half_width + 0.008, 5)


class TableSettingSimulation:
    """Deterministic planner/simulator boundary with a timestamped 60 Hz motion contract."""

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
            self._paused = False
            self.status = "idle"
            self.plan: Plan | None = None
            self.transcript = {"text": "", "status": "waiting", "source": "typed input / browser speech"}
            self.objects = copy.deepcopy(OBJECT_HOME)
            self.events: list[dict[str, Any]] = []
            self.metrics = {
                "commands": 0,
                "started_runs": 0,
                "successful_runs": 0,
                "recovery_count": 0,
                "plan_latency_ms": 0,
                "last_run_ms": None,
                "sim_fps": SIM_HZ,
                "success_rate": 0,
            }
            self.motion = {
                "revision": 0,
                "elapsed_ms": 0,
                "duration_ms": RUN_DURATION_MS,
                "sample_hz": SIM_HZ,
                "updated_at": utc_now(),
                "phase": "parked",
                "release_revision": None,
                "settle_progress": 0.0,
                "return_progress": 0.0,
            }
            self.physics = {
                "engine": "rapier3d-browser",
                "status": "ready",
                "timestep_hz": SIM_HZ,
                "contact_count": 0,
                "collision_count": 0,
                "last_contact": None,
                "grasp_lock": False,
                "drop_lock": False,
                "settle_steps": 0,
                "settle_velocity": 0.0,
                "grasp_mode": None,
                "left_pad_contact": False,
                "right_pad_contact": False,
                "grasp_distance_mm": None,
                "grasp_attempts": 0,
                "smoothness_max_step_mm": 0.0,
            }
            self._append_event("system", "VoxHands simulator ready. Both hands are parked.")

    def _append_event(self, kind: str, message: str) -> None:
        self.events.append({"time": utc_now(), "kind": kind, "message": message})
        self.events = self.events[-40:]

    def _touch_motion(self, elapsed_ms: int, phase: str | None = None) -> None:
        self.motion["revision"] += 1
        self.motion["elapsed_ms"] = max(0, min(self.motion["duration_ms"], int(elapsed_ms)))
        current_phase = phase or str(self.motion.get("phase", "parked"))
        self.motion["phase"] = current_phase
        percent = elapsed_ms / RUN_DURATION_MS * 100
        self.motion["settle_progress"] = round(_clamp((percent - 92) / 3, 0, 1), 3)
        self.motion["return_progress"] = round(_clamp((percent - 96) / 4, 0, 1), 3)
        self.motion["updated_at"] = utc_now()

    def submit_command(self, raw_text: str) -> dict[str, Any]:
        planning_started = time.perf_counter()
        plan = build_plan(raw_text)
        issues = validate_plan(plan)
        with self._lock:
            if plan.intent == "calibration":
                self._append_event("system", "Calibration: simulation coordinates only; no camera or hardware calibration available.")
                return self.snapshot()
            if self.status in {"running", "paused", "stopped"}:
                return {"error": "Finish the active run, or reset after a stop before a new command."}
            if plan.intent == "home":
                self.plan = None
                self.status = "idle"
                self._touch_motion(0, "parked")
                self._append_event("control", "Simulated arms parked. Object positions preserved.")
                return self.snapshot()
            self._routes, self._offsets = {}, {}
            parallel = len(plan.actions) == 2 and {a.object_id for a in plan.actions} == {"blue_plate", "cup"} and all(
                (a.object_id, a.target_id) in {("blue_plate", "left_place"), ("cup", "right_place")} for a in plan.actions)
            parallel = parallel and self.objects["blue_plate"]["x"] <= .35 and self.objects["cup"]["x"] >= .65
            scheduled_ms = 0
            for index, action in enumerate(plan.actions):
                parent = TARGETS.get(action.target_id, {}).get("container")
                if parent and any(a.object_id == parent for a in plan.actions):
                    issues.append("Move the destination object first, then submit its placement task separately.")
                if any(o.get("container") == action.object_id or o.get("supported_by") == action.object_id for o in self.objects.values()):
                    issues.append(f"Remove objects from {action.object_label} before moving it.")
                start, target = self.objects[action.object_id], self._target_position(action)
                # A destination's supports sit below it, not inside it. Exempt
                # only recorded ancestors; unrelated overlapping objects still block.
                supports = set()
                ancestor = parent
                while ancestor in self.objects and ancestor not in supports:
                    supports.add(ancestor)
                    ancestor = self.objects[ancestor].get("supported_by")
                for other_id, other in self.objects.items():
                    if other_id in supports:
                        continue
                    if other_id != action.object_id and abs(other["x"]-target["x"]) < .02 and abs(other["y"]-target["y"]) < .03:
                        issues.append(f"{action.target_label} is occupied by {other['label']}. Move it first or choose another destination.")
                route = [(start["x"], start["y"]), (target["x"], target["y"])]
                if not self._route_clear(route):
                    route = [route[0], (start["x"], .30), (target["x"], .30), route[-1]]
                if not self._route_clear(route):
                    issues.append(f"No clear simulated center path for {action.object_label}; reset the workcell.")
                self._routes[action.id] = route
                action.duration_ms = RUN_DURATION_MS * (2 if len(route) > 2 else 1)
                self._offsets[action.id] = 0 if parallel else scheduled_ms
                scheduled_ms += action.duration_ms
                action.note = "Separate lanes in parallel" if parallel else f"Sequential move {index+1}"
            self.metrics["plan_latency_ms"] = round((time.perf_counter()-planning_started)*1000, 3)
            self._run_token += 1
            token = self._run_token
            self.metrics["commands"] += 1
            self.transcript = {"text": plan.raw_text, "status": "final", "source": "typed input / browser speech"}
            self.plan = plan
            plan.safety_issues = issues
            if issues:
                self.status = "blocked"
                plan.status = "blocked"
                self._append_event("safety", "Command blocked: " + " ".join(issues))
                return self.snapshot()

            self.metrics["started_runs"] += 1
            self.metrics["success_rate"] = round(self.metrics["successful_runs"] / self.metrics["started_runs"] * 100)
            self.status = "running"
            self._paused = False
            plan.status = "executing"
            self.motion = {
                "revision": 0,
                "elapsed_ms": 0,
                "duration_ms": max((self._offsets[a.id] + a.duration_ms for a in plan.actions), default=RUN_DURATION_MS),
                "sample_hz": SIM_HZ,
                "updated_at": utc_now(),
                "phase": "reaching",
                "release_revision": None,
                "settle_progress": 0.0,
                "return_progress": 0.0,
            }
            self._append_event("voice", f'Heard: “{plan.raw_text}”')
            self._append_event("planner", f"Plan {plan.id} validated with {len(plan.actions)} action(s).")
            self._append_event("safety", "Simulation center paths checked against the barrier; shared moves serialized.")
            self._worker = threading.Thread(target=self._execute, args=(token, plan), daemon=True)
            self._worker.start()
            return self.snapshot()

    @staticmethod
    def _route_clear(route):
        # 2D center clearance only, not full robot-link geometry.
        for start, end in zip(route, route[1:]):
            for step in range(201):
                t = step / 200
                x, y = start[0] + (end[0]-start[0])*t, start[1] + (end[1]-start[1])*t
                if .425 <= x <= .575 and .515 <= y <= .805:
                    return False
        return True

    @staticmethod
    def _route_pose(route, progress):
        scaled = min(len(route)-1-1e-9, max(0, progress)*(len(route)-1))
        index = int(scaled)
        return tuple(_blend(route[index][axis], route[index+1][axis], scaled-index) for axis in (0, 1))

    def pause(self) -> dict[str, Any]:
        with self._lock:
            if self.status == "running" and self.plan:
                self._paused = True
                self.status = "paused"
                self.plan.status = "paused"
                self._append_event("control", "Motion paused by operator. Current arm poses are held safely.")
            return self.snapshot()

    def resume(self) -> dict[str, Any]:
        with self._lock:
            if self.status == "paused" and self.plan:
                self._paused = False
                self.status = "running"
                self.plan.status = "executing"
                self._append_event("control", "Motion resumed by operator.")
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self.status in {"running", "paused"} and self.plan:
                self._run_token += 1
                self._paused = False
                self.status = "stopped"
                self.plan.status = "stopped"
                self.metrics["recovery_count"] += 1
                self._append_event("safety", "All motion stopped by operator. Reset before starting another run.")
            return self.snapshot()

    def _phase_for(self, progress: float) -> str:
        for threshold, phase in PHASE_THRESHOLDS:
            if progress < threshold:
                return phase
        return "parked"

    def _set_object_pose(
        self,
        object_id: str,
        x: float,
        y: float,
        z: float,
        rotation: float,
        holder: str | None,
        settled: bool,
        grasp_confirmed: bool = False,
        release_pose: dict[str, float] | None = None,
        settle_locked: bool = False,
        target: dict[str, float] | None = None,
    ) -> None:
        object_state = self.objects[object_id]
        target_error_cm = None if target is None else round((((x - target["x"]) * 5) ** 2 + ((y - target["y"]) * 3.3) ** 2) ** 0.5 * 100, 3)
        object_state.update({
            "x": round(_clamp(x, 0.02, 0.98), 5),
            "y": round(_clamp(y, 0.02, 0.98), 5),
            "z": round(_clamp(z, 0.02, 0.8), 5),
            "rotation": round(rotation, 5),
            "holder": holder,
            "settled": settled,
            "motion": "settled" if settled else ("held" if holder else "moving"),
            "target_error_cm": target_error_cm,
            "target_locked": target_error_cm is not None and target_error_cm <= PLACEMENT_TOLERANCE_M * 100,
            "release_pose": copy.deepcopy(release_pose),
            "settle_velocity": 0.0,
            "settle_locked": settle_locked,
            "grasp_confirmed": grasp_confirmed,
        })
        object_state["pose"] = {"x": object_state["x"], "y": object_state["y"], "z": object_state["z"]}

    def _target_position(self, action_or_target: Any) -> dict[str, float]:
        target_id = action_or_target if isinstance(action_or_target, str) else action_or_target.target_id
        if target_id == "plate_surface":
            plate = self.objects["blue_plate"]
            return {"x": plate["x"], "y": plate["y"]}
        if target_id == "cup_interior":
            cup = self.objects["cup"]
            return {"x": cup["x"], "y": cup["y"], "z": round(cup["z"] + 0.10, 5)}
        return TARGET_POSITIONS[target_id]

    def _update_action(self, action: Any, start: dict[str, float], progress: float) -> None:
        phase = self._phase_for(progress)
        action.progress = round(progress, 3)
        action.status = "placed" if progress >= 100 else phase
        target = self._target_position(action)
        target_z = target.get("z", OBJECT_HOME[action.object_id]["z"])
        if action.target_id == "plate_surface":
            target_z += .048 / 2.1
        if progress >= GRIP_CONFIRM_PROGRESS:
            self.objects[action.object_id].pop("container", None)
            self.objects[action.object_id].pop("supported_by", None)
        grasp_confirmed = GRIP_CONFIRM_PROGRESS <= progress < 90
        release_pose = {"x": target["x"], "y": target["y"], "z": target_z, "rotation": 0} if progress >= 90 else None
        settle_locked = progress >= 92
        if progress < 25:
            object_x, object_y = start["x"], start["y"]
            z = start["z"]
            holder = None
            settled = True
        elif progress < 42:
            object_x, object_y = start["x"], start["y"]
            z = start["z"]
            holder = action.arm if grasp_confirmed else None
            settled = not grasp_confirmed
        elif progress < 82:
            carry_progress = _clamp((progress - 46) / 36, 0, 1)
            object_x, object_y = self._route_pose(self._routes[action.id], carry_progress)
            z = _blend(start["z"], 0.28, (progress-42)/4)
            holder = action.arm
            settled = False
        elif progress < 90:
            object_x, object_y = target["x"], target["y"]
            z = _blend(0.28, target_z + 0.08, (progress - 82) / 8)
            holder = action.arm
            settled = False
        elif progress < 92:
            object_x, object_y = target["x"], target["y"]
            z = _blend(target_z + 0.08, target_z, (progress - 90) / 2)
            holder = action.arm
            settled = False
        elif progress < 95:
            object_x, object_y = target["x"], target["y"]
            z = target_z
            holder = None
            settled = False
        elif progress < 100:
            object_x, object_y = target["x"], target["y"]
            z = target_z
            holder = None
            settled = True
        else:
            object_x, object_y = target["x"], target["y"]
            z = target_z
            holder = None
            settled = True
            action.status = "placed"
            action.progress = 100.0
        self._set_object_pose(
            action.object_id,
            object_x,
            object_y,
            z,
            0 if settled or progress >= 90 else progress / 100 * 0.08,
            holder,
            settled,
            grasp_confirmed,
            release_pose,
            settle_locked,
            target,
        )
        if action.target_id == "cup_interior" and progress >= 100:
            self.objects[action.object_id]["container"] = "cup"
        if action.target_id == "plate_surface" and progress >= 92:
            self.objects[action.object_id]["supported_by"] = "blue_plate"

    def _execute(self, token: int, plan: Plan) -> None:
        started = time.perf_counter()
        with self._lock:
            if token != self._run_token:
                return
            starts = {
                action.id: {
                    "x": self.objects[action.object_id]["x"],
                    "y": self.objects[action.object_id]["y"],
                    "z": self.objects[action.object_id]["z"],
                }
                for action in plan.actions
            }
            for action in plan.actions:
                action.status = "queued"
            self._append_event("vision", "Simulated scene loaded: predefined tabletop object positions.")
            self._append_event("control", "60 Hz minimum-jerk trajectories scheduled in parallel where safe.")

        next_tick = started
        active_elapsed_ms = 0.0
        last_phase = ""
        release_announced = False
        while True:
            next_tick += SIM_DT
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            with self._lock:
                if token != self._run_token:
                    return
                if self._paused:
                    next_tick = time.perf_counter()
                    continue
                active_elapsed_ms = min(self.motion["duration_ms"], active_elapsed_ms + SIM_DT * 1000)
                elapsed_ms = round(active_elapsed_ms)
                for action in plan.actions:
                    local_ms = elapsed_ms - self._offsets[action.id]
                    if local_ms > 0 and action.status != "placed":
                        self._update_action(action, starts[action.id], min(100, local_ms / action.duration_ms * 100))
                current = next((a for a in plan.actions if a.status not in {"queued", "placed"}), plan.actions[-1])
                progress = current.progress
                phase = self._phase_for(progress)
                self._touch_motion(elapsed_ms, phase)
                if phase != last_phase and phase in {"gripping", "carrying", "placing", "releasing", "settling", "returning"}:
                    self._append_event("control", f"{phase.title()} phase · trajectories {round(progress)}% complete.")
                    last_phase = phase
                if phase == "releasing" and not release_announced:
                    self.motion["release_revision"] = self.motion["revision"]
                    self._append_event("control", "Grippers opening at exact target centers; release handoff confirmed.")
                    release_announced = True
                if elapsed_ms >= self.motion["duration_ms"]:
                    break

        with self._lock:
            if token != self._run_token:
                return
            self._touch_motion(self.motion["duration_ms"], "parked")
            for action in plan.actions:
                self._update_action(action, starts[action.id], 100)
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            plan.status = "complete"
            self.status = "complete"
            self.metrics["successful_runs"] += 1
            self.metrics["last_run_ms"] = elapsed_ms
            self.metrics["success_rate"] = round(self.metrics["successful_runs"] / self.metrics["started_runs"] * 100)
            self._append_event("physics", "Simulated placement complete; browser settling is reported separately.")
            self._append_event("success", f"Task complete in {elapsed_ms} ms. Both hands returned smoothly to safe idle.")

    def record_physics_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record browser telemetry without allowing it to mutate task safety or status."""
        kind = str(payload.get("kind", "contact"))[:24]
        pair = str(payload.get("pair", "scene contact"))[:80]
        with self._lock:
            if kind == "telemetry":
                self.physics["grasp_lock"] = bool(payload.get("grasp_lock", False))
                self.physics["drop_lock"] = bool(payload.get("drop_lock", False))
                try:
                    settle_steps = int(payload.get("settle_steps", 0))
                except (TypeError, ValueError):
                    settle_steps = 0
                settle_velocity = self._finite_float(payload.get("settle_velocity", 0.0), 0.0)
                grasp_distance_mm = self._finite_float(payload.get("grasp_distance_mm"), None)
                smoothness_max_step_mm = self._finite_float(payload.get("smoothness_max_step_mm", 0.0), 0.0)
                self.physics["settle_steps"] = max(0, min(12, settle_steps))
                self.physics["settle_velocity"] = round(max(0.0, min(10.0, settle_velocity)), 4)
                self.physics["grasp_mode"] = str(payload.get("grasp_mode"))[:24] if payload.get("grasp_mode") else None
                self.physics["left_pad_contact"] = bool(payload.get("left_pad_contact", False))
                self.physics["right_pad_contact"] = bool(payload.get("right_pad_contact", False))
                self.physics["grasp_distance_mm"] = None if grasp_distance_mm is None else round(max(0.0, min(1000.0, grasp_distance_mm)), 3)
                try:
                    grasp_attempts = int(payload.get("grasp_attempts", 0))
                except (TypeError, ValueError):
                    grasp_attempts = 0
                self.physics["grasp_attempts"] = max(0, min(120, grasp_attempts))
                self.physics["smoothness_max_step_mm"] = round(max(0.0, min(1000.0, smoothness_max_step_mm)), 3)
                return self.snapshot()
            if kind not in {"contact", "collision", "settled"}:
                return self.snapshot()
            self.physics["contact_count"] += 1
            self.physics["last_contact"] = pair
            if kind == "collision":
                if self.plan and payload.get("run_id") == self.plan.id and self.status in {"running", "paused"}:
                    self.stop()
                    self._append_event("safety", "Current-run browser collision triggered a protective stop. Reset required.")
                self.physics["collision_count"] += 1
                self._append_event("physics", f"Browser physics collision observed: {pair}.")
            elif kind == "settled":
                self._append_event("physics", f"Object settled: {pair}.")
            return self.snapshot()

    @staticmethod
    def _finite_float(value: Any, default: float | None) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "transcript": copy.deepcopy(self.transcript),
                "plan": self.plan.to_dict() if self.plan else None,
                "objects": copy.deepcopy(self.objects),
                "targets": {**copy.deepcopy(TARGET_POSITIONS), **{tid: {"label": TARGETS[tid]["label"], **self._target_position(tid)} for tid in ("plate_surface", "cup_interior")}},
                "red_zone": {"x": 0.44, "y": 0.53, "width": 0.12, "height": 0.26},
                "events": copy.deepcopy(self.events),
                "metrics": copy.deepcopy(self.metrics),
                "runtime": copy.deepcopy(self.runtime),
                "motion": copy.deepcopy(self.motion),
                "physics": copy.deepcopy(self.physics),
                "arms": {"left": self._arm_view("left"), "right": self._arm_view("right")},
            }

    def _arm_status(self, arm: str) -> str:
        if not self.plan or self.status == "idle":
            return "parked"
        statuses = [a.status for a in self.plan.actions if a.arm == arm and a.status not in {"queued", "placed"}]
        if self.status == "complete":
            return "parked"
        return statuses[-1] if statuses else "standby"

    def _arm_view(self, arm: str) -> dict[str, Any]:
        home = ARM_HOME[arm]
        parked = {
            "status": self._arm_status(arm), "phase": "parked", "x": home["x"], "y": home["y"], "z": home["z"],
            "angle": home["angle"], "progress": 0, "gripper": "open", "grip_aperture": 0.20,
            "holding": None, "grasp_confirmed": False, "grasp_target": None,
            "pose": {"x": home["x"], "y": home["y"], "z": home["z"], "yaw": home["angle"]},
        }
        if not self.plan or self.status not in {"running", "paused", "stopped"}:
            return parked
        action = next((candidate for candidate in self.plan.actions if candidate.arm == arm and candidate.status not in {"placed", "queued"}), None)
        if action is None:
            return parked
        object_position = self.objects[action.object_id]
        target = self._target_position(action)
        progress = action.progress
        grasp_z = float(GRASP_PROFILES.get(action.object_id, {}).get("hand_z", 0.14))
        carry_z = min(0.42, grasp_z + 0.24)
        if progress < 25:
            ratio = progress / 25
            hand_x = _blend(home["x"], object_position["x"], ratio)
            hand_y = _blend(home["y"], object_position["y"], ratio)
            hand_z = _blend(0.72, grasp_z, ratio)
        elif progress < 42:
            ratio = (progress - 25) / 17
            hand_x, hand_y = object_position["x"], object_position["y"]
            hand_z = _blend(grasp_z, grasp_z, ratio)
        elif progress < 82:
            ratio = (progress - 42) / 40
            hand_x, hand_y = object_position["x"], object_position["y"]
            hand_z = _blend(grasp_z, carry_z, (progress-42)/4)
        elif progress < 90:
            ratio = (progress - 82) / 8
            hand_x, hand_y = target["x"], target["y"]
            hand_z = _blend(carry_z, grasp_z, ratio)
        elif progress < 96:
            hand_x, hand_y, hand_z = target["x"], target["y"], grasp_z
        else:
            ratio = (progress - 96) / 4
            hand_x = _blend(target["x"], home["x"], ratio)
            hand_y = _blend(target["y"], 0.16, ratio)
            hand_z = _blend(grasp_z, home["z"], ratio)
        hand_x, hand_y, hand_z = _clamp(hand_x, 0.02, 0.98), _clamp(hand_y, 0.02, 0.98), _clamp(hand_z, 0.08, 0.8)
        direction = -1 if arm == "left" else 1
        angle = _blend(home["angle"], 0, progress/25) if progress < 25 else _blend(0, home["angle"], (progress-96)/4) if progress >= 96 else 0
        gripper = "closed" if action.status in {"gripping", "carrying", "placing"} else "open"
        grasp_confirmed = bool(object_position.get("grasp_confirmed")) and object_position.get("holder") == arm
        holding = action.object_id if gripper == "closed" and grasp_confirmed else None
        grasp_profile = GRASP_PROFILES.get(action.object_id, {})
        return {
            "status": self._arm_status(arm), "phase": action.status, "x": round(hand_x, 5), "y": round(hand_y, 5),
            "z": round(hand_z, 5), "angle": round(angle, 3), "progress": round(progress, 3), "object_id": action.object_id,
            "gripper": gripper, "grip_aperture": _grip_aperture(action.object_id, gripper == "closed"),
            "holding": holding, "grasp_confirmed": grasp_confirmed,
            "grasp_target": {"object_id": action.object_id, "profile": grasp_profile.get("grasp_target", "object_center")},
            "pose": {"x": round(hand_x, 5), "y": round(hand_y, 5), "z": round(hand_z, 5), "yaw": round(angle, 3)},
        }
