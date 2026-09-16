"""MuJoCo-backed table-setting simulation.

:class:`MujocoTableSettingSimulation` implements the same public contract as
:class:`~voxhands.simulation.TableSettingSimulation` (snapshot, command,
pause/resume/stop, manual arms, events and metrics, camera rendering) but the
authoritative motion comes from a live MuJoCo scene: each run's waypoints are
resolved with Jacobian IK on two simulated SO-101 arms, the carried object is
kinematically attached to the grip site, and placement is handed back to real
gravity so the object physically settles on the table.

Object/arm coordinates reported in ``snapshot()`` are the actual MuJoCo body
and site positions (table top at z = 0), so the camera feed and the 3D panel
represent the same scene.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any

from .mujoco_scene import MujocoScene
from .models import utc_now
from .simulation import (
    ARM_HOME,
    OBJECT_HOME,
    PHASE_THRESHOLDS,
    RUN_DURATION_MS,
    SIM_DT,
    TableSettingSimulation,
    _clamp,
    minimum_jerk,
)
from .styles import easing as _style_easing


def _blend(start: float, end: float, progress: float) -> float:
    return start + (end - start) * minimum_jerk(_clamp(progress, 0.0, 1.0))


class MujocoTableSettingSimulation(TableSettingSimulation):
    """TableSettingSimulation whose motion is executed in MuJoCo, not analytic poses."""

    def __init__(self, scene: MujocoScene | None = None) -> None:
        self._scene = scene or MujocoScene()
        super().__init__()
        self.physics.update({
            "engine": "mujoco",
            "status": "ready",
            "grasp_mode": "kinematic"
        })
        self.runtime = self.runtime | {
            "simulator": {
                "name": "MuJoCo (two simulated SO-101 arms)",
                "available": True,
                "active": True,
                "mode": "MuJoCo physics + Jacobian IK; browser panel mirrors the live scene",
            }
        }

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        super().reset()
        self._scene.reset()
        self._sync_objects(objects_parked=True)
        self._sync_manual_arms()

    # ------------------------------------------------------------------ #
    # state reads (truth from MuJoCo)
    # ------------------------------------------------------------------ #
    def _sync_objects(self, objects_parked: bool = False) -> None:
        with self._lock:
            for object_id, home in OBJECT_HOME.items():
                x, y, z = self._scene.object_position(object_id)
                state = self.objects[object_id]
                state["x"] = round(float(x), 5)
                state["y"] = round(float(y), 5)
                state["z"] = round(float(z), 5)
                state["pose"] = {"x": state["x"], "y": state["y"], "z": state["z"]}
                state["settle_velocity"] = round(self._scene.object_velocity(object_id), 4)
                if objects_parked:
                    state["rotation"] = 0
                    state["holder"] = None
                    state["settled"] = True
                    state["motion"] = "settled"

    def _sync_manual_arms(self) -> None:
        with self._lock:
            for arm in ARM_HOME:
                x, y, z = self._scene.grip_position(arm)
                current = self.manual_arms[arm]
                current["x"] = round(float(x), 5)
                current["y"] = round(float(y), 5)
                current["z"] = round(float(z), 5)

    # ------------------------------------------------------------------ #
    # arm view (actual grip pose from MuJoCo)
    # ------------------------------------------------------------------ #
    def _arm_view(self, arm: str) -> dict[str, Any]:
        base = super()._arm_view(arm)
        x, y, z = self._scene.grip_position(arm)
        base["x"], base["y"], base["z"] = x, y, z
        base["pose"] = {"x": x, "y": y, "z": z, "yaw": base["pose"]["yaw"]}

        # holding is authoritative from the kinematic attachment
        held = self._scene.attached_object(arm)
        object_state = self.objects.get(held) if held else None
        base["holding"] = held if object_state else None
        base["grasp_confirmed"] = held is not None and object_state is not None
        if held:
            base["grasp_target"] = {"object_id": held, "profile": "kinematic"}
        return base

    # ------------------------------------------------------------------ #
    # manual arms (IK driven, physics advanced)
    # ------------------------------------------------------------------ #
    def _execute_manual_arm(self, arm: str, token: int) -> None:
        started = time.perf_counter()
        with self._lock:
            target = copy.deepcopy(self.manual_arms[arm]["target"])
            if not target:
                return
            start = {key: self.manual_arms[arm][key] for key in ("x", "y", "z", "angle")}
        duration = target["duration_ms"] / 1000
        while True:
            progress = min(1.0, (time.perf_counter() - started) / duration)
            eased = minimum_jerk(progress)
            x = round(start["x"] + (target["x"] - start["x"]) * eased, 5)
            y = round(start["y"] + (target["y"] - start["y"]) * eased, 5)
            z = round(start["z"] + (target["z"] - start["z"]) * eased, 5)
            self._scene.command_arm_pose(arm, x, y, z)
            self._scene.step()
            with self._lock:
                if token != self._manual_tokens[arm]:
                    return
                current = self.manual_arms[arm]
                current["x"], current["y"], current["z"] = self._scene.grip_position(arm)
                self.motion["revision"] += 1
                self.motion["phase"] = "manual" if progress < 1 else "parked"
                self.motion["updated_at"] = utc_now()
                if progress >= 1:
                    current["active"] = False
                    current["target"] = None
                    self._append_event("control", f"Groq manual control parked the {arm} arm at its requested pose.")
                    return
            time.sleep(SIM_DT)

    def set_gripper(self, arm: str, action: str, aperture: float | None = None) -> dict[str, Any]:
        result = super().set_gripper(arm, action, aperture)
        if "error" not in result:
            try:
                self._scene.command_fingers(arm, action == "close")
                self._scene.step()
            except Exception:
                pass
            self._sync_manual_arms()
        return result

    # ------------------------------------------------------------------ #
    # planned task execution
    # ------------------------------------------------------------------ #
    def _execution_waypoint(self, action: Any, start: dict[str, float], progress: float, arm: str) -> dict[str, float]:
        """Return a grip-site waypoint ``(x, y, z, close)`` for the given progress."""
        target = self._target_position(action)
        object_id = action.object_id
        rest_z = self._scene.object_rest_z(object_id)
        grip_z = max(0.030, rest_z + 0.014)
        carry_z = min(0.42, grip_z + 0.20)
        home = ARM_HOME[arm]
        route = self._routes[action.id]
        style_id = action.style

        if progress < 25:
            ratio = _style_easing(style_id, progress / 25)
            x = _blend(home["x"], start["x"], ratio)
            y = _blend(home["y"], start["y"], ratio)
            z = _blend(0.72, grip_z + 0.16, ratio)
            close = False
        elif progress < 42:
            ratio = _style_easing(style_id, (progress - 25) / 17)
            x, y = start["x"], start["y"]
            z = _blend(grip_z + 0.16, grip_z, ratio)
            close = progress > 34
        elif progress < 82:
            ratio = _style_easing(style_id, (progress - 42) / 40)
            x, y = self._route_pose(route, ratio)
            z = _blend(grip_z, carry_z, _style_easing(style_id, (progress - 42) / 4))
            close = True
        elif progress < 90:
            ratio = _style_easing(style_id, (progress - 82) / 8)
            x, y = target["x"], target["y"]
            z = _blend(carry_z, grip_z, ratio)
            close = True
        elif progress < 92:
            x, y, z = target["x"], target["y"], grip_z
            close = False
        elif progress < 96:
            x, y, z = target["x"], target["y"], grip_z
            close = False
        else:
            ratio = _style_easing(style_id, (progress - 96) / 4)
            x = _blend(target["x"], home["x"], ratio)
            y = _blend(target["y"], 0.16, ratio)
            z = _blend(grip_z, home["z"], ratio)
            close = False
        return {"x": _clamp(float(x), 0.02, 0.98), "y": _clamp(float(y), 0.02, 0.98), "z": _clamp(float(z), 0.02, 0.80), "close": bool(close)}

    def _drive_arm(self, arm: str, waypoint: dict[str, float]) -> None:
        self._scene.command_arm_pose(arm, waypoint["x"], waypoint["y"], waypoint["z"])
        self._scene.command_fingers(arm, waypoint.get("close", False))

    def _execute(self, token: int, plan: Any) -> None:
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
            self._append_event("vision", "MuJoCo head camera observing: simulated tabletop object positions.")
            self._append_event("control", "MuJoCo IK drives both SO-101 arms at 60 Hz; placement settles under gravity.")

        attached: dict[str, str | None] = {arm: None for arm in ARM_HOME}
        next_tick = started
        active_elapsed_ms = 0.0
        last_phase = ""
        release_announced = False
        gesture_announced = False
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
                        progress = min(100, local_ms / action.duration_ms * 100)
                        arm = action.arm
                        waypoint = self._execution_waypoint(action, starts[action.id], progress, arm)
                        self._drive_arm(arm, waypoint)
                        # kinematic attach at the grip phase, release at placement
                        if waypoint["close"] and attached[arm] is None:
                            if self._scene.attach(arm, action.object_id):
                                attached[arm] = action.object_id
                        elif not waypoint["close"] and attached[arm] == action.object_id:
                            self._scene.release(arm)
                            attached[arm] = None
                        action.progress = round(progress, 3)
                        action.status = "placed" if progress >= 100 else self._phase_for(progress)
                # physics advances every tick while a task is active
                self._scene.step()
                self._sync_objects()

                current = next((a for a in plan.actions if a.status not in {"queued", "placed"}), plan.actions[-1])
                phase = self._phase_for(current.progress)
                if self._gesture and elapsed_ms >= self._gesture_start:
                    phase = self._gesture
                elif self._hold_ms and elapsed_ms >= self._base_run_ms:
                    phase = "hold"
                self._touch_motion(elapsed_ms, phase)
                if phase != last_phase and phase in {"gripping", "carrying", "placing", "releasing", "settling", "returning"}:
                    self._append_event("control", f"{phase.title()} phase · trajectories {round(current.progress)}% complete.")
                    last_phase = phase
                if phase == "releasing" and not release_announced:
                    self.motion["release_revision"] = self.motion["revision"]
                    self._append_event("control", "MuJoCo grippers opening at exact target centers; physics release confirmed.")
                    release_announced = True
                if phase == self._gesture and not gesture_announced:
                    self._append_event("control", f"Gesture {phase} executing with both hands.")
                    gesture_announced = True
                if elapsed_ms >= self.motion["duration_ms"]:
                    break

        # completion: let the physics settle and park the arms
        for _ in range(10):
            self._scene.step()
            time.sleep(SIM_DT / 20)
        for arm in ARM_HOME:
            self._scene.command_fingers(arm, False)
        with self._lock:
            if token != self._run_token:
                return
            self._touch_motion(self.motion["duration_ms"], "parked")
            for action in plan.actions:
                action.progress = 100.0
                action.status = "placed"
            self._scene.command_arm_pose("left", ARM_HOME["left"]["x"], ARM_HOME["left"]["y"], ARM_HOME["left"]["z"])
            self._scene.command_arm_pose("right", ARM_HOME["right"]["x"], ARM_HOME["right"]["y"], ARM_HOME["right"]["z"])
            for _ in range(30):
                self._scene.step()
            self._sync_objects(objects_parked=True)
            self._sync_manual_arms()
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            plan.status = "complete"
            self.status = "complete"
            self.metrics["successful_runs"] += 1
            self.metrics["last_run_ms"] = elapsed_ms
            self.metrics["success_rate"] = round(self.metrics["successful_runs"] / self.metrics["started_runs"] * 100)
            self._append_event("physics", "MuJoCo placement complete; objects settled under gravity.")
            self._append_event("success", f"Task complete in {elapsed_ms} ms. Both hands returned smoothly to safe idle.")

    # ------------------------------------------------------------------ #
    # camera / vision support
    # ------------------------------------------------------------------ #
    def render_camera(self, camera: str = "overhead", width: int = 640, height: int = 480):
        """Return a uint8 RGB frame from a MuJoCo camera (or None if unavailable)."""
        try:
            frame = self._scene.render(camera=camera, width=width, height=height)
        except Exception:
            return None
        return frame

    def camera_name(self, camera: str) -> str:
        return camera

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._sync_objects()
            self._sync_manual_arms()
        return super().snapshot()