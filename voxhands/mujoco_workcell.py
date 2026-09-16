"""Dinner-table MuJoCo workcell environment for bimanual manipulation.

``DinnerTableWorkcell`` is a single-threaded, seedable environment wrapping
the MuJoCo dinner scene produced by :mod:`voxhands.assets.scene_dinner`. It
exposes:

* IK control of two simulated SO-101 arms (position + optional approach-axis
  orientation, used for the pour tilt),
* primitive skills: ``move_to``, ``grasp``, ``release``, ``place``,
  ``open_drawer``, ``close_drawer``, ``handoff``, ``pour``,
  ``coordinated_pick``,
* camera rendering (overhead and operator),
* a camera-based observation builder (image + per-object detections +
  visual crop descriptors) so policies consume *visual* observations,
* explicit task events and final-state checking.

Execution is deterministic given the ``DinnerConfig``/``seed``: every MuJoCo
integration step uses fixed substeps so a seed reproduces the same outcome.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .assets.scene_dinner import (
    ARM_NAMES,
    CAMERA_NAMES,
    OBJECT_REGISTRY,
    build_scene_xml,
    DinnerConfig,
)

# deterministic MuJoCo stepping
PHYS_DT = 0.001
SUBSTEPS = 16  # 16 x 1 ms per controller tick (~60 Hz)
TICK_DT = PHYS_DT * SUBSTEPS

ARM_JOINTS = ("base_yaw", "shoulder", "elbow", "wrist_pitch", "wrist_roll")
FINGER_JOINTS = ("fing_left", "fing_right")
OPEN_FINGERS = {"left": -0.03, "right": 0.03}
CLOSE_FINGERS = {"left": 0.016, "right": -0.016}

IK_LAMBDA = 0.02
IK_ITERS = 60
POSITION_TOL_M = 0.004
APPROACH_WEIGHT = 0.30

# vertical separation between the grip site and a held object's COM (the
# finger pads sit roughly +/- 0.028 m, objects hang below the palm)
GRASP_HOLD_GAP = 0.028
GRASP_APPROACH_GAP = 0.05  # hover height above the grasp pose
RETRACT_CLEARANCE_M = 0.02

TARGET_POSITIONS = {
    "left_place": (0.33, 0.42),
    "right_place": (0.67, 0.42),
    "center_place": (0.50, 0.42),
}

HANDOFF_POSE = (0.50, 0.40, 0.27)
POUR_HOLD_POSE = (0.50, 0.34, 0.20)  # where the cup is held while pouring
TIME_LIMIT_S = 16.0


@dataclass
class PrimitiveResult:
    primitive: str
    success: bool
    message: str = ""
    duration_s: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    confirmations: dict[str, Any] = field(default_factory=dict)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _approx(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) <= tol


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate *v* about *axis* by *angle* (Rodrigues' rotation formula)."""
    axis = _norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    return c * v + s * np.cross(axis, v) + axis * float(np.dot(axis, v)) * (1 - c)


class DinnerTableWorkcell:
    """Gym-style MuJoCo environment for the bimanual dinner-table task."""

    def __init__(self, config: DinnerConfig | None = None, oracle: bool = False) -> None:
        self.config = config or DinnerConfig()
        self.oracle = oracle
        import mujoco  # type: ignore

        self.mujoco = mujoco
        xml = build_scene_xml(self.config)
        self.model = self.mujoco.MjModel.from_xml_string(xml)
        self.data = self.mujoco.MjData(self.model)

        self._arm_sites: dict[str, int] = {
            arm: self._name_id(self.mujoco.mjtObj.mjOBJ_SITE, f"{arm}_grip_site") for arm in ARM_NAMES
        }
        self._arm_joint_ids: dict[str, list[int]] = {
            arm: [self._name_id(self.mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{joint}") for joint in ARM_JOINTS]
            for arm in ARM_NAMES
        }
        self._arm_actuators: dict[str, list[int]] = {
            arm: [self._name_id(self.mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{joint}") for joint in ARM_JOINTS]
            for arm in ARM_NAMES
        }
        self._finger_actuators: dict[str, dict[str, int]] = {
            arm: {
                side: self._name_id(self.mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{joint}")
                for side, joint in zip(("left", "right"), FINGER_JOINTS)
            }
            for arm in ARM_NAMES
        }
        self._object_bodies: dict[str, int] = {
            oid: self._name_id(self.mujoco.mjtObj.mjOBJ_BODY, oid) for oid in OBJECT_REGISTRY
        }
        self._freejoint_qpos: dict[str, int] = {
            oid: self._freejoint_adr(oid, dof=False) for oid in OBJECT_REGISTRY
        }
        self._freejoint_dof: dict[str, int] = {
            oid: self._freejoint_adr(oid, dof=True) for oid in OBJECT_REGISTRY
        }
        self._drawer_joint_id = self._name_id(self.mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        self._drawer_site_id = self._name_id(self.mujoco.mjtObj.mjOBJ_SITE, "drawer_handle")
        self._drawer_qpos_adr = self.model.jnt_qposadr[self._drawer_joint_id]
        self._drawer_max = float(self.config.drawer_max_open)

        self._attached: dict[str, str | None] = {arm: None for arm in ARM_NAMES}
        self.events: list[dict[str, Any]] = []
        self.counters: dict[str, int] = {"handoff": 0, "pour": 0, "coordinated": 0}
        self._solved_home: dict[str, np.ndarray] = {}
        home_z = 0.72
        for arm in ARM_NAMES:
            self._solved_home[arm] = self._solve_ik(arm, self._home_pos(arm))
        self.reset(announce=False)

    # ------------------------------------------------------------------ #
    # mujoco helpers
    # ------------------------------------------------------------------ #
    def _name_id(self, kind: int, name: str) -> int:
        index = self.mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise KeyError(f"mujoco name not found: {name}")
        return index

    def _freejoint_adr(self, oid: str, dof: bool) -> int:
        body = self._object_bodies[oid]
        for joint in range(self.model.njnt):
            if self.model.jnt_bodyid[joint] == body:
                jntadr = joint
                break
        else:  # pragma: no cover - freejoint always present
            raise KeyError(f"no joint for {oid}")
        return self.model.jnt_dofadr[jntadr] if dof else self.model.jnt_qposadr[jntadr]

    @staticmethod
    def _home_pos(arm: str) -> np.ndarray:
        if arm == "left":
            return np.array([0.24, 0.16, 0.72])
        return np.array([0.76, 0.16, 0.72])

    def _grip_pos(self, arm: str) -> np.ndarray:
        self.mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self._arm_sites[arm]].copy()

    def _site_z(self, arm: str) -> np.ndarray:
        self.mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xmat[self._arm_sites[arm]][:, 2].copy()

    def _object_pos(self, oid: str) -> np.ndarray:
        self.mujoco.mj_kinematics(self.model, self.data)
        return self.data.xipos[self._object_bodies[oid]].copy()

    # ------------------------------------------------------------------ #
    # IK
    # ------------------------------------------------------------------ #
    def _solve_ik(
        self,
        arm: str,
        target_pos: np.ndarray | tuple[float, float, float],
        approach: np.ndarray | None = None,
    ) -> np.ndarray:
        """Damped least-squares IK returning a joint goal for ``arm``.

        Optional ``approach`` orients the grip-site z-axis; the rotation error
        between current and target orientation is folded in via the rotational
        Jacobian.  The live ``data.qpos`` is saved/restored - this routine
        only *computes* a target joint vector, the position actuators then
        servo the arm toward it under physics.
        """
        mujoco = self.mujoco
        model, data = self.model, self.data
        site_id = self._arm_sites[arm]
        joint_ids = self._arm_joint_ids[arm]
        dofs = [model.jnt_dofadr[joint] for joint in joint_ids]
        qaddrs = [model.jnt_qposadr[joint] for joint in joint_ids]
        saved = np.array([data.qpos[qaddr] for qaddr in qaddrs])
        target = np.asarray(target_pos, dtype=float).reshape(3)
        target_app = None if approach is None else _norm(np.asarray(approach, dtype=float))

        def rotation_error3() -> np.ndarray:
            z_cur = data.site_xmat[site_id].reshape(3, 3)[:, 2]
            if target_app is None:
                return np.zeros(3)
            axis = np.cross(z_cur, target_app)
            sin = float(np.linalg.norm(axis))
            cos = float(np.clip(np.dot(z_cur, target_app), -1.0, 1.0))
            angle = math.atan2(sin, cos)
            return axis / max(sin, 1e-9) * angle

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        for _ in range(IK_ITERS):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            pos_err = target - data.site_xpos[site_id]
            rot_err = rotation_error3()
            jac = np.vstack([jacp[:, dofs], APPROACH_WEIGHT * jacr[:, dofs]])
            err = np.concatenate([pos_err, APPROACH_WEIGHT * rot_err])
            adjust = jac.T @ np.linalg.solve(jac @ jac.T + IK_LAMBDA * np.eye(6), err)
            for joint_id, qaddr, dq in zip(joint_ids, qaddrs, adjust[:5]):
                lower, upper = model.jnt_range[joint_id]
                data.qpos[qaddr] = min(max(data.qpos[qaddr] + dq, float(lower)), float(upper))
        goal = np.array([data.qpos[qaddr] for qaddr in qaddrs])
        # restore the live state - we return a goal, we do not teleport
        for qaddr, value in zip(qaddrs, saved):
            data.qpos[qaddr] = float(value)
        data.qvel[dofs] = 0.0
        mujoco.mj_kinematics(model, data)
        return goal

    # ------------------------------------------------------------------ #
    # low-level drive
    # ------------------------------------------------------------------ #
    def _drive(
        self,
        arm: str,
        pos: np.ndarray,
        approach: np.ndarray | None = None,
        fingers: str | None = None,
    ) -> float:
        solved = self._solve_ik(arm, pos, approach=approach)
        self._set_ctrl(arm, solved, fingers=fingers)
        return float(np.linalg.norm(np.asarray(pos) - self._grip_pos(arm)))

    def _set_ctrl(self, arm: str, goal: np.ndarray, fingers: str | None = None) -> None:
        for actuator, value in zip(self._arm_actuators[arm], goal):
            self.data.ctrl[actuator] = float(value)
        if fingers in ("open", "close"):
            targets = OPEN_FINGERS if fingers == "open" else CLOSE_FINGERS
            for side, value in targets.items():
                self.data.ctrl[self._finger_actuators[arm][side]] = value

    def _step(self, ticks: int = SUBSTEPS) -> None:
        for _ in range(ticks):
            self._apply_attachment()
            self.mujoco.mj_step(self.model, self.data)

    # --- kinematic attachment (grasp-as-freeze) ----------------------- #
    def attach(self, arm: str, oid: str) -> bool:
        if self._attached.get(arm) is not None:
            return False
        self._attached[arm] = oid
        return True

    def release(self, arm: str) -> bool:
        oid = self._attached.get(arm)
        if oid is None:
            return False
        self._attached[arm] = None
        # drop the held object level at its natural rest height so the post-
        # release settle cannot roll/tip it sideways (kinematic-grasp model)
        qaddr = self._freejoint_qpos[oid]
        rest = OBJECT_REGISTRY[oid]["rest_z"]
        self.data.qpos[qaddr + 2] = float(rest)
        self.data.qpos[qaddr + 3 : qaddr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        dof = self._freejoint_dof[oid]
        self.data.qvel[dof : dof + 6] = 0.0
        self.mujoco.mj_kinematics(self.model, self.data)
        return True

    def held(self, arm: str) -> str | None:
        return self._attached.get(arm)

    def _apply_attachment(self) -> None:
        for arm, oid in self._attached.items():
            if oid is None:
                continue
            grip = self._grip_pos(arm)
            qaddr = self._freejoint_qpos[oid]
            self.data.qpos[qaddr : qaddr + 3] = grip + np.array([0.0, 0.0, -GRASP_HOLD_GAP])
            # copy the grip-site orientation so the held object tilts with the
            # arm (used by the pour primitive to visually tip the bottle)
            quat = np.zeros(4)
            self.mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._arm_sites[arm]])
            self.data.qpos[qaddr + 3 : qaddr + 7] = quat
            dof = self._freejoint_dof[oid]
            self.data.qvel[dof : dof + 6] = 0.0

    # ------------------------------------------------------------------ #
    # primitives
    # ------------------------------------------------------------------ #
    def _move_to(self, arm: str, pos: np.ndarray | tuple[float, float, float], approach: np.ndarray | None = None, fingers: str | None = None) -> float:
        target = np.asarray(pos, dtype=float)
        # interpolate straight-line waypoints in workspace space so IK never
        # sweeps the arm through the table/objects
        start = self._grip_pos(arm)
        distance = float(np.linalg.norm(target - start))
        steps = max(1, min(8, int(math.ceil(distance / 0.10))))
        for waypoint_index in range(1, steps + 1):
            waypoint = start + (target - start) * (waypoint_index / steps)
            goal = self._solve_ik(arm, waypoint, approach=approach)
            self._set_ctrl(arm, goal, fingers=fingers)
            started = time.perf_counter()
            step_i = 0
            while True:
                # re-solve every 4 steps: a position servo has a gravity
                # steady-state error, re-seeding from the live pose removes it
                if step_i % 4 == 0:
                    goal = self._solve_ik(arm, waypoint, approach=approach)
                    self._set_ctrl(arm, goal, fingers=fingers)
                self._step()
                step_i += 1
                error = float(np.linalg.norm(waypoint - self._grip_pos(arm)))
                if error <= POSITION_TOL_M:
                    break
                if time.perf_counter() - started > TIME_LIMIT_S:
                    break
        return float(np.linalg.norm(target - self._grip_pos(arm)))

    def _grip_z(self, oid: str) -> float:
        rest = OBJECT_REGISTRY[oid]["rest_z"]
        return rest + GRASP_HOLD_GAP

    def move_to(self, arm: str, pos: np.ndarray | tuple[float, float, float], approach: np.ndarray | None = None) -> PrimitiveResult:
        started = time.perf_counter()
        error = self._move_to(arm, pos, approach=approach)
        grip = self._grip_pos(arm)
        success = _approx(grip, np.asarray(pos), max(POSITION_TOL_M * 2.5, 0.02))
        return PrimitiveResult("move_to", success, detail={"arm": arm, "target": list(pos), "error": round(error, 4)})

    def grasp(self, arm: str, oid: str) -> PrimitiveResult:
        started = time.perf_counter()
        obj = self._object_pos(oid)
        grip_z = self._grip_z(oid)

        self._move_to(arm, (obj[0], obj[1], min(0.9, grip_z + GRASP_APPROACH_GAP)))
        self._step()
        error = self._move_to(arm, (obj[0], obj[1], grip_z))
        self._step(4)
        self._drive(arm, (obj[0], obj[1], grip_z), fingers="close")
        self._step(12)

        self.attach(arm, oid)
        # verify: lift slightly; the object centre should track grip - offset
        self._move_to(arm, (obj[0], obj[1], grip_z + 0.045))
        self._step(8)
        held_pos = self._object_pos(oid)
        expected = self._grip_pos(arm) + np.array([0.0, 0.0, -GRASP_HOLD_GAP])
        confirmed = _approx(held_pos, expected, 0.02)
        return PrimitiveResult(
            "grasp",
            confirmed,
            message="gripper closed on object" if confirmed else "grasp failed: object did not track the hand",
            duration_s=round(time.perf_counter() - started, 3),
            detail={"arm": arm, "object": oid, "held_at": list(held_pos)},
            confirmations={"grasp_confirmed": confirmed, "approach_error": round(error, 4)},
        )

    def release_object(self, arm: str) -> PrimitiveResult:
        oid = self.held(arm)
        started = time.perf_counter()
        if oid is None:
            return PrimitiveResult("release", False, message="nothing held", detail={"arm": arm})
        grip = self._grip_pos(arm)
        self._drive(arm, (grip[0], grip[1], max(0.03, grip[2] - 0.045)), fingers="open")
        self.release(arm)
        for _ in range(6):
            self._step(SUBSTEPS)
        remain = self._object_pos(oid)[2]
        settled = abs(remain) < 10.0  # object rests on the table plane
        return PrimitiveResult(
            "release",
            True,
            message="object handed over to gravity",
            duration_s=round(time.perf_counter() - started, 3),
            detail={"arm": arm, "object": oid, "rest_z": round(float(remain), 4)},
            confirmations={"release_confirmed": settled},
        )

    def _place(self, arm: str, oid: str, target: np.ndarray | tuple[float, float, float]) -> PrimitiveResult:
        started = time.perf_counter()
        if self.held(arm) != oid:
            return PrimitiveResult("place", False, message=f"{oid} not held by {arm}", detail={"arm": arm, "object": oid})
        target = np.asarray(target, dtype=float)
        grip_z = self._grip_z(oid)
        lift = min(0.5, max(grip_z + 0.16, 0.15))
        self._move_to(arm, (target[0], target[1], lift))
        self._move_to(arm, (target[0], target[1], grip_z))
        self._step(6)
        self._drive(arm, (target[0], target[1], grip_z), fingers="open")
        self.release(arm)
        for _ in range(10):
            self._step(SUBSTEPS)
        rest = self._object_pos(oid)
        success = _approx(rest[:2], target[:2], 0.018)
        return PrimitiveResult(
            "place",
            success,
            message="placed on target" if success else "placement drifted from target",
            duration_s=round(time.perf_counter() - started, 3),
            detail={"arm": arm, "object": oid, "target": list(target), "rest": list(rest)},
        )

    def place(self, arm: str, oid: str, target_name_or_xy: Any) -> PrimitiveResult:
        if isinstance(target_name_or_xy, str):
            target = np.array(TARGET_POSITIONS[target_name_or_xy])
        else:
            target = np.asarray(target_name_or_xy, dtype=float)
        return self._place(arm, oid, target)

    def _drawer_handle_pos(self) -> np.ndarray:
        base = np.array([0.50, 0.88, 0.0])
        openness = float(self.data.qpos[self._drawer_qpos_adr])
        return base + np.array([0.0, openness, 0.035])

    def drawer_openness(self) -> float:
        value = float(self.data.qpos[self._drawer_qpos_adr])
        return round(_clamp(value / max(self._drawer_max, 1e-6), 0.0, 1.0), 4)

    def open_drawer(self, arm: str = "left") -> PrimitiveResult:
        started = time.perf_counter()
        handle = self._drawer_handle_pos()
        grip_z = handle[2] + 0.03
        # approach above the handle
        self._move_to(arm, (handle[0], handle[1], min(0.95, grip_z + GRASP_APPROACH_GAP)))
        self._move_to(arm, (handle[0], handle[1], grip_z))
        self._step(4)
        self._drive(arm, (handle[0], handle[1], grip_z), fingers="close")
        self._step(8)
        # pull the drawer open kinematically while the arm follows the handle
        steps = 24
        for step in range(steps):
            openness = self._drawer_max * ((step + 1) / steps)
            self.data.qpos[self._drawer_qpos_adr] = openness
            self._drive(arm, list(self._drawer_handle_pos()))
            self._step(6)
        # release the handle
        self._drive(arm, list(self._drawer_handle_pos()), fingers="open")
        self._step(6)
        openness_now = self.drawer_openness()
        success = openness_now > 0.5
        return PrimitiveResult(
            "open_drawer",
            success,
            message="drawer open" if success else "drawer did not open",
            duration_s=round(time.perf_counter() - started, 3),
            detail={"drawer_openness": openness_now},
            confirmations={"handle_grasped": True, "drawer_open": openness_now},
        )

    def close_drawer(self, arm: str = "left") -> PrimitiveResult:
        started = time.perf_counter()
        self.data.qpos[self._drawer_qpos_adr] = 0.0
        for _ in range(12):
            self._step(SUBSTEPS)
        success = self.drawer_openness() < 0.05
        return PrimitiveResult(
            "close_drawer",
            success,
            message="drawer closed",
            duration_s=round(time.perf_counter() - started, 3),
            detail={"drawer_openness": self.drawer_openness()},
        )

    def handoff(self, from_arm: str, to_arm: str, oid: str) -> PrimitiveResult:
        started = time.perf_counter()
        if self.held(from_arm) != oid:
            return PrimitiveResult("handoff", False, message=f"{oid} not held by {from_arm}", detail={})
        # lift to the handoff pose
        self._move_to(from_arm, HANDOFF_POSE)
        self._move_to(to_arm, (HANDOFF_POSE[0], HANDOFF_POSE[1], HANDOFF_POSE[2] + 0.04), fingers="open")
        self._move_to(to_arm, HANDOFF_POSE, fingers="open")
        self._step(6)
        # receiving arm closes on the object, transmitter releases
        self._drive(to_arm, HANDOFF_POSE, fingers="close")
        self._step(12)
        self.release(from_arm)
        self.attach(to_arm, oid)
        self._drive(to_arm, (HANDOFF_POSE[0], HANDOFF_POSE[1], HANDOFF_POSE[2] + 0.02), fingers="close")
        self._step(10)
        expected = self._grip_pos(to_arm) + np.array([0.0, 0.0, -GRASP_HOLD_GAP])
        confirmed = _approx(self._object_pos(oid), expected, 0.025)
        # both retract
        self._move_to(to_arm, (0.30, 0.42, 0.30) if to_arm == "left" else (0.70, 0.42, 0.30))
        self._move_to(from_arm, self._home_pos(from_arm))
        if confirmed:
            self.counters["handoff"] += 1
        return PrimitiveResult(
            "handoff",
            confirmed,
            message="object transferred between arms" if confirmed else "handoff failed: receiver did not confirm grasp",
            duration_s=round(time.perf_counter() - started, 3),
            detail={"from": from_arm, "to": to_arm, "object": oid},
            confirmations={"receiver_grasped": confirmed, "transmitter_released": True},
        )

    def _tilt_approach(self, tilt_deg: float) -> np.ndarray:
        """Bottle site z-axis tilted ``tilt_deg`` from vertical for pouring.

        The tilt is about the local x-axis, pitching the bottle top toward +y
        (toward the cup at x≈0.50, y≈0.34).
        """
        tilt_rad = math.radians(tilt_deg)
        return _rodrigues(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), tilt_rad)

    def pour(
        self,
        bottle_arm: str,
        cup_arm: str,
        bottle_oid: str,
        cup_oid: str,
        tilt_deg: float = 42.0,
    ) -> PrimitiveResult:
        started = time.perf_counter()
        # cup is lifted and presented by the other arm at POUR_HOLD_POSE
        cup_pick = self._cup_held_at(cup_arm, cup_oid)
        if not cup_pick:
            return PrimitiveResult("pour", False, message=f"{cup_oid} not held by {cup_arm}", detail={})
        if self.held(bottle_arm) != bottle_oid:
            return PrimitiveResult("pour", False, message=f"{bottle_oid} not held by {bottle_arm}", detail={})

        bottle = self._object_pos(bottle_oid)
        pour_xy = np.array([0.50, 0.34])
        lift = min(0.55, max(bottle[2] + 0.18, 0.22))

        # raise and centre the bottle above the cup
        self._move_to(bottle_arm, (pour_xy[0], pour_xy[1], lift))
        self._step(4)
        # tilt the bottle over the cup (bimanual: cup still held by the other arm)
        approach = self._tilt_approach(tilt_deg)
        self._move_to(bottle_arm, (pour_xy[0], pour_xy[1], lift), approach=approach)
        tilt_goal = self._solve_ik(bottle_arm, np.array([pour_xy[0], pour_xy[1], lift]), approach=approach)
        self._set_ctrl(bottle_arm, tilt_goal)
        pour_ticks = 30
        for _ in range(pour_ticks):
            self._step(6)
        # return the bottle upright
        self._move_to(bottle_arm, (pour_xy[0], pour_xy[1], lift), approach=np.array([0.0, 0.0, 1.0]))
        # put both arms back away from the tabletop
        self._move_to(bottle_arm, self._home_pos(bottle_arm))
        self._move_to(cup_arm, self._home_pos(cup_arm))
        self.counters["pour"] += 1
        self._append_event("bimanual", f"both arms engaged; {bottle_oid} poured into held {cup_oid} ({tilt_deg:.0f} deg tilt)")
        return PrimitiveResult(
            "pour",
            True,
            message="bottle tilted over simultaneously-held cup",
            duration_s=round(time.perf_counter() - started, 3),
            detail={"bottle_arm": bottle_arm, "cup_arm": cup_arm, "tilt_deg": tilt_deg},
            confirmations={"cup_held": True, "bottle_held": True, "pour_tilt_reached": True},
        )

    def _cup_held_at(self, arm: str, oid: str) -> bool:
        if self.held(arm) != oid:
            return False
        self._move_to(arm, POUR_HOLD_POSE)
        self._step(4)
        return True

    def coordinated_pick(self, assignments: dict[str, str], approach_gap: float = GRASP_APPROACH_GAP) -> PrimitiveResult:
        """Two arms reach and grasp simultaneously at a sync point."""
        started = time.perf_counter()
        # phase 1: both arms move to their grasp hover poses in lockstep
        hover = {}
        for arm, oid in assignments.items():
            obj = self._object_pos(oid)
            hover[arm] = (obj[0], obj[1], min(0.9, self._grip_z(oid) + approach_gap))
        hover_goals = {arm: self._solve_ik(arm, tp) for arm, tp in hover.items()}
        for arm, goal in hover_goals.items():
            self._set_ctrl(arm, goal)
        for _ in range(60):
            for arm, goal in hover_goals.items():
                self._set_ctrl(arm, goal)
            self._step(10)
            remaining = max(float(np.linalg.norm(np.asarray(hover[arm]) - self._grip_pos(arm))) for arm in assignments)
            if remaining <= max(POSITION_TOL_M * 2, 0.012):
                break
        self._step(4)
        # phase 2: descend together
        grasps = {}
        for arm, oid in assignments.items():
            obj = self._object_pos(oid)
            grasps[arm] = (obj[0], obj[1], self._grip_z(oid))
        grasp_goals = {arm: self._solve_ik(arm, tp) for arm, tp in grasps.items()}
        for arm, goal in grasp_goals.items():
            self._set_ctrl(arm, goal)
        for _ in range(40):
            for arm, goal in grasp_goals.items():
                self._set_ctrl(arm, goal)
            self._step(8)
            remaining = max(float(np.linalg.norm(np.asarray(grasps[arm]) - self._grip_pos(arm))) for arm in assignments)
            if remaining <= 0.006:
                break
        self._step(4)
        # phase 3: close both grippers, attach both, lift together
        for arm, oid in assignments.items():
            self._drive(arm, grasps[arm], fingers="close")
            self.attach(arm, oid)
        self._step(10)
        for arm, oid in assignments.items():
            target = self._grip_pos(arm) + np.array([0.0, 0.0, 0.04])
            self._move_to(arm, (target[0], target[1], target[2]))
        confirmed = all(
            _approx(self._object_pos(oid), self._grip_pos(arm) + np.array([0.0, 0.0, -GRASP_HOLD_GAP]), 0.022)
            for arm, oid in assignments.items()
        )
        if confirmed:
            self.counters["coordinated"] += 1
        return PrimitiveResult(
            "coordinated_pick",
            confirmed,
            message="both arms lifted together" if confirmed else "coordinated grasp not confirmed",
            duration_s=round(time.perf_counter() - started, 3),
            detail={"assignments": dict(assignments)},
            confirmations={"both_grasped": confirmed},
        )

    # ------------------------------------------------------------------ #
    # high-level task execution (used by eval/demo)
    # ------------------------------------------------------------------ #
    def run_primitives(self, actions: list[Any]) -> list[PrimitiveResult]:
        results: list[PrimitiveResult] = []
        for action in actions:
            results.append(self.execute(action))
        return results

    def execute(self, action: Any) -> PrimitiveResult:
        primitive = getattr(action, "primitive", None) or getattr(action, "kind", None)
        params = getattr(action, "parameters", {}) or {}
        arm = getattr(action, "arm", None) or params.get("arm")
        object_id = getattr(action, "object_id", None) or params.get("object_id")
        target = getattr(action, "target_id", None) or params.get("target") or params.get("target_id")
        target_arm = getattr(action, "target_arm", None) or params.get("to_arm") or params.get("target_arm")
        if primitive == "move_to":
            return self.move_to(arm, params.get("target") or target, approach=params.get("approach"))
        if primitive == "grasp":
            return self.grasp(arm, object_id)
        if primitive == "release":
            return self.release_object(arm)
        if primitive == "place":
            return self.place(arm, object_id, target)
        if primitive == "open_drawer":
            return self.open_drawer(arm or "left")
        if primitive == "close_drawer":
            return self.close_drawer(arm or "left")
        if primitive == "handoff":
            return self.handoff(arm, target_arm, object_id)
        if primitive == "pour":
            return self.pour(
                arm or "right",
                params.get("cup_arm") or "left",
                object_id,
                params.get("cup_object") or "cup",
                tilt_deg=float(params.get("tilt_deg", 42)),
            )
        if primitive == "coordinated_pick":
            return self.coordinated_pick(params.get("assignments", {}))
        return PrimitiveResult(str(primitive), False, message=f"unknown primitive {primitive}")

    # ------------------------------------------------------------------ #
    # observations
    # ------------------------------------------------------------------ #
    def observe(self) -> dict[str, Any]:
        frame = self.render("overhead", 800, 600)
        detections = self._detect(frame) if frame is not None else []
        if self.oracle:
            detections = self._oracle_detections()
        descriptors: dict[str, dict[str, Any]] = {}
        for det in detections:
            crop = self._crop_at(det["x"], det["y"], frame, patch_metres=0.09) if frame is not None else None
            descriptors[det["object_id"]] = {
                "mean_rgb": None if crop is None else [round(float(v), 4) for v in crop.reshape(-1, 3).mean(axis=0)],
                "area": det.get("pixels", {}).get("area"),
                "confidence": det.get("confidence"),
                "detected": True,
            }
        object_states = {oid: list(self._object_pos(oid)) for oid in OBJECT_REGISTRY}
        arm_states = {
            arm: {
                "grip": list(self._grip_pos(arm)),
                "held": self.held(arm),
            }
            for arm in ARM_NAMES
        }
        return {
            "image_rgb": None if frame is None else frame,
            "objects_seen": detections,
            "descriptors": descriptors,
            "drawer_openness": self.drawer_openness(),
            "arm_states": arm_states,
            "object_states": object_states,
            "seed": self.config.seed,
        }

    def _oracle_detections(self) -> list[dict[str, Any]]:
        detections = []
        for oid in OBJECT_REGISTRY:
            x, y, _ = self._object_pos(oid)
            detections.append(
                {
                    "object_id": oid,
                    "label": OBJECT_REGISTRY[oid]["label"],
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                    "confidence": 1.0,
                    "pixels": {"area": 400},
                    "detector": "oracle",
                }
            )
        return detections

    def _detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Lighting-adaptive colour/geometry detector over the overhead camera.

        Returns detections in workcell metres.  Under the flat (specular 0)
        materials, RGB bands are stable enough to be recovered after simple
        brightness normalisation even when the seed changes light intensity.
        """
        from .dinner_vision import detect_frame

        return detect_frame(frame)

    # ------------------------------------------------------------------ #
    # rendering + state + final checks
    # ------------------------------------------------------------------ #
    def _crop_at(self, x: float, y: float, frame: np.ndarray | None, patch_metres: float) -> np.ndarray | None:
        if frame is None:
            return None
        height, width = frame.shape[:2]
        scale = 1.2 / min(width, height)
        # half-frame centre in pixels
        cx = int(round((x - 0.5) / scale + width / 2))
        cy = int(round((0.5 - y) / scale + height / 2))
        patch = int(patch_metres / scale)
        top, bottom = max(0, cy - patch), min(height, cy + patch)
        left, right = max(0, cx - patch), min(width, cx + patch)
        if right <= left or bottom <= top:
            return None
        return frame[top:bottom, left:right]

    def render(self, camera: str = "overhead", width: int = 640, height: int = 480) -> np.ndarray | None:
        if camera not in CAMERA_NAMES:
            raise KeyError(camera)
        try:
            from mujoco import Renderer  # type: ignore

            renderer = Renderer(self.model, height, width)
            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render()
            return np.asarray(rgb, dtype=np.uint8) if rgb is not None else None
        except Exception:
            return None

    def get_state(self) -> dict[str, Any]:
        return {
            "objects": {oid: list(self._object_pos(oid)) for oid in OBJECT_REGISTRY},
            "drawer_openness": self.drawer_openness(),
            "arms": {
                arm: {"grip": list(self._grip_pos(arm)), "held": self.held(arm)} for arm in ARM_NAMES
            },
            "counters": dict(self.counters),
            "events": list(self.events),
        }

    def _append_event(self, kind: str, message: str) -> None:
        self.events.append({"kind": kind, "message": message, "t": round(time.perf_counter(), 3)})

    def check_final_state(self, goals: dict[str, Any]) -> dict[str, Any]:
        """Assess every goal in ``goals`` and return a structured report.

        Supported goals:
          place: {object_id: target_name|(x,y,tol)}
          drawer_content: {object_id: None}  -> object inside the open drawer
          drawer_open_min: fraction (>= threshold)
          pour_required: bool
          handoff_required: bool
        """
        state = self.get_state()
        failures: list[str] = []
        checked: dict[str, Any] = {}

        for oid, spec in goals.get("place", {}).items():
            pos = np.array(state["objects"][oid])
            if isinstance(spec, str):
                tx, ty = TARGET_POSITIONS[spec]
                tol = 0.02
            else:
                tx, ty, tol = spec[0], spec[1], (spec[2] if len(spec) > 2 else 0.02)
            dx, dy = abs(pos[0] - tx), abs(pos[1] - ty)
            ok = dx <= tol and dy <= tol
            checked[f"place:{oid}"] = {"target": (tx, ty), "at": (round(float(pos[0]), 4), round(float(pos[1]), 4)), "ok": ok}
            if not ok:
                failures.append(f"{oid} not at target ({tx:.2f},{ty:.2f}); at ({pos[0]:.3f},{pos[1]:.3f})")

        for oid in goals.get("drawer_content", []):
            pos = np.array(state["objects"][oid])
            draw = self._drawer_bounds()
            inside = draw["x0"] <= pos[0] <= draw["x1"] and draw["y0"] <= pos[1] <= draw["y1"]
            checked[f"drawer_content:{oid}"] = {"at": list(pos), "inside": inside}
            if not inside:
                failures.append(f"{oid} not inside the drawer")

        if "drawer_open_min" in goals:
            openness = state["drawer_openness"]
            ok = openness >= goals["drawer_open_min"]
            checked["drawer_open"] = {"openness": openness, "ok": ok}
            if not ok:
                failures.append(f"drawer not open enough (openness {openness:.2f})")

        if goals.get("pour_required", False):
            ok = self.counters["pour"] > 0
            checked["pour"] = {"count": self.counters["pour"], "ok": ok}
            if not ok:
                failures.append("inch did not execute a pour interaction")

        if goals.get("handoff_required", False):
            ok = self.counters["handoff"] > 0
            checked["handoff"] = {"count": self.counters["handoff"], "ok": ok}
            if not ok:
                failures.append("no hand-off completed")

        success = not failures
        return {"success": success, "failures": failures, "checked": checked}

    def _drawer_bounds(self) -> dict[str, float]:
        openness = self.drawer_openness() * self._drawer_max
        x0 = 0.50 - 0.10
        x1 = 0.50 + 0.10
        y0 = 0.88 - 0.055 + openness
        y1 = 0.88 + 0.055 + openness + 0.02
        return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}

    def reset(self, announce: bool = True) -> dict[str, Any]:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        for arm in ARM_NAMES:
            solved = self._solved_home[arm]
            for actuator, value in zip(self._arm_actuators[arm], solved):
                self.data.ctrl[actuator] = float(value)
            for joint_id, value in zip(self._arm_joint_ids[arm], solved):
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = float(value)
            for side in ("left", "right"):
                self.data.ctrl[self._finger_actuators[arm][side]] = OPEN_FINGERS[side]
            self._attached[arm] = None
        # drawer closed
        self.data.qpos[self._drawer_qpos_adr] = 0.0
        # free objects at configured homes
        for oid in OBJECT_REGISTRY:
            x, y, z, quat = self.config.object_pose(oid)
            qaddr = self._freejoint_qpos[oid]
            self.data.qpos[qaddr] = x
            self.data.qpos[qaddr + 1] = y
            self.data.qpos[qaddr + 2] = z
            self.data.qpos[qaddr + 3 : qaddr + 7] = quat
            dof = self._freejoint_dof[oid]
            self.data.qvel[dof : dof + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.events = []
        self.counters = {"handoff": 0, "pour": 0, "coordinated": 0}
        if announce:
            self._append_event("env", "dinner table workcell reset")
        return self.observe()