"""MuJoCo scene driver for the VoxHands table-setting task.

This module owns ``voxhands/assets/scene.xml`` (two simulated SO-101 arms,
a dining table, four freejoint objects and two cameras) and exposes a small
command surface used by :class:`MujocoTableSettingSimulation`:

* IK control of each 5-DOF arm's grip site (damped least squares on
  ``mj_jacSite``),
* position-servo stepping of the physics,
* a kinematic grasp attachment (the carried object follows the grip site
  exactly; release hands the object back to gravity for settling),
* camera rendering for the vision pipeline.

The scene keeps the browser workcell coordinate convention (x, y in metres,
table top at z = 0) so object and target positions map 1:1 to the plan layer.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .simulation import ARM_HOME, OBJECT_HOME


ASSETS = Path(__file__).resolve().parent / "assets"
SCENE_XML = ASSETS / "scene.xml"

PHYS_DT = 0.001
SUBSTEPS = 16  # 16 x 1ms substeps per 60 Hz tick

ARM_JOINTS = ("base_yaw", "shoulder", "elbow", "wrist_pitch", "wrist_roll")
FINGER_JOINTS = ("fing_left", "fing_right")
OPEN_FINGERS = {"left": -0.03, "right": 0.03}
CLOSE_FINGERS = {"left": 0.016, "right": -0.016}

# MuJoCo resting centre height of each object (body COM on the table top).
OBJECT_REST_Z = {
    "blue_plate": 0.012,
    "cup": 0.06,
    "fork": 0.012,
    "spoon": 0.012,
}

IK_LAMBDA = 0.02
IK_ITERS = 40
GRASP_APPROACH_GAP = 0.09
# Vertical separation between the grip site and a held object's centre (m).
# The finger pads sit at +/- 0.028 m, so 0.026 clears them while keeping the
# object visually gripped between the jaws.
GRASP_HOLD_GAP = 0.028


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class MujocoScene:
    """Wraps the MuJoCo model/data and provides arm + object + camera control.

    All MuJoCo state is guarded by one re-entrant lock so the physics thread,
    the plan worker and the camera/render endpoints can safely interleave.
    """

    def __init__(self, scene_file: Path = SCENE_XML, import_mujoco: Any = None) -> None:
        self._lock = threading.RLock()
        self._renderers: Any = threading.local()
        if import_mujoco is not None:
            import mujoco as _mujoco

            self.mujoco = _mujoco
        else:
            import mujoco

            self.mujoco = mujoco
        self.xml = (scene_file or SCENE_XML).read_text()
        self.model = self.mujoco.MjModel.from_xml_string(self.xml)
        self.data = self.mujoco.MjData(self.model)

        self._arm_sites = {arm: self._name(self.mujoco.mjtObj.mjOBJ_SITE, f"{arm}_grip_site") for arm in ARM_HOME}
        self._arm_joint_ids = {
            arm: [self._name(self.mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{joint}") for joint in ARM_JOINTS]
            for arm in ARM_HOME
        }
        self._arm_actuators = {
            arm: [self._name(self.mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{joint}") for joint in ARM_JOINTS]
            for arm in ARM_HOME
        }
        self._finger_actuators = {
            arm: {
                side: self._name(self.mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{joint}")
                for side, joint in zip(("left", "right"), FINGER_JOINTS)
            }
            for arm in ARM_HOME
        }
        self._object_bodies = {
            object_id: self._name(self.mujoco.mjtObj.mjOBJ_BODY, object_id)
            for object_id in OBJECT_HOME
        }
        self._freejoint_qpos = {
            object_id: self._freejoint_adr(self._object_bodies[object_id])
            for object_id in OBJECT_HOME
        }
        self._freejoint_dof = {
            object_id: self._freejoint_adr(self._object_bodies[object_id], dof=True)
            for object_id in OBJECT_HOME
        }

        # grasp bookkeeping: arm -> object id tied to the grip site
        self._attached: dict[str, str | None] = {arm: None for arm in ARM_HOME}
        self._attach_offsets: dict[str, np.ndarray] = {}

        self._solved_home: dict[str, np.ndarray] = {}
        for arm in ARM_HOME:
            self._solved_home[arm] = self._solve_ik_raw(arm, self._arm_home_pos(arm))
        self.reset(announce=False)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _name(self, kind: int, name: str) -> int:
        index = self.mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise KeyError(f"mujoco name not found: {name}")
        return index

    def _freejoint_adr(self, body_id: int, dof: bool = False) -> int:
        model = self.model
        jntadr = -1
        for joint in range(model.njnt):
            if model.jnt_bodyid[joint] == body_id:
                jntadr = joint
                break
        if jntadr < 0:
            raise KeyError(f"no joint for body id {body_id}")
        return model.jnt_dofadr[jntadr] if dof else model.jnt_qposadr[jntadr]

    @staticmethod
    def _arm_home_pos(arm: str) -> list[float]:
        home = ARM_HOME[arm]
        return [home["x"], home["y"], home["z"]]

    def _arm_grip(self, arm: str) -> np.ndarray:
        return self.data.site_xpos[self._arm_sites[arm]].copy()

    def set_joint_overrides(self, joint_values: dict[str, float]) -> None:
        """Inject pre-solved joint angles (used to park arms at IK solutions)."""
        for joint_name, value in joint_values.items():
            joint_id = self._name(self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = float(value)
        self.mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ #
    # IK
    # ------------------------------------------------------------------ #
    def _solve_ik_raw(self, arm: str, target_pos: Iterable[float]) -> np.ndarray:
        """Damped least-squares IK returning the five revolute joint angles."""
        mujoco = self.mujoco
        model, data = self.model, self.data
        site_id = self._arm_sites[arm]
        joint_ids = self._arm_joint_ids[arm]
        dofs = [model.jnt_dofadr[joint] for joint in joint_ids]
        qaddrs = [model.jnt_qposadr[joint] for joint in joint_ids]
        target = np.asarray(target_pos, dtype=float).reshape(3)
        error = float("inf")
        for _ in range(IK_ITERS):
            mujoco.mj_forward(model, data)
            error = float(np.linalg.norm(target - data.site_xpos[site_id]))
            if error < 5e-4:
                break
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jacobian = jacp[:, dofs]
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + IK_LAMBDA * np.eye(3), target - data.site_xpos[site_id]
            )
            for qaddr, dq in zip(qaddrs, delta):
                data.qpos[qaddr] += dq
        mujoco.mj_forward(model, data)
        return np.array([data.qpos[qaddr] for qaddr in qaddrs])

    def command_arm_pose(self, arm: str, x: float, y: float, z: float) -> float:
        """Solve IK to ``(x, y, z)`` and feed the joint servos.

        Returns the residual distance in metres (grip site to target).
        """
        with self._lock:
            solved = self._solve_ik_raw(arm, (x, y, z))
            error = float(np.linalg.norm(np.asarray([x, y, z]) - self._arm_grip(arm)))
            for actuator_id, value in zip(self._arm_actuators[arm], solved):
                self.data.ctrl[actuator_id] = value
            return error

    def command_fingers(self, arm: str, closed: bool) -> None:
        with self._lock:
            targets = CLOSE_FINGERS if closed else OPEN_FINGERS
            for side, value in targets.items():
                self.data.ctrl[self._finger_actuators[arm][side]] = value

    def finger_opening(self, arm: str) -> float:
        """Estimated jaw opening (metres) from the two finger joint positions."""
        with self._lock:
            model = self.model
            values = [
                self.data.qpos[model.jnt_qposadr[joint]]
                for joint in (
                    self._name(self.mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_fing_left"),
                    self._name(self.mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_fing_right"),
                )
            ]
            return round(0.056 - (values[0] - values[1]), 5)

    # ------------------------------------------------------------------ #
    # grasping (kinematic attachment)
    # ------------------------------------------------------------------ #
    def attach(self, arm: str, object_id: str) -> bool:
        """Freeze an object to the grip site so it tracks the hand exactly.

        The object is held at a fixed offset just below the grip site (world
        frame), so on descent the object settles onto the tabletop at the hand's
        lowest point without penetrating it.
        """
        with self._lock:
            if self._attached.get(arm) is not None or self._attached.get(arm) == object_id:
                return False
            self._attached[arm] = object_id
            self._attach_offsets[arm] = np.array([0.0, 0.0, -GRASP_HOLD_GAP])
            return True

    def release(self, arm: str) -> bool:
        """Hand the carried object back to gravity."""
        with self._lock:
            if self._attached.get(arm) is None:
                return False
            object_id = self._attached[arm]
            self._attached[arm] = None
            self._attach_offsets.pop(arm, None)
            dof = self._freejoint_dof[object_id]
            self.data.qvel[dof : dof + 6] = 0.0
            return True

    def attached_object(self, arm: str) -> str | None:
        return self._attached.get(arm)

    def _apply_attachment(self) -> None:
        for arm, object_id in self._attached.items():
            if object_id is None:
                continue
            offset = self._attach_offsets.get(arm)
            if offset is None:
                continue
            grip = self.data.site_xpos[self._arm_sites[arm]]
            qaddr = self._freejoint_qpos[object_id]
            self.data.qpos[qaddr : qaddr + 3] = grip + offset
            self.data.qpos[qaddr + 3 : qaddr + 7] = (1.0, 0.0, 0.0, 0.0)
            self.data.qvel[self._freejoint_dof[object_id] : self._freejoint_dof[object_id] + 6] = 0.0

    def step(self, substeps: int = SUBSTEPS) -> None:
        """Advance physics ``substeps`` x 1 ms ticks, honouring grasps."""
        with self._lock:
            for _ in range(substeps):
                self._apply_attachment()
                self.mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ #
    # state reads
    # ------------------------------------------------------------------ #
    def object_position(self, object_id: str) -> np.ndarray:
        with self._lock:
            return self.data.xipos[self._object_bodies[object_id]].copy()

    def object_velocity(self, object_id: str) -> float:
        """Resultant COM speed in m/s (physics frame)."""
        with self._lock:
            start = self._freejoint_dof[object_id]
            return float(np.linalg.norm(self.data.cvel[start : start + 3]))

    def grip_position(self, arm: str) -> tuple[float, float, float]:
        with self._lock:
            return tuple(round(float(value), 5) for value in self._arm_grip(arm))

    def object_rest_z(self, object_id: str) -> float:
        return OBJECT_REST_Z[object_id]

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def render(self, camera: str = "overhead", width: int = 640, height: int = 480) -> np.ndarray | None:
        """Return an (h, w, 3) uint8 RGB frame, or None when OpenGL fails.

        MuJoCo's Renderer binds an OpenGL context to the creating thread, so
        each thread renders through its own renderer instance.
        """
        with self._lock:
            try:
                renderer = getattr(self._renderers, "current", None)
                if renderer is None:
                    from mujoco import Renderer  # type: ignore

                    renderer = Renderer(self.model, height, width)
                    self._renderers.current = renderer
                renderer.update_scene(self.data, camera=camera)
                rgb = renderer.render()
                return np.asarray(rgb, dtype=np.uint8) if rgb is not None else None
            except Exception:
                return None

    def camera_names(self) -> list[str]:
        return [self.model.camera(i).name for i in range(self.model.ncam)]

    def reset(self, announce: bool = True) -> None:
        """Restore arms to their parked IK pose and objects to their homes."""
        with self._lock:
            for arm in ARM_HOME:
                solved = self._solved_home[arm]
                for actuator_id, value in zip(self._arm_actuators[arm], solved):
                    self.data.ctrl[actuator_id] = float(value)
                for joint_id, value in zip(self._arm_joint_ids[arm], solved):
                    joint_name = self.model.joint(joint_id).name
                    self.data.qpos[self.model.jnt_qposadr[joint_id]] = float(value)
                for side in ("left", "right"):
                    self.data.ctrl[self._finger_actuators[arm][side]] = OPEN_FINGERS[side]
                self._attached[arm] = None
            for object_id in OBJECT_HOME:
                home = OBJECT_HOME[object_id]
                qaddr = self._freejoint_qpos[object_id]
                self.data.qpos[qaddr] = home["x"]
                self.data.qpos[qaddr + 1] = home["y"]
                self.data.qpos[qaddr + 2] = OBJECT_REST_Z[object_id]
                self.data.qpos[qaddr + 3 : qaddr + 7] = (1.0, 0.0, 0.0, 0.0)
                dof = self._freejoint_dof[object_id]
                self.data.qvel[dof : dof + 6] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            self._attach_offsets.clear()