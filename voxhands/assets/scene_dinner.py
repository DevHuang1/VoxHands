"""Dinner-table workcell scene generator for the Physical AI challenge.

Builds a MuJoCo XML describing two simulated SO-101 arms around a dining
table with a sliding drawer, five freejoint objects (plate, cup, spoon,
fork, bottle) and overhead / operator cameras. Every physical and visual
parameter referenced by the robustness harness is injected here, so a
per-seed ``DinnerConfig`` produces a deterministic, reproducible scene.

The scene keeps the browser workcell coordinate convention: the table top is
at z = 0 and x/y are metres in the workcell plane (arms mounted at low y,
diners/homes at higher y).

This module is intentionally declarative: no MuJoCo imports, so it stays
importable even where mujoco is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# object registry shared by generator, env, vision and the planner helpers
# --------------------------------------------------------------------------- #
OBJECT_REGISTRY: dict[str, dict[str, Any]] = {
    "plate": {
        "label": "Plate",
        "color": "#3377ff",
        "geom": "cylinder",
        "size": (0.06, 0.013),
        "mass": 0.08,
        "rest_z": 0.0238,
        "home": (0.24, 0.64),
    },
    "cup": {
        "label": "Cup",
        "color": "#e8c96a",
        "geom": "cylinder",
        "size": (0.038, 0.055),
        "mass": 0.06,
        "rest_z": 0.0643,
        "home": (0.76, 0.64),
    },
    "spoon": {
        "label": "Spoon",
        "color": "#c7a252",
        "geom": "box",
        "size": (0.015, 0.075, 0.012),
        "mass": 0.03,
        "rest_z": 0.0218,
        "home": (0.40, 0.74),
    },
    "fork": {
        "label": "Fork",
        "color": "#4f6f9f",
        "geom": "box",
        "size": (0.015, 0.075, 0.012),
        "mass": 0.03,
        "rest_z": 0.0218,
        "home": (0.60, 0.74),
    },
    "bottle": {
        "label": "Bottle",
        "color": "#3e9e4f",
        "geom": "cylinder",
        "size": (0.026, 0.10),
        "mass": 0.10,
        "rest_z": 0.1087,
        "home": (0.50, 0.52),
    },
}

# arm names and their standing parked poses (grip site targets)
ARM_NAMES = ("left", "right")

# drawer geometry (metres).  The drawer is an open tray on the far side of
# the table; its body slides along the y axis.  +y opens it (away from the
# arms) so the inner contents are reachable once open.
DRAWER = {
    "x": 0.50,
    "start_y": 0.88,  # body centre y when fully closed
    "size_x": 0.22,
    "size_y": 0.11,
    "height": 0.035,
    "max_open": 0.12,  # travel along +y
}

# camera names consistent with the existing server contract
CAMERA_NAMES = ("overhead", "operator")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class DinnerConfig:
    """Every knob the robustness harness may vary for a seed."""

    seed: int = 0
    object_masses: dict[str, float] = field(
        default_factory=lambda: {oid: float(info["mass"]) for oid, info in OBJECT_REGISTRY.items()}
    )
    object_frictions: dict[str, float] = field(
        default_factory=lambda: {oid: 0.9 for oid in OBJECT_REGISTRY}
    )
    object_scales: dict[str, float] = field(
        default_factory=lambda: {oid: 1.0 for oid in OBJECT_REGISTRY}
    )
    object_yaws: dict[str, float] = field(
        default_factory=lambda: {oid: 0.0 for oid in OBJECT_REGISTRY}
    )
    # per-object (x, y) metre offsets from the registry home
    object_offset: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {oid: (0.0, 0.0) for oid in OBJECT_REGISTRY}
    )
    drawer_max_open: float = DRAWER["max_open"]
    drawer_friction: float = 0.5
    light_key_diffuse: tuple[float, float, float] = (0.55, 0.55, 0.55)
    light_key_direction: tuple[float, float, float, float] = (0.0, -0.45, -0.9)
    light_fill_diffuse: tuple[float, float, float] = (0.32, 0.32, 0.32)
    light_fill_direction: tuple[float, float, float, float] = (-1.0, 0.5, -0.6)
    ambient_light: tuple[float, float, float] = (0.18, 0.18, 0.20)
    background_rgba: tuple[float, float, float, float] = (0.10, 0.11, 0.13, 1.0)
    table_rgba: tuple[float, float, float, float] = (0.68, 0.67, 0.66, 1.0)  # neutral linen cloth
    camera_overhead_position: tuple[float, float, float] = (0.5, 0.5, 1.35)
    camera_overhead_xyaxes: tuple[float, float, float, float, float, float] = (1, 0, 0, 0, 1, 0)
    camera_operator_position: tuple[float, float, float] = (1.05, 0.45, 0.95)
    camera_operator_xyaxes: tuple[float, float, float, float, float, float] = (
        0.5, 0.866, 0, -0.5, 0.29, 0.82
    )
    devices_preferred: tuple[str, ...] = ("NPU", "GPU", "CPU")  # not a scene knob

    def object_geom_size(self, oid: str) -> tuple[float, ...]:
        base = OBJECT_REGISTRY[oid]["size"]
        scale = float(self.object_scales.get(oid, 1.0))
        if OBJECT_REGISTRY[oid]["geom"] == "box":
            return tuple(round(value * scale, 4) for value in base)
        # cylinder: radius and height both scale
        return (round(base[0] * scale, 4), round(max(0.004, base[0] * scale * 0.22) if False else base[1] * scale, 4))

    def object_pose(self, oid: str) -> tuple[float, float, float, tuple[float, ...]]:
        hx, hy = OBJECT_REGISTRY[oid]["home"]
        ox, oy = self.object_offset.get(oid, (0.0, 0.0))
        return (hx + ox, hy + oy, OBJECT_REGISTRY[oid]["rest_z"], _yaw_quat(self.object_yaws.get(oid, 0.0)))


def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    import math

    half = math.radians(yaw_deg) / 2.0
    return round(math.cos(half), 6), round(math.sin(half), 6), 0.0, 0.0


# --------------------------------------------------------------------------- #
# XML builders
# --------------------------------------------------------------------------- #
def _arm_xml(arm: str, side: str, base_x: float, base_y: float) -> str:
    if side not in {"left", "right"}:
        raise ValueError(side)
    left_sign = -1.0 if side == "left" else 1.0
    return f"""
<body name="{arm.upper()}_base" pos="{base_x} {base_y} 0.0">
  <geom name="{arm}_base_geom" type="cylinder" size="0.06 0.035" pos="0 0 0.035" rgba="0.28 0.28 0.30 1" mass="1.2" contype="0" conaffinity="0"/>
  <joint name="{arm}_base_yaw" type="hinge" axis="0 0 1" pos="0 0 0.02" limited="true" range="-175 175"/>
  <body name="{arm.upper()}_shoulder" pos="0 0 0.0">
    <geom name="{arm}_shoulder_geom" type="sphere" size="0.055" pos="0 0 0.14" rgba="0.32 0.32 0.34 1" mass="0.8" contype="0" conaffinity="0"/>
    <joint name="{arm}_shoulder" type="hinge" axis="0 1 0" pos="0 0 0.14" limited="true" range="-175 175"/>
    <body name="{arm.upper()}_upper" pos="0 0 0.14">
      <geom name="{arm}_upper_geom" type="capsule" size="0.045" fromto="0 0 0 0.44 0 0" rgba="0.40 0.40 0.42 1" mass="1.4" contype="0" conaffinity="0"/>
      <joint name="{arm}_elbow" type="hinge" axis="0 1 0" pos="0.44 0 0" limited="true" range="-175 175"/>
      <body name="{arm.upper()}_fore" pos="0.44 0 0">
        <geom name="{arm}_fore_geom" type="capsule" size="0.038" fromto="0 0 0 0.40 0 0" rgba="0.50 0.50 0.52 1" mass="1.0" contype="0" conaffinity="0"/>
        <joint name="{arm}_wrist_pitch" type="hinge" axis="0 1 0" pos="0.40 0 0" limited="true" range="-175 175"/>
        <body name="{arm.upper()}_hand" pos="0.40 0 0">
          <geom name="{arm}_hand_geom" type="capsule" size="0.028" fromto="0 0 0 0.14 0 0" rgba="0.56 0.56 0.58 1" mass="0.3" contype="0" conaffinity="0"/>
          <site name="{arm}_grip_site" pos="0.14 0 0" type="sphere" size="0.008" rgba="0.12 0.12 0.14 1"/>
          <joint name="{arm}_wrist_roll" type="hinge" axis="1 0 0" pos="0.14 0 0" limited="true" range="-130 130"/>
          <body name="{arm.upper()}_grip" pos="0.14 0 0">
            <joint name="{arm}_fing_left" type="slide" axis="0 0 1" pos="0 0 0" limited="true" range="-0.03 0.02"/>
            <body name="{arm}_finger_left" pos="0 0 0">
              <geom name="{arm}_finger_left_geom" type="box" size="0.012 0.03 0.006" pos="0 0.015 {left_sign * -0.028}" rgba="0.85 0.85 0.87 1" mass="0.02" contype="0" conaffinity="0"/>
            </body>
            <joint name="{arm}_fing_right" type="slide" axis="0 0 1" pos="0 0 0" limited="true" range="-0.02 0.03"/>
            <body name="{arm}_finger_right" pos="0 0 0">
              <geom name="{arm}_finger_right_geom" type="box" size="0.012 0.03 0.006" pos="0 0.015 {left_sign * 0.028}" rgba="0.85 0.85 0.87 1" mass="0.02" contype="0" conaffinity="0"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </body>
</body>
"""


def _object_xml(config: DinnerConfig, oid: str) -> str:
    registry = OBJECT_REGISTRY[oid]
    mass = float(config.object_masses.get(oid, registry["mass"]))
    friction = float(config.object_frictions.get(oid, 0.9))
    size = config.object_geom_size(oid)
    x, y, z, quat = config.object_pose(oid)
    scale = float(config.object_scales.get(oid, 1.0))
    rgba = list(_hex_to_rgba(registry["color"], 255 * max(0.8, 1.0 if scale > 1 else 0.95)))
    geom = registry["geom"]
    if geom == "cylinder":
        geom_tag = f'<geom name="{oid}_geom" type="cylinder" size="{size[0]} {size[1]}" rgba="{_rgba(config, oid, scale)}" mass="{mass}" friction="{friction}"/>'
    else:
        geom_tag = f'<geom name="{oid}_geom" type="box" size="{size[0]} {size[1]} {size[2]}" rgba="{_rgba(config, oid, scale)}" mass="{mass}" friction="{friction}"/>'
    # visual helper: nothing extra beyond the single geom; body keeps the rest z
    return f"""<body name="{oid}" pos="{x} {y} {z}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
  <freejoint/>
  {geom_tag}
</body>
"""


def _builtin_rgba(color: str, alpha: float = 1.0, brightness: float = 1.0) -> str:
    r, g, b = _hex_to_rgb(color)
    return f"{r:.2f} {g:.2f} {b:.2f} {alpha:.2f}"


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _hex_to_rgba(color: str, alpha: int) -> tuple[float, float, float, float]:
    r, g, b = _hex_to_rgb(color)
    return r, g, b, float(alpha) / 255.0


def _rgba(config: DinnerConfig, oid: str, scale: float) -> str:
    base = _hex_to_rgb(OBJECT_REGISTRY[oid]["color"])
    brightness = float(scale)
    strength = max(0.0, min(1.0, brightness * 1.15))
    r, g, b = base[0] * strength, base[1] * strength, base[2] * strength
    return f"{r:.2f} {g:.2f} {b:.2f} 1.0"


def _drawer_xml(config: DinnerConfig) -> str:
    drawer = DRAWER
    start_y = drawer["start_y"]
    size_x = drawer["size_x"]
    size_y = drawer["size_y"]
    height = drawer["height"]
    handle_site = f'<site name="drawer_handle" pos="0 {-size_y / 2} 0.02" type="sphere" size="0.014" rgba="0.35 0.36 0.38 1"/>'
    friction = float(config.drawer_friction)
    return f"""
<body name="drawer_frame" pos="{drawer['x']} {start_y} 0.0">
  <geom name="drawer_frame" type="box" size="{size_x} {size_y} 0.008" pos="0 0 0.002" rgba="0.44 0.43 0.42 1" contype="0" conaffinity="0"/>
</body>
<body name="drawer" pos="{drawer['x']} {start_y} 0.0">
  <joint name="drawer_slide" type="slide" axis="0 1 0" pos="0 0 0" limited="true" range="0 {config.drawer_max_open}" frictionloss="0.001"/>
  <geom name="drawer_body" type="box" size="{size_x} {size_y} {height}" pos="0 0 {height}" rgba="0.55 0.54 0.53 1" contype="0" conaffinity="0" mass="0.4" friction="{friction}"/>
  {handle_site}
</body>
"""


def _light_xml(name: str, pos: tuple[float, float, float], direction: tuple[float, float, float], diffuse: tuple[float, float, float]) -> str:
    return (
        f'<light name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" '
        f'dir="{direction[0]} {direction[1]} {direction[2]}" directional="true" '
        f'diffuse="{diffuse[0]:.3f} {diffuse[1]:.3f} {diffuse[2]:.3f}" specular="0.0 0.0 0.0" castshadow="false"/>'
    )


def build_scene_xml(config: DinnerConfig | None = None) -> str:
    config = config or DinnerConfig()
    key_dir = config.light_key_direction.to_tuple() if hasattr(config.light_key_direction, "to_tuple") else config.light_key_direction
    fill_dir = config.light_fill_direction.to_tuple() if hasattr(config.light_fill_direction, "to_tuple") else config.light_fill_direction
    background = config.background_rgba
    table = config.table_rgba
    ambient = config.ambient_light
    overhead_pos = config.camera_overhead_position
    overhead_axes = config.camera_overhead_xyaxes
    operator_pos = config.camera_operator_position
    operator_axes = config.camera_operator_xyaxes

    objects = "".join(_object_xml(config, oid) for oid in OBJECT_REGISTRY)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="dinner_workcell">
  <compiler angle="degree" coordinate="local" eulerseq="xyz" discardvisual="false"/>
  <option timestep="0.001" gravity="0 0 -9.81" iterations="40" tolerance="1e-8"/>
  <size njmax="2000" nconmax="1200" nstack="7000000"/>
  <visual>
    <global offwidth="1024" offheight="1024"/>
    <headlight ambient="{ambient[0]:.3f} {ambient[1]:.3f} {ambient[2]:.3f}" diffuse="0 0 0"/>
  </visual>

  <default>
    <geom contype="1" conaffinity="1" friction="1 0.005 0.001"/>
    <joint damping="0.02" armature="0.1"/>
    <material specular="0.0" shininess="0.0"/>
  </default>

  <worldbody>
    {_light_xml("key", (0.5, 1.0, 2.2), key_dir, config.light_key_diffuse)}
    {_light_xml("fill", (1.4, 0.5, 1.4), fill_dir, config.light_fill_diffuse)}
    <geom name="ground" type="plane" pos="0 0 -0.05" size="3 3 0.05" rgba="{background[0]:.2f} {background[1]:.2f} {background[2]:.2f} 1" contype="1" conaffinity="1"/>
    <geom name="table" type="box" pos="0.5 0.5 -0.03" size="0.5 0.52 0.03" rgba="{table[0]:.3f} {table[1]:.3f} {table[2]:.3f} 1" friction="0.9 0.005 0.001"/>

    {_arm_xml("left", "left", 0.24, 0.16)}
    {_arm_xml("right", "right", 0.76, 0.16)}

    {_drawer_xml(config)}

    {objects}

    <camera name="overhead" pos="{overhead_pos[0]} {overhead_pos[1]} {overhead_pos[2]}" xyaxes="{' '.join(str(v) for v in overhead_axes)}" fovy="55"/>
    <camera name="operator" pos="{operator_pos[0]} {operator_pos[1]} {operator_pos[2]}" xyaxes="{' '.join(str(v) for v in operator_axes)}" fovy="50"/>
  </worldbody>

  <actuator>
    <position name="left_base_yaw" joint="left_base_yaw" kp="40000" damping="600" ctrlrange="-175 175"/>
    <position name="left_shoulder" joint="left_shoulder" kp="40000" damping="600" ctrlrange="-175 175"/>
    <position name="left_elbow" joint="left_elbow" kp="40000" damping="600" ctrlrange="-175 175"/>
    <position name="left_wrist_pitch" joint="left_wrist_pitch" kp="40000" damping="600" ctrlrange="-175 175"/>
    <position name="left_wrist_roll" joint="left_wrist_roll" kp="16000" damping="240" ctrlrange="-130 130"/>
    <position name="left_fing_left" joint="left_fing_left" kp="200" damping="15"/>
    <position name="left_fing_right" joint="left_fing_right" kp="200" damping="15"/>
    <position name="right_base_yaw" joint="right_base_yaw" kp="40000" damping="600" ctrlrange="-175 175"/>
    <position name="right_shoulder" joint="right_shoulder" kp="40000" damping="600" ctrlrange="-175 175"/>
    <position name="right_elbow" joint="right_elbow" kp="40000" damping="600" ctrlrange="-175 175"/>
    <position name="right_wrist_pitch" joint="right_wrist_pitch" kp="40000" damping="600" ctrlrange="-175 175"/>
    <position name="right_wrist_roll" joint="right_wrist_roll" kp="16000" damping="240" ctrlrange="-130 130"/>
    <position name="right_fing_left" joint="right_fing_left" kp="200" damping="15"/>
    <position name="right_fing_right" joint="right_fing_right" kp="200" damping="15"/>
  </actuator>
</mujoco>
"""


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "scene_dinner.xml"
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(build_scene_xml())
    print(f"wrote {out}")