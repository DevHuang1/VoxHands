from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .assets.scene_dinner import ARM_NAMES, DinnerConfig
from .mujoco_workcell import DinnerTableWorkcell, PrimitiveResult, TARGET_POSITIONS


@dataclass
class PlanStep:
    arm: str
    primitive: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    parallel_ok: bool = False


LEFT_CRESCENT = {"x": (0.22, 0.58), "y": (0.30, 1.12)}
RIGHT_CRESCENT = {"x": (0.42, 0.78), "y": (0.30, 1.12)}
ARM_CRESCENT = {"left": LEFT_CRESCENT, "right": RIGHT_CRESCENT}


class BimanualController:
    def __init__(
        self,
        config: DinnerConfig | None = None,
        oracle: bool = True,
    ) -> None:
        self.workcell = DinnerTableWorkcell(
            config=config or DinnerConfig(), oracle=oracle
        )

    # ------------------------------------------------------------------ #
    # plan construction
    # ------------------------------------------------------------------ #
    def plan_sequence(self) -> list[PlanStep]:
        steps: list[PlanStep] = [
            PlanStep("left", "open_drawer"),
            PlanStep("left", "grasp", ("plate",)),
            PlanStep("left", "place", ("plate", "left_place")),
            PlanStep("right", "grasp", ("cup",)),
            PlanStep("right", "place", ("cup", "right_place")),
            PlanStep("right", "grasp", ("cup",)),
            PlanStep("left", "grasp", ("bottle",)),
            PlanStep(
                "left",
                "pour",
                ("left", "right", "bottle", "cup"),
            ),
            PlanStep("right", "place", ("cup", "right_place")),
            PlanStep("left", "handoff", ("left", "right", "bottle")),
            PlanStep("left", "grasp", ("spoon",)),
            PlanStep("left", "place", ("spoon", (0.50, 1.00))),
            PlanStep("left", "grasp", ("fork",)),
            PlanStep("left", "place", ("fork", (0.50, 0.98))),
            PlanStep("left", "check_final_state"),
        ]
        self._annotate_parallelism(steps)
        return steps

    def _annotate_parallelism(self, steps: list[PlanStep]) -> None:
        for step in steps:
            step.parallel_ok = False
        for index in range(1, len(steps)):
            lhs, rhs = steps[index - 1], steps[index]
            if (
                self._zone_type(lhs) in ("left", "right")
                and self._zone_type(rhs) in ("left", "right")
                and self._zone_type(lhs) != self._zone_type(rhs)
            ):
                lhs.parallel_ok = True
                rhs.parallel_ok = True

    @staticmethod
    def _zone_type(step: PlanStep) -> str | None:
        if step.primitive in ("open_drawer", "close_drawer", "check_final_state"):
            return None
        if step.primitive == "pour":
            return "center"
        if step.primitive == "handoff":
            return "center"
        if step.args:
            oid_or_target = step.args[0]
            if step.primitive == "place":
                target = step.args[1]
                if isinstance(target, str):
                    if target in ("left_place", "center_place"):
                        return "left"
                    if target == "right_place":
                        return "right"
                return "left" if step.arm == "left" else "right"
            if step.primitive == "grasp":
                if oid_or_target == "cup":
                    return "right"
                if oid_or_target == "bottle":
                    return "center"
                if oid_or_target == "fork":
                    return "right"
                return "left"
        return "left" if step.arm == "left" else "right"

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    def run(self, plan: list[PlanStep]) -> list[PrimitiveResult]:
        return [self._execute_step(step) for step in plan]

    def _execute_step(self, step: PlanStep) -> PrimitiveResult:
        wc = self.workcell
        primitive, args, kwargs = step.primitive, step.args, step.kwargs
        if primitive == "open_drawer":
            return wc.open_drawer(arm=step.arm)
        if primitive == "close_drawer":
            return wc.close_drawer(arm=step.arm)
        if primitive == "grasp":
            return wc.grasp(step.arm, args[0])
        if primitive == "place":
            return wc.place(step.arm, args[0], args[1])
        if primitive == "pour":
            bottle_arm, cup_arm, bottle_oid, cup_oid = args
            return wc.pour(
                bottle_arm=bottle_arm,
                cup_arm=cup_arm,
                bottle_oid=bottle_oid,
                cup_oid=cup_oid,
                tilt_deg=kwargs.get("tilt_deg", 42.0),
            )
        if primitive == "handoff":
            from_arm, to_arm, oid = args
            return wc.handoff(from_arm=from_arm, to_arm=to_arm, oid=oid)
        if primitive == "check_final_state":
            report = wc.check_final_state(self.dinner_goals())
            return PrimitiveResult(
                "check_final_state",
                report["success"],
                message=("; ".join(report["failures"]) if report["failures"] else "all goals satisfied"),
                detail=report,
            )
        return PrimitiveResult(primitive, False, message=f"unknown primitive {primitive}")

    @staticmethod
    def dinner_goals() -> dict[str, Any]:
        return {
            "place": {"plate": "left_place", "cup": "right_place"},
            "drawer_content": ["spoon", "fork"],
            "drawer_open_min": 0.5,
            "pour_required": True,
            "handoff_required": True,
        }

    # ------------------------------------------------------------------ #
    # focused sub-plans
    # ------------------------------------------------------------------ #
    def handoff_plan(self, oid: str = "bottle") -> list[PlanStep]:
        return [
            PlanStep("right", "grasp", (oid,)),
            PlanStep("left", "handoff", ("right", "left", oid)),
        ]

    def pour_plan(
        self,
        bottle_arm: str = "left",
        cup_arm: str = "right",
    ) -> list[PlanStep]:
        return [
            PlanStep(cup_arm, "grasp", ("cup",)),
            PlanStep(bottle_arm, "grasp", ("bottle",)),
            PlanStep(
                bottle_arm,
                "pour",
                (bottle_arm, cup_arm, "bottle", "cup"),
            ),
        ]

    def drawer_plan(self, arm: str = "left") -> list[PlanStep]:
        return [
            PlanStep(arm, "open_drawer"),
            PlanStep(arm, "close_drawer"),
        ]

    # ------------------------------------------------------------------ #
    # reachability / goal snapping
    # ------------------------------------------------------------------ #
    def refresh_goals(self) -> dict[str, Any]:
        wc = self.workcell
        refreshed: dict[str, Any] = {}
        for name, (tx, ty) in TARGET_POSITIONS.items():
            best_arm: str | None = None
            best_value = float("inf")
            for arm in ARM_NAMES:
                position = (tx, ty, 0.15)
                goal = wc._solve_ik(arm, position)
                jids = wc._arm_joint_ids[arm]
                in_range = all(
                    float(wc.model.jnt_range[jid][0])
                    <= value
                    <= float(wc.model.jnt_range[jid][1])
                    for jid, value in zip(jids, goal)
                )
                if not in_range:
                    continue
                crescent = ARM_CRESCENT[arm]
                dx = max(crescent["x"][0] - tx, 0.0, tx - crescent["x"][1])
                dy = max(crescent["y"][0] - ty, 0.0, ty - crescent["y"][1])
                distance = math.hypot(dx, dy)
                if distance < best_value:
                    best_value = distance
                    best_arm = arm
            if best_arm is None:
                snapped = (tx, ty)
            else:
                crescent = ARM_CRESCENT[best_arm]
                snapped = (
                    max(crescent["x"][0], min(crescent["x"][1], tx)),
                    max(crescent["y"][0], min(crescent["y"][1], ty)),
                )
            refreshed[name] = {
                "target": (tx, ty),
                "snapped": (snapped[0], snapped[1]),
                "arm": best_arm,
                "snapped_away": snapped != (tx, ty),
            }
        return refreshed