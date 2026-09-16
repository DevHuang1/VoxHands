from __future__ import annotations

from typing import Any

from voxhands.models import Action, Plan, new_id

from .base import Policy

CANONICAL_DINNER_STEPS: list[dict[str, Any]] = [
    {"action": "open_drawer", "arm": "right", "object_id": "drawer", "object_label": "Drawer", "target_id": "drawer_interior", "target_label": "Drawer interior", "duration_ms": 2000},
    {"action": "grasp_plate", "arm": "left", "object_id": "blue_plate", "object_label": "Blue plate", "target_id": "left_place", "target_label": "Left place", "duration_ms": 1400},
    {"action": "place_plate", "arm": "left", "object_id": "blue_plate", "object_label": "Blue plate", "target_id": "left_place", "target_label": "Left place", "duration_ms": 2000},
    {"action": "grasp_cup", "arm": "right", "object_id": "cup", "object_label": "Cup", "target_id": "right_place", "target_label": "Right place", "duration_ms": 1400},
    {"action": "place_cup", "arm": "right", "object_id": "cup", "object_label": "Cup", "target_id": "right_place", "target_label": "Right place", "duration_ms": 2000},
    {"action": "pour", "arm": "right", "object_id": "cup", "object_label": "Cup", "target_id": "blue_plate", "target_label": "Blue plate", "duration_ms": 3000},
    {"action": "handoff", "arm": "left", "object_id": "cup", "object_label": "Cup", "target_id": "right_place", "target_label": "Right place", "duration_ms": 1400},
    {"action": "silverware_into_drawer", "arm": "right", "object_id": "spoon", "object_label": "Spoon", "target_id": "drawer_interior", "target_label": "Drawer interior", "duration_ms": 2000},
    {"action": "check_final_state", "arm": "left", "object_id": "blue_plate", "object_label": "Blue plate", "target_id": "left_place", "target_label": "Left place", "duration_ms": 500},
]


class CanonicalDinnerPolicy(Policy):
    _step_index: int = 0

    def __init__(self) -> None:
        self._step_index = 0

    def __call__(self, state: dict[str, Any]) -> Action | Plan:
        step = state.get("step")
        if isinstance(step, int) and 0 <= step < len(CANONICAL_DINNER_STEPS):
            self._step_index = step
        elif self._step_index >= len(CANONICAL_DINNER_STEPS):
            self._step_index = 0

        if self._step_index >= len(CANONICAL_DINNER_STEPS):
            self._step_index = 0

        s = CANONICAL_DINNER_STEPS[self._step_index]
        action = Action(
            id=new_id("act"),
            arm=s["arm"],
            object_id=s["object_id"],
            object_label=s["object_label"],
            target_id=s["target_id"],
            target_label=s["target_label"],
            duration_ms=s["duration_ms"],
        )
        self._step_index = (self._step_index + 1) % len(CANONICAL_DINNER_STEPS)
        return action

    def train(self, demos: list[dict[str, Any]]) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def params(self) -> list[Any]:
        return []
