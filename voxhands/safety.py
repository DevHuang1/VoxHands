from __future__ import annotations

from .models import Plan
from .planner import TARGETS


def validate_plan(plan: Plan) -> list[str]:
    issues: list[str] = []
    used_targets: set[str] = set()
    used_arms: set[str] = set()

    if not plan.actions:
        issues.append("No recognized grippable tabletop object was found in the command.")
        text = plan.raw_text.lower()
        if "red" in text and any(word in text for word in ("pick", "grab", "lift", "move", "place")):
            issues.append("The red item is the no-go safety barrier and cannot be picked up.")

    for action in plan.actions:
        if action.target_id not in TARGETS:
            issues.append(f"Unknown target: {action.target_id}.")
        if action.target_id in used_targets:
            issues.append(f"Two actions target the same place: {action.target_label}.")
        used_targets.add(action.target_id)
        if action.arm not in {"left", "right"}:
            issues.append(f"Invalid arm assignment for {action.object_label}.")
        if action.arm in used_arms and action.target_id != "center_place":
            # Repeated use of an arm is allowed only when the simulator can queue it.
            action.note = "Queued on same arm"
        used_arms.add(action.arm)

    if "avoid_red_zone" not in plan.constraints:
        plan.constraints.append("avoid_red_zone")
    return issues
