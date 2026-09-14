from __future__ import annotations

from .models import Plan
from .planner import TARGETS, OBJECTS


def validate_plan(plan: Plan) -> list[str]:
    issues: list[str] = list(plan.safety_issues)
    if plan.intent in {"home", "calibration"}:
        return issues
    used_targets: set[str] = set()
    used_arms: set[str] = set()

    if not plan.actions and not plan.recognized_objects:
        issues.append("No recognized grippable tabletop object was found in the command.")
        text = plan.raw_text.lower()
        if "red" in text and any(word in text for word in ("pick", "grab", "lift", "move", "place")):
            issues.append("The red item is the no-go safety barrier and cannot be picked up.")

    for action in plan.actions:
        if action.target_id == "plate_surface" and action.object_id != "cup":
            issues.append("Only the cup can be placed onto the plate in this workcell.")
        if action.target_id == "cup_interior" and action.object_id == "blue_plate":
            issues.append("The plate is too large to fit inside the cup.")
        if action.target_id == "cup_interior" and action.object_id != "spoon":
            issues.append("Only the spoon fits the supported cup insertion profile.")
        if action.object_id not in OBJECTS:
            issues.append(f"Unknown object: {action.object_id}.")
        if action.target_id not in TARGETS:
            issues.append(f"Unknown target: {action.target_id}.")
        if TARGETS.get(action.target_id, {}).get("container") == action.object_id:
            issues.append(f"A {action.object_label.lower()} cannot be placed inside itself.")
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
