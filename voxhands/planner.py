from __future__ import annotations

import re

from .models import Action, Plan, new_id


OBJECTS = {
    "blue_plate": {"label": "Blue plate", "aliases": ("blue plate", "plate")},
    "cup": {"label": "Cup", "aliases": ("cup", "mug")},
    "fork": {"label": "Fork", "aliases": ("fork",)},
    "spoon": {"label": "Spoon", "aliases": ("spoon",)},
}

TARGETS = {
    "left_place": {"label": "Left place", "arm": "left"},
    "right_place": {"label": "Right place", "arm": "right"},
    "center_place": {"label": "Center place", "arm": "left"},
}


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _object_mentions(text: str) -> list[str]:
    found: list[str] = []
    for object_id, definition in OBJECTS.items():
        if any(alias in text for alias in definition["aliases"]):
            found.append(object_id)
    return found


def _target_for(object_id: str, text: str, index: int) -> str:
    if object_id == "blue_plate" and _has_word(text, "left"):
        return "left_place"
    if object_id == "cup" and _has_word(text, "right"):
        return "right_place"
    if object_id in {"fork", "spoon"} and _has_word(text, "right"):
        return "right_place"
    if index == 0:
        return "left_place"
    if index == 1:
        return "right_place"
    return "center_place"


def build_plan(raw_text: str) -> Plan:
    text = " ".join(raw_text.lower().strip().split())
    if not text:
        text = "set the table for two"

    mentions = _object_mentions(text)
    # Only use the demo's default pair for an explicit table-setting request.
    # A command such as "place the red one" must not silently become a table
    # setting plan merely because it contains the verb "place".
    default_table_request = any(
        phrase in text
        for phrase in ("set the table", "set table", "arrange the table", "prepare the table", "table for")
    )
    if not mentions and default_table_request:
        mentions = ["blue_plate", "cup"]

    # Preserve a stable, demo-friendly order even if the sentence names the cup first.
    preferred_order = ["blue_plate", "cup", "fork", "spoon"]
    mentions = [object_id for object_id in preferred_order if object_id in mentions]
    constraints: list[str] = []
    if "red zone" in text or "red area" in text or "avoid red" in text:
        constraints.append("avoid_red_zone")
    if "do not collide" in text or "don't collide" in text or "no collision" in text:
        constraints.append("no_arm_collision")
    if not constraints:
        constraints.append("no_arm_collision")

    actions: list[Action] = []
    for index, object_id in enumerate(mentions):
        target_id = _target_for(object_id, text, index)
        target = TARGETS[target_id]
        action_arm = target["arm"]
        if target_id == "center_place" and object_id in {"cup", "spoon"}:
            action_arm = "right"
        actions.append(
            Action(
                id=new_id("act"),
                arm=action_arm,
                object_id=object_id,
                object_label=OBJECTS[object_id]["label"],
                target_id=target_id,
                target_label=target["label"],
                note="Parallel-safe move" if index < 2 else "Queued after primary pair",
            )
        )

    if "table" in text or "arrange" in text or "place" in text:
        intent = "table_setting"
    else:
        intent = "object_placement"

    return Plan(
        id=new_id("plan"),
        raw_text=raw_text.strip() or "Set the table for two",
        intent=intent,
        actions=actions,
        constraints=constraints,
    )
