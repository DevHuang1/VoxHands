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
    "cup_interior": {"label": "Inside cup", "arm": "right", "container": "cup"},
    "plate_surface": {"label": "On plate", "arm": "left", "container": "blue_plate"},
}


def build_plan(raw_text: str) -> Plan:
    text = " ".join(raw_text.lower().strip().split())
    issues = []
    actions = []
    if re.fullmatch(r"(?:please )?home (?:both |the )?arms[.!]?", text):
        return Plan(new_id("plan"), raw_text, "home", [], [])
    if re.fullmatch(r"(?:show )?calibration(?: status)?[.!]?", text):
        return Plan(new_id("plan"), raw_text, "calibration", [], [])
    text = re.sub(r"\b(?:do not collide|don't collide|no collisions?)\b", "safely", text)
    # Never execute a partial interpretation of negated or unknown instructions.
    if re.search(r"\b(?:not|never|don't|except|instead)\b", text):
        issues.append("Please use positive placement instructions; negation is ambiguous.")
    aliases = {alias: oid for oid, obj in OBJECTS.items() for alias in obj["aliases"]}
    pattern = r"\b(?:" + "|".join(sorted(aliases, key=len, reverse=True)) + r")\b"
    matches = list(re.finditer(pattern, text))
    source_matches = []
    for match in matches:
        prefix = text[max(0, match.start() - 16):match.start()]
        if re.search(r"\b(?:into|inside|in|onto|on)\s+(?:the\s+)?$", prefix):
            continue
        source_matches.append(match)
    matches = source_matches
    table_request = bool(re.search(r"\b(?:set|arrange|prepare) (?:the )?table\b|\btable for", text))
    destinations = {}
    for index, match in enumerate(matches):
        oid = aliases[match.group()]
        end = matches[index+1].start() if index+1 < len(matches) else len(text)
        clause = text[match.end():end]
        found = set(re.findall(r"\b(left|right|center|centre|middle)\b", clause))
        found = {"center" if side in {"centre", "middle"} else side for side in found}
        for relation in re.finditer(r"\b(into|inside|in|onto|on)\s+(?:the\s+)?(" + pattern + r")", clause):
            destination = aliases[relation.group(2)]
            target_relation = "cup_interior" if destination == "cup" and relation.group(1) in {"into", "inside", "in"} else "plate_surface" if destination == "blue_plate" and relation.group(1) in {"on", "onto"} else None
            if target_relation:
                found.add(target_relation)
            else:
                issues.append(f"Unsupported placement: {OBJECTS[oid]['label']} {relation.group(1)} {OBJECTS[destination]['label']}.")
        if len(found) > 1:
            issues.append(f"Specify one destination for {OBJECTS[oid]['label']}.")
        target = next(iter(found)) if len(found) == 1 and next(iter(found)) in TARGETS else next(iter(found)) + "_place" if len(found) == 1 else None
        if oid in destinations and destinations[oid] != target:
            issues.append(f"Conflicting destinations for {OBJECTS[oid]['label']}.")
        destinations[oid] = target
    if not matches and table_request:
        destinations = {"blue_plate": "left_place", "cup": "right_place"}
    # Residual nouns after placement verbs identify unsupported objects instead of
    # silently executing the recognized subset of a compound instruction.
    residual = re.sub(pattern, " object ", text)
    allowed = set("please set arrange prepare the table for two put place move pick up grab lift a an object on to in into inside at left right center centre middle target place and then both avoid red zone area no collision do collide safely with arms arm".split())
    unknown = sorted(set(re.findall(r"[a-z]+", residual)) - allowed)
    unknown = [word for word in unknown if word != "onto"]
    if unknown:
        issues.append("Unrecognized wording: " + ", ".join(unknown) + ". Use 'Place the cup on the right'.")
    if re.search(r"(?:pick|grab|lift|move|place|put).*\bred (?:one|item|barrier|object)", text):
        issues.append("The red item is the no-go safety barrier and cannot be picked up.")
    missing = [oid for oid, target in destinations.items() if target is None]
    suggestions = []
    if len(missing) == 1 and not issues and not table_request:
        oid = missing[0]
        for side in ("left", "center", "right"):
            if side + "_place" in destinations.values():
                continue
            clauses = [f"Place the {OBJECTS[obj]['label']} on the {(target or side + '_place').removesuffix('_place')}" for obj, target in destinations.items()]
            suggestions.append({"label": f"{OBJECTS[oid]['label']} to {side}", "text": ". ".join(clauses) + "."})
    for index, (oid, target) in enumerate(destinations.items()):
        if target is None:
            if table_request:
                target = ["left_place", "right_place", "center_place"][min(index, 2)]
            else:
                issues.append(f"Where should {OBJECTS[oid]['label']} go: left, right, or center?")
                continue
        arm = TARGETS[target]["arm"]
        if target == "center_place" and oid in {"cup", "spoon"}:
            arm = "right"
        actions.append(Action(new_id("act"), arm, oid, OBJECTS[oid]["label"], target, TARGETS[target]["label"], duration_ms=5000))
    return Plan(new_id("plan"), raw_text, "table_setting" if table_request else "object_placement", actions,
                ["avoid_red_zone", "no_arm_collision"], safety_issues=issues, recognized_objects=[OBJECTS[oid]["label"] for oid in destinations], suggestions=suggestions)
