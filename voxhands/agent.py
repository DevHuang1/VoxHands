from __future__ import annotations

import json
import re
import time
from typing import Any

from .groq import GroqClient
from .models import Action, Plan, new_id
from .planner import OBJECTS, TARGETS
from .simulation import TableSettingSimulation
from .styles import normalize_gesture, normalize_style


MAX_TOOL_STEPS = 6
UNSUPPORTED_TARGET_PATTERN = re.compile(r"\b(?:to|onto|on|into|in|at)\s+(?:the\s+)?(?:ground|floor|off\s+the\s+table|outside|trash|bin)\b", re.IGNORECASE)

AGENT_SYSTEM_PROMPT = """You are the safe tool controller for the VoxHands dual-arm tabletop simulator.

You control the workcell only through the supplied tools. The Python simulator is authoritative for object state, trajectories, safety, and completion. Browser Rapier telemetry is authoritative for contact and settling observations; never claim a collision-free grasp unless the tool result says so.

Rules:
- Call observe_workcell when you need current state.
- For moving a tabletop object, use execute_placement. Do not fake a grasp with manual arm and gripper tools.
- Use move_arm and set_gripper only for empty-arm demonstrations or calibration-style motion.
- Never move, target, or pick the red safety barrier.
- Never invent object IDs or target IDs. If the requested target is unavailable, explain the available targets.
- Use stop_all_motion when the user asks to stop or when a tool result reports a collision.
- Do not reset unless the user explicitly asks to reset.
- Keep responses concise and truthful. A tool call is not completion; report the simulator status returned by the tools.
"""


WORKCELL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "observe_workcell",
            "description": "Read the authoritative workcell, arms, objects, targets, and browser physics telemetry.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_placement",
            "description": "Pick one known tabletop object and place it at a known safe target using the server trajectory and gripper physics contract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_id": {"type": "string", "enum": list(OBJECTS)},
                    "target_id": {"type": "string", "enum": list(TARGETS)},
                    "style": {"type": "string", "enum": ["standard", "gentle", "precise", "rapid", "playful", "wavy"]},
                    "speed": {"type": "number", "minimum": 0.3, "maximum": 3.0},
                    "gesture": {"type": "string", "enum": ["none", "wave", "bow", "dance", "point"]},
                    "pause_ms": {"type": "integer", "minimum": 0, "maximum": 15000},
                    "condition": {"type": "string"},
                },
                "required": ["object_id", "target_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_arm",
            "description": "Move an empty left or right arm to a bounded workcell pose. This does not pick up an object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arm": {"type": "string", "enum": ["left", "right"]},
                    "x": {"type": "number", "minimum": 0.02, "maximum": 0.98},
                    "y": {"type": "number", "minimum": 0.02, "maximum": 0.98},
                    "z": {"type": "number", "minimum": 0.08, "maximum": 0.8},
                    "angle": {"type": "number", "minimum": -90, "maximum": 90},
                    "duration_ms": {"type": "integer", "minimum": 100, "maximum": 15000},
                },
                "required": ["arm", "x", "y", "z"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_gripper",
            "description": "Open or close an empty arm gripper within a bounded aperture. This does not attach objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arm": {"type": "string", "enum": ["left", "right"]},
                    "action": {"type": "string", "enum": ["open", "close"]},
                    "aperture": {"type": "number", "minimum": 0.04, "maximum": 0.30},
                },
                "required": ["arm", "action"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "home_arms",
            "description": "Return both empty arms to their safe home poses and open their grippers.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_settle",
            "description": "Wait for the current simulator task to finish or reach a terminal state, up to 15 seconds.",
            "parameters": {"type": "object", "properties": {"timeout_ms": {"type": "integer", "minimum": 100, "maximum": 15000}}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_motion",
            "description": "Pause the active simulator task while holding the current arm poses.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_motion",
            "description": "Resume a paused simulator task.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_all_motion",
            "description": "Stop all active motion and require reset before another placement run.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_workcell",
            "description": "Reset the simulator, object poses, arms, metrics, and physics telemetry. Use only when explicitly requested.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def _compact_state(state: dict[str, Any]) -> dict[str, Any]:
    if "error" in state:
        return {"error": state["error"], "status": state.get("status", "error")}
    return {
        "status": state.get("status"),
        "plan": state.get("plan"),
        "objects": {
            object_id: {
                key: object_state.get(key)
                for key in ("label", "x", "y", "z", "holder", "settled", "target_locked", "container", "supported_by")
                if key in object_state
            }
            for object_id, object_state in (state.get("objects") or {}).items()
        },
        "targets": state.get("targets"),
        "arms": {
            arm: {
                key: arm_state.get(key)
                for key in ("status", "phase", "x", "y", "z", "angle", "gripper", "holding", "grasp_confirmed")
            }
            for arm, arm_state in (state.get("arms") or {}).items()
        },
        "physics": state.get("physics"),
    }


def _placement_plan(raw_text: str, args: dict[str, Any]) -> Plan | None:
    object_id = str(args.get("object_id", "")).strip()
    target_id = str(args.get("target_id", "")).strip()
    if object_id not in OBJECTS or target_id not in TARGETS:
        return None
    action = Action(
        new_id("act"),
        TARGETS[target_id]["arm"],
        object_id,
        OBJECTS[object_id]["label"],
        target_id,
        TARGETS[target_id]["label"],
        duration_ms=5000,
        style=normalize_style(args.get("style")),
        speed=max(0.3, min(3.0, float(args.get("speed", 1.0)))),
        gesture=normalize_gesture(args.get("gesture")),
        pause_ms=max(0, min(15_000, int(args.get("pause_ms", 0)))),
        condition=str(args.get("condition", "")).strip(),
    )
    plan = Plan(new_id("plan"), raw_text, "object_placement", [action], ["avoid_red_zone", "no_arm_collision"], recognized_objects=[OBJECTS[object_id]["label"]])
    plan.llm_provider = "groq"
    plan.mode = "command"
    return plan


def unsupported_target_issue(raw_text: str) -> str | None:
    match = UNSUPPORTED_TARGET_PATTERN.search(raw_text)
    if not match:
        return None
    target_text = match.group(0).split()[-1].rstrip(".,!?\"")
    return f"Unsupported destination '{target_text}'. Safe targets are left, right, center, inside the cup, or on the plate."


class WorkcellToolDispatcher:
    def __init__(self, simulation: TableSettingSimulation, raw_text: str, blocked_issue: str | None = None) -> None:
        self.simulation = simulation
        self.raw_text = raw_text
        self.blocked_issue = blocked_issue

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "observe_workcell":
            return _compact_state(self.simulation.snapshot())
        if self.blocked_issue and name in {"execute_placement", "move_arm", "set_gripper"}:
            return {"error": self.blocked_issue}
        if name == "execute_placement":
            try:
                plan = _placement_plan(self.raw_text, args)
            except (TypeError, ValueError):
                plan = None
            if plan is None:
                return {"error": "Unknown object or target. Use only the supplied object_id and target_id enums."}
            return _compact_state(self.simulation.submit_command(self.raw_text, plan=plan))
        if name == "move_arm":
            return _compact_state(self.simulation.move_arm(**args))
        if name == "set_gripper":
            return _compact_state(self.simulation.set_gripper(**args))
        if name == "home_arms":
            return _compact_state(self.simulation.home_arms())
        if name == "pause_motion":
            return _compact_state(self.simulation.pause())
        if name == "resume_motion":
            return _compact_state(self.simulation.resume())
        if name == "stop_all_motion":
            return _compact_state(self.simulation.stop())
        if name == "reset_workcell":
            self.simulation.reset()
            return _compact_state(self.simulation.snapshot())
        if name == "wait_for_settle":
            try:
                timeout_ms = max(100, min(15_000, int(args.get("timeout_ms", 10_000))))
            except (TypeError, ValueError):
                timeout_ms = 10_000
            deadline = time.monotonic() + timeout_ms / 1000
            while time.monotonic() < deadline:
                state = self.simulation.snapshot()
                if state["status"] not in {"running", "paused"}:
                    return _compact_state(state)
                time.sleep(0.05)
            return _compact_state(self.simulation.snapshot())
        return {"error": f"Unknown workcell tool: {name}."}


def run_agent(
    raw_text: str,
    simulation: TableSettingSimulation,
    client: GroqClient,
    style: object | None = None,
    gesture: object | None = None,
    max_steps: int = MAX_TOOL_STEPS,
) -> dict[str, Any]:
    """Run a bounded Groq tool loop against the authoritative simulator."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]
    if style or gesture:
        messages.append({"role": "system", "content": f"UI overrides: style={style or 'standard'}, gesture={gesture or 'none'}."})
    blocked_issue = unsupported_target_issue(raw_text)
    if blocked_issue:
        messages.append({"role": "system", "content": f"Safety gate: {blocked_issue} Do not reinterpret it as another target and do not execute motion."})
    dispatcher = WorkcellToolDispatcher(simulation, raw_text, blocked_issue=blocked_issue)
    mutating_tools = {"execute_placement", "move_arm", "set_gripper", "home_arms", "pause_motion", "resume_motion", "stop_all_motion", "reset_workcell"}
    changed_workcell = False

    for _ in range(max(1, min(8, max_steps))):
        payload = client.tool_complete(messages, WORKCELL_TOOLS)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        content = str(message.get("content") or "").strip()
        if not tool_calls:
            if blocked_issue:
                return simulation.block_request(raw_text, blocked_issue, reply=content or blocked_issue, provider="groq")
            if changed_workcell or simulation.snapshot().get("plan"):
                return simulation.set_ai_reply(content or "The simulator completed the requested control step.")
            return simulation.reply(raw_text, content or "I could not map that request to a safe workcell action.", "groq")

        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls})
        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            if name == "execute_placement":
                if style and "style" not in args:
                    args["style"] = style
                if gesture and "gesture" not in args:
                    args["gesture"] = gesture
            result = dispatcher.call(name, args)
            changed_workcell = changed_workcell or name in mutating_tools and "error" not in result
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", new_id("tool")),
                "name": name,
                "content": json.dumps(result, separators=(",", ":")),
            })

    if blocked_issue:
        return simulation.block_request(raw_text, blocked_issue, reply=blocked_issue, provider="groq")
    if changed_workcell or simulation.snapshot().get("plan"):
        return simulation.set_ai_reply("The requested workcell action was accepted; the simulator remains authoritative for completion.")
    return simulation.reply(raw_text, "The AI control loop reached its safety step limit without an executable action.", "groq")
