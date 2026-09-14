from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .models import Action, Plan, new_id
from .planner import OBJECTS, TARGETS, build_plan
from .styles import apply_motion_preferences, normalize_gesture, normalize_style


def load_env_files(*paths: str) -> None:
    """Load KEY=VALUE lines from dotenv files without overriding the process env.

    Minimal standard-library stand-in for python-dotenv so VoxHands keeps its
    zero hard-dependency posture.
    """
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except OSError:
            continue


_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_env_files(str(Path.cwd() / ".env.local"), str(Path.cwd() / ".env"), str(_PROJECT_ROOT / ".env.local"), str(_PROJECT_ROOT / ".env"))


GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
DEFAULT_ACTION_MS = 5000

SYSTEM_PROMPT = """You are the Groq-powered controller for VoxHands, a dual-arm tabletop workcell simulator.

KNOWN OBJECTS (id -> label, aliases):
- "blue_plate" / Blue plate (blue plate, plate)
- "cup" / Cup (cup, mug, glass)
- "fork" / Fork (fork)
- "spoon" / Spoon (spoon)

KNOWN PLACEMENT TARGETS (id -> label, arm that must handle the object):
- "left_place" (left place, left side) -> left arm
- "right_place" (right place, right side) -> right arm
- "center_place" (center place, middle) -> left arm
- "cup_interior" (inside/into the cup) -> right arm; ONLY the spoon fits inside the cup
- "plate_surface" (on/onto the plate) -> left arm; ONLY the cup fits on the plate

MOVEMENT STYLES (id -> meaning): standard (default), gentle (slow/soft/careful), precise (exact/accurate), rapid (fast/quick), playful (fun/bouncy), wavy (zigzag/sinuous).
GESTURES (id -> when the arms celebrate after finishing a task): none (default), wave (greet/hi/hello), bow, dance, point (present).
SPEED is a multiplier: 0.5 very slow, 1.0 normal, 2.0 rapid. Pick a sensible speed for the style.

SAFETY RULES (must be respected):
- The RED item is an unmovable no-go safety barrier. Never pick it up, move it, or target it.
- Only the spoon may go into the cup; only the cup may go onto the plate; never place the plate into the cup.
- Closely spaced objects are guarded by the simulator; keep the plan sequential when the route may cross the barrier.
- If the command is ambiguous (for example an object without a destination), do not invent a destination; return the recognized object under recognized_objects and explain in reply.

CONVERSATION GUIDELINES:
- For greetings, questions, thanks, or any message that cannot map to a task, set intent "conversation" and actions [].
- For "home both arms" use intent "home". For "calibration status" use intent "calibration".
- Reply concisely (1-2 sentences) in a friendly operator voice. Confirm the plan, and mention the style, gesture, or condition the user asked for.
- Read the user's wording for style cues: gently/slow/soft/careful -> gentle; fast/quick/speedy -> rapid; wave/greet/hi -> gesture wave; dance -> gesture dance; zigzag/wiggly -> wavy.

Respond with ONE JSON object, no prose, shaped exactly like:
{
  "intent": "object_placement" | "home" | "calibration" | "conversation",
  "actions": [
    {
      "object_id": "blue_plate" | "cup" | "fork" | "spoon",
      "target_id": "left_place" | "right_place" | "center_place" | "cup_interior" | "plate_surface",
      "style": "standard" | "gentle" | "precise" | "rapid" | "playful" | "wavy",
      "speed": 0.5,
      "gesture": "none" | "wave" | "bow" | "dance" | "point",
      "pause_ms": 0,
      "condition": "string or empty"
    }
  ],
  "constraints": ["avoid_red_zone", "no_arm_collision"],
  "recognized_objects": ["Blue plate", "Cup"],
  "suggestions": [{"label": "Cup to right", "text": "Place the cup on the right."}],
  "reply": "string"
}"""


class GroqClient:
    """Minimal OpenAI-compatible client for Groq's chat completions endpoint.

    Uses only the standard library so VoxHands keeps its zero hard-dependency
    posture. A response is only attempted when GROQ_API_KEY is set.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, fallback_model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.fallback_model = fallback_model or os.getenv("GROQ_FALLBACK_MODEL", FALLBACK_MODEL)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def status(self) -> dict[str, Any]:
        configured = self.available
        return {
            "name": "Groq AI",
            "available": configured,
            "active": configured,
            "mode": f"Groq LLM conversational control ({self.model})" if configured else "GROQ_API_KEY not set; keyword planner active",
            "model": self.model if configured else None,
            "fallback_model": self.fallback_model if configured else None,
        }

    def _complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1200,
        json_mode: bool = True,
        model: str | None = None,
    ) -> str:
        if not self.available:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            GROQ_ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "VoxHands/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            message = error.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Groq API error {error.code}: {message}") from error
        return str(payload["choices"][0]["message"]["content"])

    def parse_command(self, text: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        errors: list[str] = []
        models = list(dict.fromkeys((self.model, self.fallback_model)))
        for model in models:
            try:
                content = self._complete(messages, model=model)
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("Groq returned a non-object response.")
                return parsed
            except Exception as error:
                errors.append(f"{model}: {error}")
        raise RuntimeError("Groq request failed for all configured models. " + " | ".join(errors))


def _float(value: Any, default: float, minimum: float = 0.3, maximum: float = 3.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(15_000, number))


def plan_from_ai_payload(raw_text: str, payload: dict[str, Any]) -> Plan:
    """Build a Plan from Groq's JSON payload, rejecting unknown references."""
    intent = str(payload.get("intent", "conversation")).strip()
    if intent not in {"object_placement", "home", "calibration", "conversation"}:
        intent = "conversation"

    mode = {"home": "home", "calibration": "calibration", "conversation": "conversation"}.get(intent, "command")

    actions: list[Action] = []
    issues: list[str] = []
    recognized: list[str] = []
    for raw in payload.get("actions") or []:
        if not isinstance(raw, dict):
            continue
        object_id = str(raw.get("object_id", "")).strip()
        target_id = str(raw.get("target_id", "")).strip()
        if object_id not in OBJECTS or target_id not in TARGETS:
            issues.append(f"AI produced an unknown reference: {object_id or '?'} -> {target_id or '?'}.")
            continue
        arm = TARGETS[target_id]["arm"]
        style = normalize_style(raw.get("style"))
        gesture = normalize_gesture(raw.get("gesture"))
        speed = _float(raw.get("speed"), 1.0)
        pause_ms = _int(raw.get("pause_ms"), 0)
        condition = str(raw.get("condition", "")).strip()
        action = Action(
            new_id("act"),
            arm,
            object_id,
            OBJECTS[object_id]["label"],
            target_id,
            TARGETS[target_id]["label"],
            duration_ms=DEFAULT_ACTION_MS,
            style=style,
            speed=speed,
            gesture=gesture,
            pause_ms=pause_ms,
            condition=condition,
        )
        actions.append(action)
        if object_id not in {item for item in recognized}:
            recognized.append(OBJECTS[object_id]["label"])

    plan = Plan(
        new_id("plan"),
        raw_text,
        intent,
        actions,
        list(payload.get("constraints") or ["avoid_red_zone", "no_arm_collision"]),
        safety_issues=issues,
        recognized_objects=recognized,
        suggestions=list(payload.get("suggestions") or []),
    )
    plan.llm_reply = str(payload.get("reply", ""))
    plan.llm_provider = "groq"
    plan.mode = mode
    return plan


def ai_build_plan(
    raw_text: str,
    client: GroqClient | None = None,
    style: object | None = None,
    gesture: object | None = None,
) -> tuple[Plan, str]:
    """Plan a command with Groq when configured, otherwise fall back to the regex planner.

    Returns a (plan, reply) tuple. The reply is the AI's natural language
    response; it is empty when the keyword planner handled the command.
    """
    client = client or GroqClient()
    if client.available:
        try:
            payload = client.parse_command(raw_text)
            plan = plan_from_ai_payload(raw_text, payload)
        except Exception:
            plan = build_plan(raw_text)
            plan.llm_provider = "keyword"
            plan.mode = "conversation" if plan.intent == "conversation" else "command"
    else:
        plan = build_plan(raw_text)
        plan.llm_provider = "keyword"
        plan.mode = "conversation" if plan.intent == "conversation" else "command"

    apply_motion_preferences(plan, style or None, gesture or None)
    return plan, plan.llm_reply
