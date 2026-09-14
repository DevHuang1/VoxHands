from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


@dataclass
class Action:
    id: str
    arm: str
    object_id: str
    object_label: str
    target_id: str
    target_label: str
    duration_ms: int = 1400
    status: str = "queued"
    progress: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "arm": self.arm,
            "object_id": self.object_id,
            "object_label": self.object_label,
            "target_id": self.target_id,
            "target_label": self.target_label,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "progress": round(self.progress, 1),
            "note": self.note,
        }


@dataclass
class Plan:
    id: str
    raw_text: str
    intent: str
    actions: list[Action]
    constraints: list[str]
    status: str = "ready"
    safety_issues: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    recognized_objects: list[str] = field(default_factory=list)
    suggestions: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "raw_text": self.raw_text,
            "intent": self.intent,
            "actions": [action.to_dict() for action in self.actions],
            "constraints": self.constraints,
            "status": self.status,
            "safety_issues": self.safety_issues,
            "created_at": self.created_at,
            "recognized_objects": self.recognized_objects,
            "suggestions": self.suggestions,
        }

