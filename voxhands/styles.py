from __future__ import annotations

import math


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def minimum_jerk(progress: float) -> float:
    """Quintic easing with zero velocity and acceleration at both endpoints."""
    t = clamp01(progress)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)


def _ease_out(t: float) -> float:
    return 1 - (1 - t) ** 3


def _ease_in(t: float) -> float:
    return t * t * t


STYLE_ALIASES = {
    "standard": "standard", "normal": "standard", "default": "standard", "balanced": "standard",
    "gentle": "gentle", "soft": "gentle", "careful": "gentle", "delicate": "gentle",
    "slow": "gentle", "gently": "gentle",
    "precise": "precise", "exact": "precise", "accurate": "precise", "fine": "precise",
    "surgical": "precise", "pin-point": "precise",
    "rapid": "rapid", "fast": "rapid", "quick": "rapid", "speedy": "rapid", "hurried": "rapid",
    "playful": "playful", "fun": "playful", "cheerful": "playful", "lively": "playful", "bouncy": "playful",
    "wavy": "wavy", "zigzag": "wavy", "sinuous": "wavy", "snake": "wavy", "serpentine": "wavy", "wiggle": "wavy",
}

STYLES = {
    "standard": {"label": "Standard", "speed": 1.0, "easing": "minimum_jerk", "lift": 0.28, "trajectory": "straight", "description": "Balanced minimum-jerk motion."},
    "gentle": {"label": "Gentle", "speed": 0.55, "easing": "smoothstep", "lift": 0.36, "trajectory": "straight", "description": "Slow, smooth, high-lift handling."},
    "precise": {"label": "Precise", "speed": 0.72, "easing": "minimum_jerk", "lift": 0.22, "trajectory": "straight", "description": "Low, careful, straight-line precision."},
    "rapid": {"label": "Rapid", "speed": 1.7, "easing": "ease_out", "lift": 0.22, "trajectory": "straight", "description": "Fast workcell transfers."},
    "playful": {"label": "Playful", "speed": 1.2, "easing": "ease_in_out", "lift": 0.40, "trajectory": "bounce", "description": "Buoyant carry path with a lift-and-bob arc."},
    "wavy": {"label": "Wavy", "speed": 1.0, "easing": "minimum_jerk", "lift": 0.30, "trajectory": "wave", "description": "Sinuous lateral weave during transport."},
}

EASINGS = {
    "minimum_jerk": minimum_jerk,
    "smoothstep": _smoothstep,
    "ease_out": _ease_out,
    "ease_in": _ease_in,
}


def easing(style_id: str, progress: float) -> float:
    profile = STYLES.get(style_id, STYLES["standard"])
    return EASINGS.get(profile["easing"], minimum_jerk)(clamp01(progress))


def trajectory_offset(style_id: str, progress: float) -> dict[str, float]:
    """Lateral/bob offsets applied during the transport phase of a style."""
    profile = STYLES.get(style_id, STYLES["standard"])
    t = clamp01(progress)
    if profile["trajectory"] == "wave":
        return {"lateral": math.sin(t * math.tau * 2.0) * 0.055}
    if profile["trajectory"] == "bounce":
        return {"bob": -abs(math.sin(t * math.tau * 1.5)) * 0.018}
    return {}


def normalize_style(value: object) -> str:
    if value is None:
        return "standard"
    key = str(value).strip().lower()
    return STYLE_ALIASES.get(key, "standard")


GESTURE_ALIASES = {
    "none": "none", "no": "none", "off": "none",
    "wave": "wave", "hello": "wave", "hi": "wave", "greet": "wave", "bye": "wave", "goodbye": "wave",
    "bow": "bow", "curtsy": "bow",
    "dance": "dance", "wiggle": "dance", "groove": "dance",
    "point": "point", "present": "point", "indicate": "point", "show": "point", "ta-da": "point",
}


def normalize_gesture(value: object) -> str:
    if value is None:
        return "none"
    key = str(value).strip().lower()
    return GESTURE_ALIASES.get(key, "none")


GESTURES = {
    "none": {"label": "No gesture", "duration_ms": 0, "keyframes": {"left": [], "right": []}},
    "wave": {
        "label": "Wave",
        "duration_ms": 2600,
        "keyframes": {
            "left": [
                {"x": 0.26, "y": 0.52, "z": 0.56, "angle": -12},
                {"x": 0.48, "y": 0.62, "z": 0.60, "angle": -28},
                {"x": 0.14, "y": 0.55, "z": 0.56, "angle": -10},
                {"x": 0.46, "y": 0.62, "z": 0.60, "angle": -30},
                {"x": 0.26, "y": 0.30, "z": 0.62, "angle": -12},
            ],
            "right": [
                {"x": 0.74, "y": 0.52, "z": 0.56, "angle": 12},
                {"x": 0.52, "y": 0.62, "z": 0.60, "angle": 28},
                {"x": 0.86, "y": 0.55, "z": 0.56, "angle": 10},
                {"x": 0.54, "y": 0.62, "z": 0.60, "angle": 30},
                {"x": 0.74, "y": 0.30, "z": 0.62, "angle": 12},
            ],
        },
    },
    "bow": {
        "label": "Bow",
        "duration_ms": 2200,
        "keyframes": {
            "left": [
                {"x": 0.26, "y": 0.30, "z": 0.60, "angle": -12},
                {"x": 0.20, "y": 0.40, "z": 0.38, "angle": -42},
                {"x": 0.16, "y": 0.46, "z": 0.34, "angle": -50},
                {"x": 0.24, "y": 0.30, "z": 0.62, "angle": -12},
            ],
            "right": [
                {"x": 0.74, "y": 0.30, "z": 0.60, "angle": 12},
                {"x": 0.80, "y": 0.40, "z": 0.38, "angle": 42},
                {"x": 0.84, "y": 0.46, "z": 0.34, "angle": 50},
                {"x": 0.76, "y": 0.30, "z": 0.62, "angle": 12},
            ],
        },
    },
    "dance": {
        "label": "Dance",
        "duration_ms": 3400,
        "keyframes": {
            "left": [
                {"x": 0.26, "y": 0.40, "z": 0.60, "angle": -10},
                {"x": 0.52, "y": 0.62, "z": 0.64, "angle": -32},
                {"x": 0.24, "y": 0.30, "z": 0.62, "angle": -12},
                {"x": 0.20, "y": 0.64, "z": 0.60, "angle": -30},
                {"x": 0.26, "y": 0.30, "z": 0.62, "angle": -12},
            ],
            "right": [
                {"x": 0.74, "y": 0.40, "z": 0.60, "angle": 10},
                {"x": 0.74, "y": 0.30, "z": 0.62, "angle": 12},
                {"x": 0.78, "y": 0.64, "z": 0.60, "angle": 30},
                {"x": 0.48, "y": 0.62, "z": 0.64, "angle": 32},
                {"x": 0.74, "y": 0.30, "z": 0.62, "angle": 12},
            ],
        },
    },
    "point": {
        "label": "Point",
        "duration_ms": 2200,
        "keyframes": {
            "left": [
                {"x": 0.26, "y": 0.30, "z": 0.62, "angle": -12},
                {"x": 0.32, "y": 0.64, "z": 0.66, "angle": -44},
                {"x": 0.30, "y": 0.62, "z": 0.64, "angle": -42},
                {"x": 0.26, "y": 0.30, "z": 0.62, "angle": -12},
            ],
            "right": [
                {"x": 0.74, "y": 0.30, "z": 0.62, "angle": 12},
                {"x": 0.68, "y": 0.64, "z": 0.66, "angle": 44},
                {"x": 0.70, "y": 0.62, "z": 0.64, "angle": 42},
                {"x": 0.74, "y": 0.30, "z": 0.62, "angle": 12},
            ],
        },
    },
}


def apply_motion_preferences(plan, style: object | None = None, gesture: object | None = None):
    """Apply UI-selected style/gesture overrides to every action of a plan."""
    style_id = normalize_style(style) if style else None
    gesture_id = normalize_gesture(gesture) if gesture else None
    for action in plan.actions:
        if style_id:
            action.style = style_id
        if gesture_id:
            action.gesture = gesture_id
    return plan