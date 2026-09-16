from __future__ import annotations

import os
from typing import Any


def groq_status() -> dict[str, Any]:
    has_key = bool(os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return {
        "name": "Groq AI",
        "available": has_key,
        "active": has_key,
        "mode": "Groq LLM conversational control; styles & gestures enabled" if has_key else "GROQ_API_KEY not set; keyword planner active",
        "model": model if has_key else None,
        "fallback_model": os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b") if has_key else None,
    }


def _mujoco_available() -> bool:
    try:
        import mujoco  # type: ignore  # noqa: F401
        import voxhands.mujoco_sim  # noqa: F401

        return True
    except Exception:
        return False


def openvino_status() -> dict[str, Any]:
    try:
        import openvino as ov  # type: ignore

        devices = list(ov.Core().available_devices)
        accelerators = [device for device in devices if device in {"NPU", "GPU"} or device.startswith("GPU")]
        active = bool(accelerators) or "CPU" in devices
        mode = "Runtime detected; NPU/GPU/CPU inference adapter ready"
        return {
            "name": "OpenVINO",
            "available": True,
            "active": active,
            "mode": mode,
            "version": getattr(ov, "__version__", "installed"),
            "devices": devices,
            "accelerators": accelerators,
        }
    except Exception:
        return {
            "name": "OpenVINO",
            "available": False,
            "active": False,
            "mode": "Optional runtime unavailable; colour-mask detector active",
            "version": None,
            "devices": [],
            "accelerators": [],
        }


def speechmatics_status() -> dict[str, Any]:
    try:
        import speechmatics.rt  # type: ignore # noqa: F401

        package_available = True
    except Exception:
        package_available = False
    import os

    has_key = bool(os.getenv("SPEECHMATICS_API_KEY"))
    return {
        "name": "Speechmatics",
        "available": package_available and has_key,
        "package_available": package_available,
        "credentials_configured": has_key,
        "active": False,
        "mode": "Package and credentials detected; transcription adapter not implemented" if package_available and has_key else "Browser speech or typed input; server transcription not connected",
    }


def runtime_status() -> dict[str, Any]:
    mujoco_active = _mujoco_available()
    return {
        "groq": groq_status(),
        "openvino": openvino_status(),
        "speechmatics": speechmatics_status(),
        "robotics": {
            "name": "MuJoCo (SO-101-style arms)",
            "available": mujoco_active,
            "active": mujoco_active,
            "mode": "MuJoCo physical simulation; IK + kinematic grasp on two SO-101-style arms" if mujoco_active else "Robot execution adapter unavailable; install mujoco",
        },
        "simulator": {
            "name": "VoxHands deterministic simulator",
            "available": True,
            "active": not mujoco_active,
            "mode": "Deterministic simulation; no physical hardware",
        },
        "vision": {
            "name": "MujocoVision",
            "available": True,
            "active": True,
            "mode": "Overhead camera segmentation; OpenVINO inference when available",
        },
    }
