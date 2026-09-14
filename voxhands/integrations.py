from __future__ import annotations

from typing import Any


def openvino_status() -> dict[str, Any]:
    try:
        import openvino as ov  # type: ignore

        devices = list(ov.Core().available_devices)
        accelerators = [device for device in devices if device in {"NPU", "GPU"} or device.startswith("GPU")]
        return {
            "name": "OpenVINO",
            "available": True,
            "active": False,
            "mode": "Runtime detected; inference adapter not implemented",
            "version": getattr(ov, "__version__", "installed"),
            "devices": devices,
            "accelerators": accelerators,
        }
    except Exception:
        return {
            "name": "OpenVINO",
            "available": False,
            "active": False,
            "mode": "Optional runtime unavailable; keyword planner active",
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
    return {
        "openvino": openvino_status(),
        "speechmatics": speechmatics_status(),
        "robotics": {"name": "MuJoCo / LeRobot", "available": False, "active": False, "mode": "Robot execution adapter not implemented"},
        "simulator": {"name": "VoxHands deterministic simulator", "available": True, "active": True, "mode": "Deterministic simulation; no physical hardware"},
    }

