"""OpenVINO self-test for the host that actually serves the dashboard.

VoxHands runs on the Python standard library alone, so a bare checkout has no
OpenVINO runtime and the dashboard honestly reports the integration as
unavailable.  The reference deployment additionally installs the runtime
(``requirements-render.txt``), so the live demo can report *genuine*
compiled-model CPU inference measured on the machine answering requests rather
than a number borrowed from a development laptop.

The self-test records the processor brand string it measured on and never
assumes a vendor: hosted CPUs vary (the reference Render worker reports an AMD
EPYC), and Intel-specific NPU/GPU acceleration stays gated on Intel hardware.

Everything in this module is best-effort and failure-isolated.  A missing
runtime, a failed model build, or a slow host degrades to an explanatory
record; it must never stop the server from booting or serving.
"""

from __future__ import annotations

import copy
import json
import platform
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Model shapes exercised by the deployed self-test.  ``policy`` is the trained
# workcell policy; ``workcell-large`` is a deployment-scale stand-in where the
# work is actually compute-bound rather than dispatch-bound.
MODELS: dict[str, list[int]] = {
    "policy": [8, 16, 16, 8],
    "workcell-large": [256, 2048, 2048, 128],
}

_STATE_LOCK = threading.Lock()
_SELF_TEST: dict[str, Any] = {"state": "idle", "available": False}
_PROBE: dict[str, Any] | None = None


def host_cpu_model() -> str:
    """Best-effort brand string for the processor running this process."""
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    if platform.system() == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    try:
        value = platform.processor() or platform.machine()
        if value:
            return str(value)
    except Exception:
        pass
    return "unknown CPU"


def runtime_probe(refresh: bool = False) -> dict[str, Any]:
    """Import the runtime once and report the devices it exposes (cached).

    Called at boot and by every ``/api/intel`` request, so the ``ov.Core()``
    construction is memoised rather than repeated.  ``refresh`` re-probes; it
    exists for tests that toggle runtime availability within one process.
    """
    global _PROBE
    with _STATE_LOCK:
        if _PROBE is not None and not refresh:
            return copy.deepcopy(_PROBE)
    try:
        import openvino as ov  # type: ignore
    except Exception as exc:
        probe: dict[str, Any] = {
            "installable": False,
            "reason": f"openvino is not importable on this host: {exc}",
            "version": None,
            "devices": [],
            "accelerators": [],
        }
    else:
        try:
            devices = [str(device) for device in ov.Core().available_devices]
        except Exception as exc:  # pragma: no cover - depends on host runtime
            probe = {
                "installable": False,
                "reason": f"openvino is installed but the runtime failed to start: {exc}",
                "version": getattr(ov, "__version__", "installed"),
                "devices": [],
                "accelerators": [],
            }
        else:
            probe = {
                "installable": True,
                "reason": None,
                "version": getattr(ov, "__version__", "installed"),
                "devices": devices,
                "accelerators": [
                    device
                    for device in devices
                    if device == "NPU" or device.startswith("GPU")
                ],
            }
    with _STATE_LOCK:
        _PROBE = probe
        return copy.deepcopy(probe)


def self_test_status() -> dict[str, Any]:
    """Current self-test record; safe to call on any request."""
    with _STATE_LOCK:
        status = copy.deepcopy(_SELF_TEST)
    status.setdefault("host_cpu", host_cpu_model())
    probe = runtime_probe()
    status["installable"] = probe["installable"]
    status["available_devices"] = probe["devices"]
    status["accelerators"] = probe["accelerators"]
    if status.get("openvino_version") is None:
        status["openvino_version"] = probe["version"]
    return status


def _param_count(sizes: list[int]) -> int:
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def run_self_test(
    iterations: int = 200,
    repeats: int = 3,
    requests: int = 1500,
    jobs: int = 8,
    device: str = "CPU",
) -> dict[str, Any]:
    """Build real IRs and measure compiled-model inference on this host.

    Kept deliberately small so it finishes in about a second on a shared CPU:
    the goal is verifiable evidence that the Intel runtime is live on the demo
    host, not a records-grade benchmark (``scripts/benchmark_openvino.py``
    remains the full sweep, including NNCF INT8).
    """
    import numpy as np

    from .openvino_adapter import (
        OpenVINOAdapter,
        benchmark_inference,
        benchmark_throughput,
        load_policy_weights,
        make_mlp_weights,
    )
    from .policy import ensure_policy_weights

    started = time.perf_counter()
    report: dict[str, Any] = {
        "state": "ready",
        "available": True,
        "device": device,
        "host_cpu": host_cpu_model(),
        "models": {},
        "async": {},
        "concurrency_gain": {},
        "error": None,
    }

    with tempfile.TemporaryDirectory(prefix="voxhands-openvino-") as tmp:
        workdir = Path(tmp)
        adapter = OpenVINOAdapter()
        # Trained-policy weights are built into a temp dir: the deployed
        # filesystem is ephemeral and the repo stays clean either way.
        policy_weights = load_policy_weights(
            ensure_policy_weights(workdir / "policy.json")
        )
        for name, sizes in MODELS.items():
            if name == "policy":
                weights = policy_weights
            else:
                weights = make_mlp_weights(sizes, seed=0)
            xml_path = adapter.to_ir(weights, workdir / "ir" / name, model_name=name)
            sample = np.zeros((1, sizes[0]), dtype=np.float32)
            latency = benchmark_inference(
                xml_path,
                iterations=iterations,
                device=device,
                repeats=repeats,
                sample=sample,
            )
            latency["sizes"] = sizes
            latency["params"] = _param_count(sizes)
            report["models"][name] = latency
            throughput = benchmark_throughput(
                xml_path,
                requests=requests,
                jobs=jobs,
                device=device,
                sample=sample,
                performance_mode=None,
            )
            report["async"][name] = throughput
            sequential = latency.get("throughput_infers_per_s")
            measured = throughput.get("throughput_infers_per_s")
            if sequential and measured:
                report["concurrency_gain"][name] = round(measured / sequential, 2)

    probe = runtime_probe()
    report["openvino_version"] = probe["version"]
    report["available_devices"] = probe["devices"]
    report["accelerators"] = probe["accelerators"]
    report["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    report["measured_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report["note"] = (
        "Measured on this deployment's own CPU with the real OpenVINO runtime "
        "(compiled IR, CPU plugin). NPU/GPU acceleration is gated on hardware "
        "and is not claimed here."
    )
    return report


def _worker(**kwargs: Any) -> None:
    try:
        report = run_self_test(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on host runtime
        report = {
            "state": "unavailable",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "host_cpu": host_cpu_model(),
            "models": {},
            "async": {},
        }
    with _STATE_LOCK:
        _SELF_TEST.clear()
        _SELF_TEST.update(report)


def start_self_test(**kwargs: Any) -> str:
    """Run the self-test once, in the background.  Returns the resulting state."""
    probe = runtime_probe()
    if not probe["installable"]:
        with _STATE_LOCK:
            _SELF_TEST.clear()
            _SELF_TEST.update(
                {
                    "state": "unavailable",
                    "available": False,
                    "error": probe["reason"],
                    "host_cpu": host_cpu_model(),
                    "models": {},
                    "async": {},
                }
            )
        return "unavailable"
    with _STATE_LOCK:
        if _SELF_TEST.get("state") in {"measuring", "ready"}:
            return str(_SELF_TEST["state"])
        _SELF_TEST.clear()
        _SELF_TEST.update({"state": "measuring", "available": False, "host_cpu": host_cpu_model()})
    threading.Thread(
        target=_worker, kwargs=kwargs, name="openvino-self-test", daemon=True
    ).start()
    return "measuring"


if __name__ == "__main__":
    print(json.dumps(run_self_test(), indent=2))
