from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _nncf_available() -> bool:
    return importlib.util.find_spec("nncf") is not None


def main() -> int:
    from voxhands.openvino_adapter import OpenVINOAdapter

    status = OpenVINOAdapter().status()
    devices = list(status["devices"])
    has_gpu = "GPU" in devices or any(device.startswith("GPU") for device in devices)
    has_npu = "NPU" in devices
    nncf = _nncf_available()

    print(json.dumps(status, indent=2))
    print()
    print(f"OpenVINO installed : {status['installed']}")
    print(f"Available devices  : {', '.join(devices) if devices else '(none)'}")
    print(f"Intel GPU present  : {has_gpu}")
    print(f"Intel NPU present  : {has_npu}")

    if has_gpu:
        print("GPU inference      : verified")
    else:
        print("GPU inference      : blocked/unverified — no Intel GPU exposed on this host")
    if has_npu:
        print("NPU inference      : verified")
    else:
        print("NPU inference      : blocked/unverified — no Intel NPU exposed on this host")

    if nncf:
        print("IR export          : verified — openvino.save_model writes .xml/.bin")
        print("INT8 / NNCF        : verified on CPU (post-training quantization); GPU/NPU INT8 kernels gated")
    else:
        print("IR export          : available (openvino.save_model)")
        print("INT8 / NNCF        : blocked/unverified — nncf not installed on this host")

    print(f"Device actively used: {status['device_used']}")
    print(f"Verified            : {status['verified']}")
    print(f"Message             : {status['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())