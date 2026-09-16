from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voxhands.openvino_adapter import (
    OpenVINOAdapter,
    benchmark_inference,
    load_policy_weights,
)
from voxhands.policy import ensure_policy_weights


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenVINO CPU latency/throughput benchmark for the policy IR")
    parser.add_argument("--weights", type=Path, default=Path("data/policy.json"))
    parser.add_argument("--ir-dir", type=Path, default=Path("data/openvino_ir"))
    parser.add_argument("--int8-dir", type=Path, default=Path("data/openvino_ir_int8"))
    parser.add_argument("--device", type=str, default="CPU")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        import openvino as ov
    except Exception as exc:
        print(f"openvino not importable on this host: {exc}")
        return 1

    weights_path = ensure_policy_weights(args.weights)
    model_weights = load_policy_weights(weights_path)
    input_dim = int(model_weights["weights"][0].shape[0])
    core = ov.Core()
    devices = [str(d) for d in core.available_devices]
    if args.device not in devices and args.device != "AUTO":
        print(f"Requested device {args.device!r} not in {devices}; using CPU.")
        args.device = "CPU"

    adapter = OpenVINOAdapter(model_weights=model_weights)
    fp32_xml = args.ir_dir / "policy_mlp.xml"
    if not fp32_xml.exists():
        fp32_xml = adapter.to_ir(model_weights, args.ir_dir)
        print(f"FP32 IR missing; exported {fp32_xml}")

    probe = np.random.default_rng(args.seed).normal(size=(1, input_dim)).astype(np.float32)
    report: dict[str, object] = {
        "device": args.device,
        "available_devices": devices,
        "input_dim": input_dim,
        "fp32": benchmark_inference(fp32_xml, iterations=args.iters, device=args.device, sample=probe),
    }

    int8_xml = args.int8_dir / "policy_mlp_int8.xml"
    if int8_xml.exists():
        report["int8"] = benchmark_inference(int8_xml, iterations=args.iters, device=args.device, sample=probe)
        fp32_ms = report["fp32"]["latency_ms_mean"]  # type: ignore[index]
        int8_ms = report["int8"]["latency_ms_mean"]  # type: ignore[index]
        report["int8_vs_fp32_speedup"] = round(fp32_ms / int8_ms, 3) if int8_ms else None
    else:
        report["int8"] = None
        print(f"INT8 IR not found at {int8_xml}; run scripts/quantize_openvino.py to add it.")

    report["note"] = (
        "Measured on the actual OpenVINO compiled model; NPU/GPU acceleration is "
        "gated on Intel hardware and is not claimed here."
    )
    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"Benchmark report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
