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
    quantize_ir,
)
from voxhands.policy import ensure_policy_weights


def main() -> int:
    parser = argparse.ArgumentParser(description="INT8 post-training quantization (NNCF) of the policy IR")
    parser.add_argument("--weights", type=Path, default=Path("data/policy.json"))
    parser.add_argument("--ir-dir", type=Path, default=Path("data/openvino_ir"))
    parser.add_argument("--out", type=Path, default=Path("data/openvino_ir_int8"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--subset-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        import nncf  # noqa: F401
        import openvino as ov
    except Exception as exc:
        print(f"nncf/openvino not importable on this host: {exc}")
        return 1

    weights_path = ensure_policy_weights(args.weights)
    model_weights = load_policy_weights(weights_path)
    input_dim = int(model_weights["weights"][0].shape[0])

    adapter = OpenVINOAdapter(model_weights=model_weights)
    fp32_xml = args.ir_dir / "policy_mlp.xml"
    if not fp32_xml.exists():
        fp32_xml = adapter.to_ir(model_weights, args.ir_dir)
        print(f"FP32 IR missing; exported {fp32_xml}")

    rng = np.random.default_rng(args.seed)
    calibration = rng.normal(size=(args.samples, input_dim)).astype(np.float32)

    int8_xml = quantize_ir(fp32_xml, args.out, calibration, subset_size=args.subset_size)
    bin_size = int8_xml.with_suffix(".bin").stat().st_size if int8_xml.with_suffix(".bin").exists() else 0
    print(f"INT8 IR written: {int8_xml} ({int8_xml.stat().st_size} B) + .bin ({bin_size} B)")

    core = ov.Core()
    fp32_model = core.compile_model(str(fp32_xml), "CPU")
    int8_model = core.compile_model(str(int8_xml), "CPU")
    probe = rng.normal(size=(1, input_dim)).astype(np.float32)
    fp32_out = np.asarray(fp32_model([probe])[fp32_model.output(0)]).reshape(-1)
    int8_out = np.asarray(int8_model([probe])[int8_model.output(0)]).reshape(-1)
    max_abs = float(np.max(np.abs(fp32_out - int8_out)))
    rel = max_abs / (float(np.max(np.abs(fp32_out))) + 1e-9)
    print(f"FP32 vs INT8 max abs diff={max_abs:.6f} (rel={rel:.4%})")

    fp32_bench = benchmark_inference(fp32_xml, iterations=args.iters, sample=probe)
    int8_bench = benchmark_inference(int8_xml, iterations=args.iters, sample=probe)
    speedup = (
        round(fp32_bench["latency_ms_mean"] / int8_bench["latency_ms_mean"], 3)
        if int8_bench["latency_ms_mean"]
        else None
    )
    report = {
        "device": "CPU",
        "fp32": fp32_bench,
        "int8": int8_bench,
        "int8_vs_fp32_speedup": speedup,
        "parity_max_abs_diff": round(max_abs, 6),
        "parity_rel_diff": round(rel, 6),
        "note": "INT8 PTQ verified on CPU; NPU/GPU acceleration is gated on Intel hardware.",
    }
    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"Quantization report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
