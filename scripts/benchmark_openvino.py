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
    benchmark_throughput,
    load_policy_weights,
    make_mlp_weights,
    quantize_ir,
)
from voxhands.policy import ensure_policy_weights

# Name -> [input, hidden..., output].  ``policy`` matches the trained workcell
# policy exactly; the larger entries are deployment-scale models used to show
# where INT8 weight compression actually pays off (compute-bound vs. the
# dispatch-bound tiny models).
MODEL_LIBRARY: dict[str, list[int]] = {
    "policy": [8, 16, 16, 8],
    "vision": [3, 512, 256, 4],
    "workcell-large": [256, 2048, 2048, 128],
    "workcell-xlarge": [512, 4096, 4096, 256],
}


def _parse_sizes(text: str) -> list[int]:
    sizes = [int(part) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(sizes) < 2:
        raise argparse.ArgumentTypeError("--sizes needs at least input,output dims")
    return sizes


def _build_case(name: str, sizes: list[int], args: argparse.Namespace, adapter: OpenVINOAdapter):
    ir_dir = args.ir_dir / name
    int8_dir = args.int8_dir / name
    if name == "policy":
        weights = load_policy_weights(ensure_policy_weights(args.weights))
    else:
        weights = make_mlp_weights(sizes, seed=args.seed)
    fp32_xml = adapter.to_ir(weights, ir_dir, model_name=name)
    calibration = np.random.default_rng(args.seed).normal(
        size=(args.calib_samples, sizes[0])
    ).astype(np.float32)
    int8_xml = None
    if args.no_int8:
        int8_xml = None
    else:
        try:
            int8_xml = quantize_ir(fp32_xml, int8_dir, calibration, subset_size=args.subset_size, model_name=f"{name}_int8")
        except RuntimeError as exc:
            print(f"[{name}] INT8 skipped: {exc}")
            int8_xml = None
    return fp32_xml, int8_xml, int(sizes[0])


def _measure(xml_path: Path, args: argparse.Namespace, input_dim: int) -> dict:
    sample = np.random.default_rng(args.seed + 1).normal(size=(1, input_dim)).astype(np.float32)
    report = benchmark_inference(
        xml_path,
        iterations=args.iters,
        device=args.device,
        sample=sample,
        repeats=args.repeats,
    )
    if args.throughput:
        report["throughput"] = benchmark_throughput(
            xml_path,
            requests=args.requests,
            jobs=args.jobs,
            device=args.device,
            num_streams=args.streams,
            sample=sample,
        )
    return report


def _case_report(name: str, sizes: list[int], args: argparse.Namespace, adapter: OpenVINOAdapter) -> dict:
    fp32_xml, int8_xml, input_dim = _build_case(name, sizes, args, adapter)
    fp32 = _measure(fp32_xml, args, input_dim)
    entry: dict = {"sizes": sizes, "fp32": fp32}
    if int8_xml is not None:
        int8 = _measure(int8_xml, args, input_dim)
        entry["int8"] = int8
        if int8["latency_ms_mean"]:
            entry["int8_vs_fp32_speedup"] = round(
                fp32["latency_ms_mean"] / int8["latency_ms_mean"], 3
            )
        entry["int8_bin_ratio"] = round(
            int8["bin_size_bytes"] / max(1, fp32["bin_size_bytes"]), 3
        )
    return entry


def _print_table(results: dict[str, dict]) -> None:
    print()
    header = f"{'model':<16}{'params':>10}{'fp32 ms':>10}{'int8 ms':>10}{'speedup':>9}{'size x':>8}"
    print(header)
    print("-" * len(header))
    for name, entry in results.items():
        fp32 = entry["fp32"]["latency_ms_mean"]
        int8 = entry.get("int8", {}).get("latency_ms_mean")
        speedup = entry.get("int8_vs_fp32_speedup")
        ratio = entry.get("int8_bin_ratio")
        params = _param_count(entry["sizes"])
        print(
            f"{name:<16}{params:>10,}{fp32:>10.4f}"
            f"{(f'{int8:.4f}' if int8 is not None else '   n/a'):>10}"
            f"{(f'{speedup:.2f}x' if speedup else '    n/a'):>9}"
            f"{(f'{ratio:.2f}' if ratio else '   n/a'):>8}"
        )


def _param_count(sizes: list[int]) -> int:
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenVINO CPU latency/throughput benchmark with an INT8 sweep")
    parser.add_argument("--weights", type=Path, default=Path("data/policy.json"))
    parser.add_argument("--ir-dir", type=Path, default=Path("data/openvino_ir"))
    parser.add_argument("--int8-dir", type=Path, default=Path("data/openvino_ir_int8"))
    parser.add_argument("--device", type=str, default="CPU")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sweep", action="store_true", help="benchmark every model in the library")
    parser.add_argument("--models", type=str, default="", help="comma-separated subset of the library")
    parser.add_argument("--sizes", type=str, default="", help="custom layer sizes, e.g. 8,16,16,8")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--subset-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-int8", action="store_true", help="skip NNCF quantization")
    parser.add_argument("--throughput", action="store_true", help="also measure async aggregate throughput")
    parser.add_argument("--requests", type=int, default=4000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--streams", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        import openvino as ov
    except Exception as exc:
        print(f"openvino not importable on this host: {exc}")
        return 1

    devices = [str(d) for d in ov.Core().available_devices]
    if args.device not in devices and args.device != "AUTO":
        print(f"Requested device {args.device!r} not in {devices}; using CPU.")
        args.device = "CPU"

    if args.sizes:
        cases = {"custom": _parse_sizes(args.sizes)}
    elif args.models:
        names = [n.strip() for n in args.models.split(",") if n.strip()]
        cases = {n: MODEL_LIBRARY[n] for n in names}
    elif args.sweep:
        cases = dict(MODEL_LIBRARY)
    else:
        cases = {"policy": MODEL_LIBRARY["policy"]}

    adapter = OpenVINOAdapter()
    results: dict[str, dict] = {}
    for name, sizes in cases.items():
        print(f"[{name}] {sizes} ...")
        results[name] = _case_report(name, sizes, args, adapter)

    report = {
        "device": args.device,
        "available_devices": devices,
        "iters": args.iters,
        "repeats": args.repeats,
        "throughput_measured": args.throughput,
        "models": results,
        "note": (
            "Measured on the actual OpenVINO compiled model. INT8 gain appears once the model "
            "is compute-bound; the tiny workcell policy is dispatch-bound, so INT8 is neutral "
            "there. NPU/GPU acceleration is gated on Intel hardware and is not claimed here."
        ),
    }
    print(json.dumps(report, indent=2))
    _print_table(results)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nBenchmark report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
