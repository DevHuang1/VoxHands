from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxhands.openvino_adapter import OpenVINOAdapter, load_policy_weights
from voxhands.policy import ensure_policy_weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the policy MLP to a real OpenVINO IR (.xml/.bin)")
    parser.add_argument("--weights", type=Path, default=Path("data/policy.json"))
    parser.add_argument("--out", type=Path, default=Path("data/openvino_ir"))
    parser.add_argument("--name", type=str, default="policy_mlp")
    parser.add_argument("--numpy-artifact", action="store_true", help="also write the numpy .npz fallback artifact")
    args = parser.parse_args()

    weights_path = ensure_policy_weights(args.weights)
    model_weights = load_policy_weights(weights_path)
    adapter = OpenVINOAdapter()
    status = adapter.status()
    print(f"OpenVINO status: installed={status['installed']}, devices={status['devices']}, verified={status['verified']}")

    xml_path = adapter.to_ir(model_weights, args.out, model_name=args.name)
    bin_path = xml_path.with_suffix(".bin")
    print(f"Real OpenVINO IR written: {xml_path} ({xml_path.stat().st_size} B) + {bin_path.name} ({bin_path.stat().st_size} B)")
    print("Load it with: ov.Core().read_model(...) / compile_model(...)")

    if args.numpy_artifact:
        artifact_dir = adapter.convert(model_weights, args.out / "numpy_fallback")
        print(f"numpy fallback artifact written to {artifact_dir} (weights.npz + manifest.json)")

    in_dim = int(model_weights["weights"][0].shape[0])
    out_dim = int(model_weights["weights"][-1].shape[1])
    print(f"IR input dim={in_dim}, output dim={out_dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
