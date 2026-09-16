from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voxhands.robustness import robust_eval

try:
    from voxhands.assets.scene_dinner import DinnerConfig, OBJECT_REGISTRY
    from voxhands.mujoco_workcell import DinnerTableWorkcell

    MUJOCO_IMPORTS_FAILED = None
except Exception as exc:
    DinnerConfig = None  # type: ignore
    OBJECT_REGISTRY = {}  # type: ignore
    DinnerTableWorkcell = None  # type: ignore
    MUJOCO_IMPORTS_FAILED = str(exc)


def seed_config(seed: int) -> Any:
    rng = np.random.default_rng(seed)
    cfg = DinnerConfig(seed=seed)
    for oid in OBJECT_REGISTRY:
        cfg.object_yaws[oid] = float(rng.uniform(-np.pi, np.pi))
    for oid, (hx, hy) in {
        "plate": (0.26, 0.60),
        "cup": (0.74, 0.60),
        "spoon": (0.40, 0.76),
        "fork": (0.60, 0.76),
        "bottle": (0.30, 0.86),
    }.items():
        cfg.object_offset[oid] = (
            float(rng.uniform(-0.012, 0.012)),
            float(rng.uniform(-0.012, 0.012)),
        )
    for oid in OBJECT_REGISTRY:
        cfg.object_scales[oid] = float(rng.uniform(0.96, 1.04))
        cfg.object_masses[oid] = float(rng.uniform(0.8, 1.2)) * OBJECT_REGISTRY[oid]["mass"]
    return cfg


def make_factory(oracle: bool):
    def factory(seed: int) -> Any:
        return DinnerTableWorkcell(seed_config(seed), oracle=oracle)

    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description="Run robust_eval over perturbed MuJoCo dinner scenes")
    parser.add_argument("--oracle", action="store_true", help="read object/arm poses from simulator ground truth")
    parser.add_argument("--seeds", type=int, default=10, help="number of perturbed scenes to evaluate")
    parser.add_argument("--vision", action="store_true", help="route object poses through the overhead-camera detector (forces --oracle off)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if MUJOCO_IMPORTS_FAILED is not None:
        print(f"muJoCo unavailable ({MUJOCO_IMPORTS_FAILED}); cannot run robust_eval")
        return 1

    oracle = args.oracle and not args.vision
    factory = make_factory(oracle)
    result = robust_eval(factory, seeds=args.seeds, oracle=oracle)
    print(f"robust_eval: seeds={result['seeds']}, seeds_ok={result['seeds_ok']}, all_ok={result['all_ok']}, elapsed_s={result['elapsed_s']}, oracle={result['oracle']}")
    for run in result["runs"]:
        print(f"  seed {run['seed']}: {'OK' if run['ok'] else 'FAIL'} failures={run['failures'] if run['failures'] else 'none'}")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"Detailed results written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())