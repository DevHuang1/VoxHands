from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxhands.policy import CanonicalDinnerPolicy, synthetic_demos, train_default_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a voxhands.policy MLPPolicy on demo data")
    parser.add_argument("--out", type=Path, default=Path("data/policy.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-demos", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    demos = synthetic_demos(n=args.n_demos, seed=args.seed)
    policy = train_default_policy(epochs=args.epochs, n_demos=args.n_demos, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(args.out))

    canonical = CanonicalDinnerPolicy()
    canonical_action = canonical({})
    print(f"Trained MLPPolicy on {len(demos)} synthetic (state, action) demo pairs")
    print(f"Saved policy weights (voxhands.policy MLPPolicy JSON) to {args.out}")
    print(f"CanonicalDinnerPolicy sample action: {canonical_action.arm} -> {canonical_action.object_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
