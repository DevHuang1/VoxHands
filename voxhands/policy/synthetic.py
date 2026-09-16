"""Synthetic demonstration data and a default trained policy artifact.

The OpenVINO / policy scripts need ``data/policy.json`` to exist before they can
export an IR or benchmark inference.  Rather than committing a generated
artifact, the scripts call :func:`ensure_policy_weights`, which trains the same
:class:`~voxhands.policy.MLPPolicy` on the deterministic synthetic demos below
when the file is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .mlp import MLPPolicy

OBJECTS = ("blue_plate", "cup", "fork", "spoon")
CANONICAL_TARGETS = {
    "blue_plate": (0.33, 0.76),
    "cup": (0.66, 0.76),
    "fork": (0.89, 0.69),
    "spoon": (0.24, 0.69),
}
HOMES = {
    "blue_plate": (0.22, 0.69),
    "cup": (0.72, 0.69),
    "fork": (0.60, 0.74),
    "spoon": (0.40, 0.74),
}
DEFAULT_INPUT_DIM = len(OBJECTS) * 2
DEFAULT_OUTPUT_DIM = len(OBJECTS) * 2
DEFAULT_HIDDEN_DIMS = [16, 16]


def synthetic_demos(n: int = 200, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    demos: list[dict[str, Any]] = []
    for _ in range(n):
        state: dict[str, float] = {}
        target: list[float] = []
        for obj in OBJECTS:
            hx, hy = HOMES[obj]
            tx, ty = CANONICAL_TARGETS[obj]
            state[f"{obj}_home_x"] = hx + rng.normal(0, 0.02)
            state[f"{obj}_home_y"] = hy + rng.normal(0, 0.02)
            target.append(tx + rng.normal(0, 0.005))
            target.append(ty + rng.normal(0, 0.005))
        demos.append({"state": state, "target": np.asarray(target, dtype=np.float64)})
    return demos


def train_default_policy(
    epochs: int = 50,
    n_demos: int = 200,
    seed: int = 0,
) -> MLPPolicy:
    demos = synthetic_demos(n=n_demos, seed=seed)
    policy = MLPPolicy(
        input_dim=DEFAULT_INPUT_DIM,
        hidden_dims=list(DEFAULT_HIDDEN_DIMS),
        output_dim=DEFAULT_OUTPUT_DIM,
    )
    for _ in range(epochs):
        policy.train(demos)
    return policy


def ensure_policy_weights(path: str | Path) -> Path:
    """Return ``path``, training and saving the default policy if it is missing."""
    path = Path(path)
    if path.exists():
        return path
    policy = train_default_policy()
    path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(path))
    return path
