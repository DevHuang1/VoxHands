from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

from voxhands.models import Action, Plan, new_id

from .base import Policy


class MLPPolicy(Policy):

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: list[int] | None = None,
        output_dim: int = 8,
        activation: str = "relu",
        learning_rate: float = 1e-3,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [64, 32]
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        self._lr = learning_rate
        self._activation_name = activation

        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        self._last_zs: list[np.ndarray] = []
        self._last_as: list[np.ndarray] = []

        dims = [input_dim] + hidden_dims + [output_dim]
        rng = np.random.default_rng(42)
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            w = rng.standard_normal((dims[i], dims[i + 1])).astype(np.float64) * scale
            b = np.zeros(dims[i + 1], dtype=np.float64)
            self._weights.append(w)
            self._biases.append(b)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self._activation_name == "relu":
            return np.maximum(0.0, z)
        if self._activation_name == "tanh":
            return np.tanh(z)
        if self._activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
        return z

    def _activate_deriv(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self._activation_name == "relu":
            return (z > 0).astype(np.float64)
        if self._activation_name == "tanh":
            return 1.0 - a ** 2
        if self._activation_name == "sigmoid":
            return a * (1.0 - a)
        return np.ones_like(z)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        self._last_zs = []
        self._last_as = []
        a = x
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            z = a @ w + b
            self._last_zs.append(z)
            if i < len(self._weights) - 1:
                a = self._activate(z)
            else:
                a = z
            self._last_as.append(a)
        return a

    def predict(self, state: dict[str, Any]) -> np.ndarray:
        x = np.zeros(self._input_dim, dtype=np.float64)
        for i, key in enumerate(sorted(state.keys())[: self._input_dim]):
            val = state[key]
            if isinstance(val, (int, float)):
                x[i] = float(val)
            elif isinstance(val, bool):
                x[i] = 1.0 if val else 0.0
            elif isinstance(val, str):
                x[i] = float(hash(val) % 1000) / 1000.0
        return self.forward(x)

    def backward(self, x: np.ndarray, target: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        target = np.asarray(target, dtype=np.float64).ravel()

        pred = self.forward(x)
        diff = pred - target
        loss = float(np.sum(diff ** 2) / 2.0)

        grads_w = [None] * len(self._weights)
        grads_b = [None] * len(self._biases)
        delta = diff

        for i in range(len(self._weights) - 1, -1, -1):
            a_prev = self._last_as[i - 1] if i > 0 else x
            grads_w[i] = np.outer(a_prev, delta)
            grads_b[i] = delta.copy()
            if i > 0:
                delta = (delta @ self._weights[i].T) * self._activate_deriv(self._last_zs[i - 1], self._last_as[i - 1])

        for i in range(len(self._weights)):
            self._weights[i] -= self._lr * grads_w[i]
            self._biases[i] -= self._lr * grads_b[i]

        return loss

    def train_step(self, states: list[dict[str, Any]], targets: list[np.ndarray]) -> float:
        total_loss = 0.0
        for state, target in zip(states, targets):
            x = np.zeros(self._input_dim, dtype=np.float64)
            for i, key in enumerate(sorted(state.keys())[: self._input_dim]):
                val = state[key]
                if isinstance(val, (int, float)):
                    x[i] = float(val)
                elif isinstance(val, bool):
                    x[i] = 1.0 if val else 0.0
                elif isinstance(val, str):
                    x[i] = float(hash(val) % 1000) / 1000.0
            total_loss += self.backward(x, target)
        return total_loss

    def train(self, demos: list[dict[str, Any]]) -> None:
        for demo in demos:
            state = demo.get("state", {})
            target = np.asarray(demo.get("target", np.zeros(self._output_dim)), dtype=np.float64)
            self.train_step([state], [target])

    def params(self) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for w, b in zip(self._weights, self._biases):
            result.append(w)
            result.append(b)
        return result

    def save(self, path: str) -> None:
        data = {
            "input_dim": self._input_dim,
            "hidden_dims": self._hidden_dims,
            "output_dim": self._output_dim,
            "activation": self._activation_name,
            "lr": self._lr,
            "weights": [w.tolist() for w in self._weights],
            "biases": [b.tolist() for b in self._biases],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self._input_dim = data["input_dim"]
        self._hidden_dims = data["hidden_dims"]
        self._output_dim = data["output_dim"]
        self._activation_name = data["activation"]
        self._lr = data["lr"]
        self._weights = [np.array(w, dtype=np.float64) for w in data["weights"]]
        self._biases = [np.array(b, dtype=np.float64) for b in data["biases"]]

    def __call__(self, state: dict[str, Any]) -> Action | Plan:
        out = self.predict(state)
        idx = int(np.argmax(out))
        arms = ["left", "right"]
        arm = arms[idx % len(arms)]
        action = Action(
            id=new_id("act"),
            arm=arm,
            object_id="blue_plate",
            object_label="Blue plate",
            target_id="left_place",
            target_label="Left place",
            duration_ms=1400,
        )
        return action
