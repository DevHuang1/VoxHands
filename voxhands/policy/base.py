from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from voxhands.models import Action, Plan


class Policy(ABC):

    @abstractmethod
    def __call__(self, state: dict[str, Any]) -> Action | Plan:
        ...

    def train(self, demos: list[dict[str, Any]]) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def params(self) -> list[Any]:
        return []
