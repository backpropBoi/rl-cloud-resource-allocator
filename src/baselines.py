# src/baselines.py
"""Rule-based baseline policies for comparison with the RL agent."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class FixedWorkersPolicy(BasePolicy):
    """Keep worker count at a fixed fraction of max capacity."""

    name = "fixed"

    def __init__(self, fraction: float = 0.5):
        self.fraction = float(np.clip(fraction, 0.0, 1.0))

    def predict(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        return np.array([self.fraction], dtype=np.float32)


class ThresholdPolicy(BasePolicy):
    """Scale up when queue is high, scale down when queue is low."""

    name = "threshold"

    def __init__(
        self,
        high_queue_ratio: float = 0.3,
        low_queue_ratio: float = 0.05,
        scale_up_action: float = 0.9,
        scale_down_action: float = 0.2,
    ):
        self.high_queue_ratio = high_queue_ratio
        self.low_queue_ratio = low_queue_ratio
        self.scale_up_action = scale_up_action
        self.scale_down_action = scale_down_action
        self._current_action = 0.5

    def reset(self) -> None:
        self._current_action = 0.5

    def predict(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        queue_ratio = float(obs[0])
        if queue_ratio >= self.high_queue_ratio:
            self._current_action = self.scale_up_action
        elif queue_ratio <= self.low_queue_ratio:
            self._current_action = self.scale_down_action
        return np.array([self._current_action], dtype=np.float32)


class ProportionalPolicy(BasePolicy):
    """Scale workers proportionally to current queue occupancy.

    Uses queue_len ratio (obs[0]) directly as the action, so worker count
    tracks queue pressure linearly. Stronger baseline than threshold because
    it responds to the magnitude of queue buildup, not just high/low.
    """

    name = "proportional"

    def predict(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        queue_ratio = float(np.clip(obs[0], 0.0, 1.0))
        return np.array([queue_ratio], dtype=np.float32)


class RandomPolicy(BasePolicy):
    """Uniform random action in [0, 1]."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        return np.array([self.rng.random()], dtype=np.float32)


def get_baseline_policies(seed: int | None = None) -> list[BasePolicy]:
    return [
        FixedWorkersPolicy(fraction=0.5),
        ThresholdPolicy(),
        ProportionalPolicy(),
        RandomPolicy(seed=seed),
    ]
