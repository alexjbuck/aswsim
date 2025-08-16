from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

from .physics import propagate_constant_velocity

class BehaviorModel(Protocol):
    def __call__(self, positions: np.ndarray, velocities: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        ...


def constant_velocity(positions: np.ndarray, velocities: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple behavior: targets continue at current velocity (no acceleration)."""
    return propagate_constant_velocity(positions, velocities, dt)


