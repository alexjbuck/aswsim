from __future__ import annotations

import numpy as np


def propagate_constant_velocity(positions: np.ndarray, velocities: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Point-kinematics propagation under constant velocity."""
    return positions + velocities * dt, velocities


