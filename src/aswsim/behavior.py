from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

from .physics import propagate_constant_velocity, propagate_acceleration, propagate_turn_rate

class BehaviorModel(Protocol):
    def __call__(self, positions: np.ndarray, velocities: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        ...


def constant_velocity(positions: np.ndarray, velocities: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple behavior: targets continue at current velocity (no acceleration)."""
    return propagate_constant_velocity(positions, velocities, dt)


def constant_acceleration(positions: np.ndarray, velocities: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Behavior: targets maintain constant acceleration."""
    return propagate_acceleration(positions, velocities, dt)


def turn_rate_behavior(positions: np.ndarray, velocities: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Behavior: targets turn at a constant rate while maintaining speed."""
    return propagate_turn_rate(positions, velocities, dt)


def random_walk_behavior(positions: np.ndarray, velocities: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Behavior: targets change direction randomly while maintaining speed."""
    # This is a more complex behavior that still benefits from vectorization
    # across targets, even though it can't be fully vectorized across time
    
    # Get current speeds
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    
    # Add random direction changes (small random rotations)
    # This simulates random course changes
    angles = np.random.normal(0, 0.1, size=velocities.shape[0])  # Small random angles
    
    # Create rotation matrices for each target
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    # Apply 2D rotation to velocities (ignoring z component for simplicity)
    new_vx = velocities[:, 0] * cos_angles - velocities[:, 1] * sin_angles
    new_vy = velocities[:, 0] * sin_angles + velocities[:, 1] * cos_angles
    
    new_velocities = np.column_stack([new_vx, new_vy, velocities[:, 2]])
    
    # Normalize to maintain speed
    new_speeds = np.linalg.norm(new_velocities, axis=1, keepdims=True)
    new_velocities = new_velocities * (speeds / new_speeds)
    
    # Propagate positions
    new_positions = positions + new_velocities * dt
    
    return new_positions, new_velocities


