from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .behavior import BehaviorModel, constant_velocity
from .distributions import Distribution, VelocityDistribution


Bounds = Tuple[Tuple[float, float], Tuple[float, float]]  # ((min_x, max_x), (min_y, max_y))


@dataclass
class InitialDistributions:
    # Position distribution (x, y) and depth distribution
    position_dist: Distribution  # Should sample 2D positions (x, y)
    depth_dist: Distribution     # Should sample 1D depths (z)
    
    # Velocity distribution
    velocity_dist: VelocityDistribution
    
    # Optional bounds for position truncation
    pos_bounds_xy: Bounds | None = None


def _rejection_sample_bivariate_normal(
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray,
    n: int,
    bounds: Bounds | None,
    max_attempts: int = 10,
) -> np.ndarray:
    if bounds is None:
        return rng.multivariate_normal(mean=mean, cov=cov, size=n)

    (min_x, max_x), (min_y, max_y) = bounds
    remaining = n
    samples = []
    attempts = 0
    while remaining > 0 and attempts < max_attempts:
        batch = rng.multivariate_normal(mean=mean, cov=cov, size=remaining)
        mask = (
            (batch[:, 0] >= min_x)
            & (batch[:, 0] <= max_x)
            & (batch[:, 1] >= min_y)
            & (batch[:, 1] <= max_y)
        )
        if np.any(mask):
            samples.append(batch[mask])
            remaining -= int(np.sum(mask))
        attempts += 1

    if remaining > 0:
        # Fallback: sample remaining then clip
        batch = rng.multivariate_normal(mean=mean, cov=cov, size=remaining)
        batch[:, 0] = np.clip(batch[:, 0], min_x, max_x)
        batch[:, 1] = np.clip(batch[:, 1], min_y, max_y)
        samples.append(batch)

    return np.vstack(samples) if samples else np.empty((0, 2))


def sample_initial_state(rng: np.random.Generator, n: int, init: InitialDistributions) -> tuple[np.ndarray, np.ndarray]:
    # Sample positions
    xy = init.position_dist.sample(rng, n)
    z = init.depth_dist.sample(rng, n)[:, None]
    positions = np.hstack([xy, z])
    
    # Sample velocities
    velocities = init.velocity_dist.sample(rng, n)
    
    return positions, velocities


def simulate(
    n_targets: int,
    total_time: float,
    dt: float,
    init: InitialDistributions,
    behavior: BehaviorModel = constant_velocity,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the simulation.

    Returns:
        times: (T,) array of times
        trajectories: (T, N, 6) array: columns [x, y, z, vx, vy, vz]
    """
    rng = np.random.default_rng(seed)
    positions, velocities = sample_initial_state(rng, n_targets, init)

    num_steps = int(np.floor(total_time / dt)) + 1
    times = np.linspace(0.0, dt * (num_steps - 1), num_steps)
    trajectories = np.zeros((num_steps, n_targets, 6), dtype=float)
    trajectories[0, :, 0:3] = positions
    trajectories[0, :, 3:6] = velocities

    for t_idx in range(1, num_steps):
        positions, velocities = behavior(positions, velocities, dt)
        trajectories[t_idx, :, 0:3] = positions
        trajectories[t_idx, :, 3:6] = velocities

    return times, trajectories


# Convenience constructors for common initial distributions
def bivariate_normal_position_uniform_depth(
    pos_mean: np.ndarray,
    pos_cov: np.ndarray,
    depth_min: float,
    depth_max: float,
    velocity_dist: VelocityDistribution,
    pos_bounds: Bounds | None = None,
) -> InitialDistributions:
    """Create initial distribution with bivariate normal position and uniform depth."""
    from .distributions import BivariateNormal, Uniform
    
    return InitialDistributions(
        position_dist=BivariateNormal(pos_mean, pos_cov, pos_bounds),
        depth_dist=Uniform(depth_min, depth_max),
        velocity_dist=velocity_dist,
        pos_bounds_xy=pos_bounds,
    )


def uniform_position_uniform_depth(
    pos_min: np.ndarray,
    pos_max: np.ndarray,
    depth_min: float,
    depth_max: float,
    velocity_dist: VelocityDistribution,
) -> InitialDistributions:
    """Create initial distribution with uniform position and uniform depth."""
    from .distributions import Uniform
    
    # For uniform position, we'll use independent uniform distributions
    class Uniform2D(Distribution):
        def __init__(self, min_xy: np.ndarray, max_xy: np.ndarray):
            self.min_xy = min_xy
            self.max_xy = max_xy
            
        def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
            x = rng.uniform(self.min_xy[0], self.max_xy[0], size)
            y = rng.uniform(self.min_xy[1], self.max_xy[1], size)
            return np.column_stack([x, y])
    
    return InitialDistributions(
        position_dist=Uniform2D(pos_min, pos_max),
        depth_dist=Uniform(depth_min, depth_max),
        velocity_dist=velocity_dist,
    )


