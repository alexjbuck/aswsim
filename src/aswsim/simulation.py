from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np

from .behavior import BehaviorModel, constant_velocity
from .distributions import Distribution, VelocityDistribution
from .sensors import SearchPattern, DetectionEngine


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





def _simulate_constant_velocity_vectorized(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Ultra-fast simulation for constant velocity using vectorized operations."""
    n_targets = initial_positions.shape[0]
    n_steps = len(times)
    
    # Pre-allocate trajectories array
    trajectories = np.zeros((n_steps, n_targets, 6), dtype=float)
    
    # Vectorized computation: positions = initial_positions + velocities * time
    # This computes all time steps at once using broadcasting
    time_deltas = times[:, np.newaxis, np.newaxis]  # Shape: (n_steps, 1, 1)
    
    # Compute all positions at once
    trajectories[:, :, 0:3] = initial_positions[np.newaxis, :, :] + initial_velocities[np.newaxis, :, :] * time_deltas
    
    # Velocities remain constant
    trajectories[:, :, 3:6] = initial_velocities[np.newaxis, :, :]
    
    return trajectories


def track_detection_statistics(
    trajectories: np.ndarray,
    times: np.ndarray,
    search_pattern: SearchPattern,
    dt: float,
) -> Dict[str, Any]:
    """Track detection statistics over the entire simulation.
    
    Args:
        trajectories: (T, N, 6) array of target trajectories
        times: (T,) array of simulation times
        search_pattern: Search pattern configuration
        dt: Time step size
        
    Returns:
        Dictionary with detection statistics over time
    """
    detection_engine = DetectionEngine(search_pattern)
    n_steps = len(times)
    
    # Initialize statistics arrays
    pattern_stats = {
        'times': times,
        'total_detections': np.zeros(n_steps),
        'total_detection_rates': np.zeros(n_steps),
        'average_coverage': np.zeros(n_steps),
        'active_sensors': np.zeros(n_steps, dtype=int),
        'cumulative_detections': np.zeros(n_steps),
    }
    
    # Track individual sensor statistics
    sensor_stats = {}
    for sensor in search_pattern.sensors:
        sensor_stats[sensor.name] = {
            'detections': np.zeros(n_steps),
            'detection_rates': np.zeros(n_steps),
            'coverage': np.zeros(n_steps),
            'active': np.zeros(n_steps, dtype=bool),
            'cumulative_detections': np.zeros(n_steps),
        }
    
    # Evaluate detection at each time step
    for t_idx, current_time in enumerate(times):
        # Get target positions at this time
        target_positions = trajectories[t_idx, :, 0:2]  # x, y positions only
        
        # Evaluate pattern at this time
        pattern_result = detection_engine.evaluate_pattern_at_time(
            target_positions, current_time, dt
        )
        
        # Store pattern-wide statistics
        pattern_stats['total_detections'][t_idx] = pattern_result['total_detections']
        pattern_stats['total_detection_rates'][t_idx] = pattern_result['total_detection_rate']
        pattern_stats['average_coverage'][t_idx] = pattern_result['average_coverage']
        pattern_stats['active_sensors'][t_idx] = pattern_result['active_sensors']
        
        # Calculate cumulative detections
        if t_idx > 0:
            pattern_stats['cumulative_detections'][t_idx] = (
                pattern_stats['cumulative_detections'][t_idx - 1] + 
                pattern_result['total_detections']
            )
        else:
            pattern_stats['cumulative_detections'][t_idx] = pattern_result['total_detections']
        
        # Store individual sensor statistics
        for sensor_name, sensor_result in pattern_result['sensor_results'].items():
            sensor_stats[sensor_name]['detections'][t_idx] = sensor_result['detections']
            sensor_stats[sensor_name]['detection_rates'][t_idx] = sensor_result['detection_rate']
            sensor_stats[sensor_name]['coverage'][t_idx] = sensor_result['coverage']
            sensor_stats[sensor_name]['active'][t_idx] = sensor_result['active']
            
            # Calculate cumulative detections for this sensor
            if t_idx > 0:
                sensor_stats[sensor_name]['cumulative_detections'][t_idx] = (
                    sensor_stats[sensor_name]['cumulative_detections'][t_idx - 1] + 
                    sensor_result['detections']
                )
            else:
                sensor_stats[sensor_name]['cumulative_detections'][t_idx] = sensor_result['detections']
    
    return {
        'pattern': pattern_stats,
        'sensors': sensor_stats,
        'search_pattern': search_pattern
    }


def _is_constant_velocity_behavior(behavior: BehaviorModel) -> bool:
    """Check if the behavior model is constant velocity."""
    # This is a simple heuristic - in practice, we could make this more robust
    # by checking the function name or adding a flag to BehaviorModel
    return behavior.__name__ == 'constant_velocity' if hasattr(behavior, '__name__') else False


def simulate(
    n_targets: int,
    total_time: float,
    dt: float,
    init: InitialDistributions,
    behavior: BehaviorModel = constant_velocity,
    seed: int | None = None,
    search_pattern: Optional[SearchPattern] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """Run the simulation with optimized vectorized operations.
    
    Args:
        n_targets: Number of targets to simulate
        total_time: Total simulation time
        dt: Time step size
        init: Initial distributions
        behavior: Behavior model for target motion
        seed: Random seed for reproducibility
        search_pattern: Optional search pattern for detection tracking
    
    Returns:
        times: (T,) array of times
        trajectories: (T, N, 6) array: columns [x, y, z, vx, vy, vz]
        detection_stats: Optional dictionary with detection statistics over time
    """
    rng = np.random.default_rng(seed)
    positions, velocities = sample_initial_state(rng, n_targets, init)

    num_steps = int(np.floor(total_time / dt)) + 1
    times = np.linspace(0.0, dt * (num_steps - 1), num_steps)
    
    # Fast path for constant velocity behavior
    if _is_constant_velocity_behavior(behavior):
        trajectories = _simulate_constant_velocity_vectorized(positions, velocities, times)
    else:
        # General vectorized simulation for other behaviors
        trajectories = np.zeros((num_steps, n_targets, 6), dtype=float)
        trajectories[0, :, 0:3] = positions
        trajectories[0, :, 3:6] = velocities

        # Vectorized time stepping
        for t_idx in range(1, num_steps):
            step_dt = times[t_idx] - times[t_idx - 1]
            positions, velocities = behavior(positions, velocities, step_dt)
            trajectories[t_idx, :, 0:3] = positions
            trajectories[t_idx, :, 3:6] = velocities
    
    # Track detection statistics if search pattern is provided
    detection_stats = None
    if search_pattern is not None:
        detection_stats = track_detection_statistics(trajectories, times, search_pattern, dt)
    
    return times, trajectories, detection_stats


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


