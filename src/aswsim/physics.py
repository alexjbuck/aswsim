from __future__ import annotations

import numpy as np


def propagate_constant_velocity(positions: np.ndarray, velocities: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Point-kinematics propagation under constant velocity."""
    return positions + velocities * dt, velocities


def propagate_acceleration(positions: np.ndarray, velocities: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Point-kinematics propagation under constant acceleration."""
    # For simplicity, assume constant acceleration in the direction of current velocity
    # This could be made more sophisticated with configurable acceleration vectors
    
    # Get current speeds
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    
    # Define acceleration magnitude (could be made configurable)
    acceleration_magnitude = 0.1  # m/s²
    
    # Calculate acceleration vector in direction of velocity
    # Avoid division by zero
    safe_speeds = np.where(speeds > 0, speeds, 1.0)
    acceleration = velocities / safe_speeds * acceleration_magnitude
    
    # Update velocities: v = v₀ + a*dt
    new_velocities = velocities + acceleration * dt
    
    # Update positions: x = x₀ + v₀*dt + 0.5*a*dt²
    new_positions = positions + velocities * dt + 0.5 * acceleration * dt**2
    
    return new_positions, new_velocities


def propagate_turn_rate(positions: np.ndarray, velocities: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Point-kinematics propagation with constant turn rate."""
    # Turn rate in radians per second
    turn_rate = 0.1  # rad/s (could be made configurable)
    
    # Get current speeds and directions
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    
    # Calculate current headings (angles in x-y plane)
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    
    # Update headings
    new_headings = headings + turn_rate * dt
    
    # Reconstruct velocities with new headings but same speeds
    new_vx = speeds.flatten() * np.cos(new_headings)
    new_vy = speeds.flatten() * np.sin(new_headings)
    new_vz = velocities[:, 2]  # Keep z velocity unchanged
    
    new_velocities = np.column_stack([new_vx, new_vy, new_vz])
    
    # Update positions
    new_positions = positions + new_velocities * dt
    
    return new_positions, new_velocities


