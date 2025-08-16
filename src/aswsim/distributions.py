from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Tuple

import numpy as np
from numpy.random import Generator


class Distribution(ABC):
    """Abstract base class for distributions."""
    
    @abstractmethod
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        """Sample from the distribution."""
        pass


@dataclass
class BivariateNormal(Distribution):
    """Bivariate normal distribution."""
    mean: np.ndarray
    cov: np.ndarray
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] | None = None
    
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        if self.bounds is None:
            return rng.multivariate_normal(mean=self.mean, cov=self.cov, size=size)
        
        # Rejection sampling with bounds
        (min_x, max_x), (min_y, max_y) = self.bounds
        remaining = size
        samples = []
        attempts = 0
        max_attempts = 10
        
        while remaining > 0 and attempts < max_attempts:
            batch = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=remaining)
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
            batch = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=remaining)
            batch[:, 0] = np.clip(batch[:, 0], min_x, max_x)
            batch[:, 1] = np.clip(batch[:, 1], min_y, max_y)
            samples.append(batch)
        
        return np.vstack(samples) if samples else np.empty((0, 2))


@dataclass
class Uniform(Distribution):
    """Uniform distribution."""
    low: float
    high: float
    
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=size)


@dataclass
class Rayleigh(Distribution):
    """Rayleigh distribution."""
    scale: float
    
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        return rng.rayleigh(self.scale, size=size)


@dataclass
class Beta(Distribution):
    """Beta distribution."""
    a: float
    b: float
    min_val: float = 0.0  # Minimum value
    max_val: float = 1.0  # Maximum value
    
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        # Sample from Beta(0,1) then scale to [min_val, max_val]
        beta_samples = rng.beta(self.a, self.b, size=size)
        return beta_samples * (self.max_val - self.min_val) + self.min_val


@dataclass
class Exponential(Distribution):
    """Exponential distribution."""
    scale: float
    
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        return rng.exponential(self.scale, size=size)


@dataclass
class Gamma(Distribution):
    """Gamma distribution."""
    shape: float
    scale: float
    
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        return rng.gamma(self.shape, self.scale, size=size)


class VelocityDistribution(ABC):
    """Abstract base class for velocity distributions."""
    
    @abstractmethod
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        """Sample velocities in cartesian coordinates (vx, vy, vz)."""
        pass


@dataclass
class CartesianVelocity(VelocityDistribution):
    """Velocity distribution in cartesian coordinates."""
    vx_dist: Distribution
    vy_dist: Distribution
    vz: float = 0.0
    
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        vx = self.vx_dist.sample(rng, size)
        vy = self.vy_dist.sample(rng, size)
        vz = np.full(size, self.vz)
        
        # Handle case where distributions return 2D arrays (take first column)
        if vx.ndim > 1:
            vx = vx[:, 0]
        if vy.ndim > 1:
            vy = vy[:, 0]
            
        return np.column_stack([vx, vy, vz])


@dataclass
class PolarVelocity(VelocityDistribution):
    """Velocity distribution in polar coordinates (speed, direction)."""
    speed_dist: Distribution
    direction_dist: Distribution
    vz: float = 0.0
    
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        speeds = self.speed_dist.sample(rng, size)
        directions = self.direction_dist.sample(rng, size)
        
        # Convert to cartesian
        vx = speeds * np.cos(directions)
        vy = speeds * np.sin(directions)
        vz = np.full(size, self.vz)
        
        return np.column_stack([vx, vy, vz])


# Convenience constructors for common distributions
def uniform_speed(min_speed: float, max_speed: float, vz: float = 0.0) -> PolarVelocity:
    """Create uniform speed distribution with uniform direction."""
    return PolarVelocity(
        speed_dist=Uniform(min_speed, max_speed),
        direction_dist=Uniform(0, 2 * np.pi),
        vz=vz
    )


def rayleigh_speed(scale: float, vz: float = 0.0) -> PolarVelocity:
    """Create Rayleigh speed distribution with uniform direction."""
    return PolarVelocity(
        speed_dist=Rayleigh(scale),
        direction_dist=Uniform(0, 2 * np.pi),
        vz=vz
    )


def beta_speed(a: float, b: float, min_speed: float, max_speed: float, vz: float = 0.0) -> PolarVelocity:
    """Create Beta speed distribution with uniform direction."""
    return PolarVelocity(
        speed_dist=Beta(a, b, min_speed, max_speed),
        direction_dist=Uniform(0, 2 * np.pi),
        vz=vz
    )


def bivariate_normal_velocity(
    mean: np.ndarray,
    cov: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
    vz: float = 0.0
) -> CartesianVelocity:
    """Create bivariate normal velocity distribution."""
    return CartesianVelocity(
        vx_dist=BivariateNormal(mean, cov, bounds),
        vy_dist=BivariateNormal(mean, cov, bounds),
        vz=vz
    )


def independent_normal_velocity(
    vx_mean: float, vx_std: float,
    vy_mean: float, vy_std: float,
    vz: float = 0.0
) -> CartesianVelocity:
    """Create independent normal velocity distributions for x and y."""
    return CartesianVelocity(
        vx_dist=BivariateNormal(np.array([vx_mean]), np.array([[vx_std**2]])),
        vy_dist=BivariateNormal(np.array([vy_mean]), np.array([[vy_std**2]])),
        vz=vz
    )



