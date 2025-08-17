# ASW Target Distribution Simulator

A flexible simulation framework for modeling the progression of submarine target distributions over time. The simulator supports various initial distribution types and allows easy experimentation with different target behaviors.

## Features

- **Flexible Initial Distributions**: Support for various position and velocity distributions
- **Modular Behavior Models**: Easy to swap target behavior models
- **Interactive Visualization**: Real-time heatmap visualization with time controls
- **Extensible Architecture**: Easy to add new distribution types and behaviors

## Installation

```bash
# Install dependencies
uv sync

# Install in development mode
uv sync --package
```

## Building Standalone Executable

### Local Build

To create a standalone executable for your current platform:

```bash
# Install build dependencies
uv sync --group dev

# Build console version (with console window)
python build_standalone.py --console

# Create distribution package
python create_distribution.py
```

The executable will be created in the `dist/` directory:
- `dist/aswsim-cli` - Console version (for command line usage)

### Cross-Platform Builds via GitHub Actions

For cross-platform builds (Windows, macOS, Linux), the project uses GitHub Actions:

1. **Push to main branch**: Triggers build for all platforms
2. **Create a tag**: Creates a GitHub release with all platform executables
3. **Manual trigger**: Use "workflow_dispatch" in GitHub Actions

**Build Artifacts:**
- Individual platform executables
- Cross-platform distribution package with all executables
- Complete documentation and guides

**GitHub Release Contents:**
- `aswsim_cross_platform_YYYYMMDD_HHMMSS.zip` - All platforms + docs
- Individual executables for each platform

### Distribution Package

The `create_distribution.py` script creates a complete distribution package:
- Standalone executable (no Python required)
- README documentation
- Quick start guide
- Platform-specific instructions
- Zipped for easy sharing

**Package Contents:**
- `aswsim` - Standalone executable
- `README.md` - Full documentation
- `QUICK_START.txt` - Quick usage guide
- `PLATFORM_INSTRUCTIONS.txt` - Platform-specific setup

**Note**: For cross-platform distribution, use GitHub Actions. Local builds are platform-specific.

## Quick Start

### Command Line Interface

Run a basic simulation with default parameters:
```bash
uv run aswsim
```

Run with custom distributions:
```bash
# Custom position distribution (mean, std, correlation)
uv run aswsim --pos-mean-x 100 --pos-mean-y 200 --pos-std-x 10 --pos-std-y 20 --pos-corr 0.5

# Rayleigh speed distribution
uv run aswsim --velocity-type rayleigh --rayleigh-scale 5.0

# Beta speed distribution (bounded, skewed)
uv run aswsim --velocity-type beta --beta-a 2 --beta-b 5 --min-speed 1 --max-speed 10

# Bivariate normal velocity (cartesian coordinates)
uv run aswsim --velocity-type bivariate-normal --vx-mean 1 --vx-std 2 --vy-mean 0.2 --vy-std 1

# With position bounds (truncation)
uv run aswsim --pos-min-x -50 --pos-max-x 50 --pos-min-y -50 --pos-max-y 50

# Large scale simulation
uv run aswsim --n-targets 100000
```
```

### Programmatic Usage

```python
from aswsim import (
    simulate, bivariate_normal_position_uniform_depth,
    uniform_speed, rayleigh_speed, beta_speed
)

# Create initial distribution with uniform speed
init = bivariate_normal_position_uniform_depth(
    pos_mean=np.array([0.0, 0.0]),
    pos_cov=np.array([[500.0, 0.0], [0.0, 500.0]]),
    depth_min=-100.0,
    depth_max=-10.0,
    velocity_dist=uniform_speed(2.0, 8.0, vz=0.0),
    pos_bounds=((-2000.0, 2000.0), (-2000.0, 2000.0)),
)

# Run simulation
times, trajectories = simulate(
    n_targets=2000,
    total_time=200.0,
    dt=1.0,
    init=init,
    seed=42
)
```

## Velocity Distribution Types

### Polar Coordinates (Speed + Direction)

1. **Uniform Speed**: `uniform_speed(min_speed, max_speed)`
   - Speed uniformly distributed between min and max
   - Direction uniformly distributed 0-360°

2. **Rayleigh Speed**: `rayleigh_speed(scale)`
   - Speed follows Rayleigh distribution (common in radar/sonar)
   - Direction uniformly distributed 0-360°

3. **Beta Speed**: `beta_speed(a, b, min_speed, max_speed)`
   - Speed follows Beta distribution scaled to [min_speed, max_speed]
   - Useful for bounded, skewed distributions with minimum speed
   - Direction uniformly distributed 0-360°

### Cartesian Coordinates

4. **Bivariate Normal**: `bivariate_normal_velocity(mean, cov)`
   - Velocity components follow bivariate normal distribution
   - Supports correlation between vx and vy

5. **Independent Normal**: `independent_normal_velocity(vx_mean, vx_std, vy_mean, vy_std)`
   - Independent normal distributions for vx and vy components

## Performance Optimizations

The simulation is highly optimized for performance:

- **Vectorized Operations**: Uses NumPy broadcasting to compute all time steps simultaneously for constant velocity behavior
- **Ultra-Fast Constant Velocity**: Specialized path that computes 20M+ target-time-steps per second
- **Memory Efficient**: Pre-allocated arrays and minimal memory copies

Example performance:
- 1,000,000 targets × 500 time steps = 500M operations in ~26 seconds
- Over 20M target-time-steps per second on modern hardware
- Scales linearly with target count

### Performance Benchmarking

Run the performance benchmark:
```bash
uv run python examples/benchmark_performance.py
```

Or try the speedup demo:
```bash
uv run python examples/speedup_demo.py
```

## Examples

See `examples/distribution_comparison.py` for a comprehensive comparison of different velocity distributions and their impact on target spread.

## Architecture

- **`behavior.py`**: Target behavior models (constant velocity, etc.)
- **`distributions.py`**: Flexible distribution system
- **`simulation.py`**: Core simulation engine
- **`visualize.py`**: Interactive visualization tools
- **`physics.py`**: Physics propagation models

## Adding New Distributions

To add a new distribution type:

1. Create a new class inheriting from `Distribution` in `distributions.py`
2. Implement the `sample()` method
3. Add convenience constructors if needed
4. Update the CLI in `visualize.py` if desired

Example:
```python
@dataclass
class Weibull(Distribution):
    """Weibull distribution."""
    shape: float
    scale: float
    
    def sample(self, rng: Generator, size: int) -> np.ndarray:
        return rng.weibull(self.shape, size) * self.scale
```
