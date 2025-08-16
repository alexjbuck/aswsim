from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .behavior import constant_velocity
from .simulation import InitialDistributions, simulate, bivariate_normal_position_uniform_depth
from .distributions import uniform_speed, rayleigh_speed, beta_speed, bivariate_normal_velocity, independent_normal_velocity


def make_heatmap_animation(times: np.ndarray, trajectories: np.ndarray, grid_size: int = 100) -> go.Figure:
    # Heatmap over x-y; ignore depth. Compute histogram2d per time.
    x_all = trajectories[:, :, 0]
    y_all = trajectories[:, :, 1]
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, grid_size + 1)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, grid_size + 1)

    frames = []
    z0, _, _ = np.histogram2d(x_all[0], y_all[0], bins=[x_edges, y_edges], density=True)
    heatmap = go.Heatmap(z=z0.T, x=x_edges, y=y_edges, colorscale="Viridis", zsmooth="best")

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(heatmap)

    for i in range(1, len(times)):
        z, _, _ = np.histogram2d(x_all[i], y_all[i], bins=[x_edges, y_edges], density=True)
        frames.append(
            go.Frame(
                data=[go.Heatmap(z=z.T, x=x_edges, y=y_edges, colorscale="Viridis", zsmooth="best")],
                name=str(i),
            )
        )

    fig.frames = frames
    fig.update_layout(
        title="Target Distribution Over Time (x-y heatmap)",
        xaxis_title="x",
        yaxis_title="y",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 100, "redraw": True}, "transition": {"duration": 0}}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": [
                    {
                        "label": f"t={times[i]:.2f}",
                        "method": "animate",
                        "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                    }
                    for i in range(len(times))
                ],
            }
        ],
    )
    return fig


def run_demo() -> None:
    # Example with uniform speed distribution
    init = bivariate_normal_position_uniform_depth(
        pos_mean=np.array([0.0, 0.0]),
        pos_cov=np.array([[500.0, 0.0], [0.0, 500.0]]),
        depth_min=-100.0,
        depth_max=-10.0,
        velocity_dist=uniform_speed(2.0, 8.0, vz=0.0),
        pos_bounds=((-2000.0, 2000.0), (-2000.0, 2000.0)),
    )
    times, traj = simulate(
        n_targets=2000,
        total_time=200.0,
        dt=1.0,
        init=init,
        behavior=constant_velocity,
        seed=42,
    )
    fig = make_heatmap_animation(times, traj, grid_size=60)
    fig.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="ASW target distribution simulation and visualization")
    parser.add_argument("--n-targets", type=int, default=2000)
    parser.add_argument("--total-time", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--depth-min", type=float, default=-100.0)
    parser.add_argument("--depth-max", type=float, default=-10.0)
    parser.add_argument("--grid-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    # Position distribution parameters
    parser.add_argument("--pos-mean-x", type=float, default=0.0, help="Mean x position")
    parser.add_argument("--pos-mean-y", type=float, default=0.0, help="Mean y position")
    parser.add_argument("--pos-std-x", type=float, default=22.36, help="Standard deviation of x position (sqrt(500))")
    parser.add_argument("--pos-std-y", type=float, default=22.36, help="Standard deviation of y position (sqrt(500))")
    parser.add_argument("--pos-corr", type=float, default=0.0, help="Correlation between x and y positions")
    # Bounds for initial position distribution (xy) - optional truncation
    parser.add_argument("--pos-min-x", type=float, default=None, help="Minimum x position (optional truncation)")
    parser.add_argument("--pos-max-x", type=float, default=None, help="Maximum x position (optional truncation)")
    parser.add_argument("--pos-min-y", type=float, default=None, help="Minimum y position (optional truncation)")
    parser.add_argument("--pos-max-y", type=float, default=None, help="Maximum y position (optional truncation)")
    # Velocity distribution type and parameters
    parser.add_argument("--velocity-type", choices=["uniform", "rayleigh", "beta", "bivariate-normal", "independent-normal"], 
                       default="uniform", help="Type of velocity distribution")
    parser.add_argument("--min-speed", type=float, default=2.0, help="Min speed for uniform/beta distributions")
    parser.add_argument("--max-speed", type=float, default=8.0, help="Max speed for uniform/beta distributions")
    parser.add_argument("--rayleigh-scale", type=float, default=3.0, help="Scale for Rayleigh distribution")
    parser.add_argument("--beta-a", type=float, default=2.0, help="Alpha parameter for Beta distribution")
    parser.add_argument("--beta-b", type=float, default=5.0, help="Beta parameter for Beta distribution")
    parser.add_argument("--vx-mean", type=float, default=1.0, help="Mean vx for normal distributions")
    parser.add_argument("--vx-std", type=float, default=2.0, help="Std vx for normal distributions")
    parser.add_argument("--vy-mean", type=float, default=0.2, help="Mean vy for normal distributions")
    parser.add_argument("--vy-std", type=float, default=1.0, help="Std vy for normal distributions")
    parser.add_argument("--vz", type=float, default=0.0)
    args = parser.parse_args()

    # Create velocity distribution based on type
    if args.velocity_type == "uniform":
        velocity_dist = uniform_speed(args.min_speed, args.max_speed, args.vz)
    elif args.velocity_type == "rayleigh":
        velocity_dist = rayleigh_speed(args.rayleigh_scale, args.vz)
    elif args.velocity_type == "beta":
        velocity_dist = beta_speed(args.beta_a, args.beta_b, args.min_speed, args.max_speed, args.vz)
    elif args.velocity_type == "bivariate-normal":
        velocity_dist = bivariate_normal_velocity(
            np.array([args.vx_mean, args.vy_mean]),
            np.array([[args.vx_std**2, 0], [0, args.vy_std**2]]),
            vz=args.vz
        )
    elif args.velocity_type == "independent-normal":
        velocity_dist = independent_normal_velocity(
            args.vx_mean, args.vx_std, args.vy_mean, args.vy_std, args.vz
        )
    else:
        raise ValueError(f"Unknown velocity type: {args.velocity_type}")

    # Build position covariance matrix
    pos_cov = np.array([
        [args.pos_std_x**2, args.pos_corr * args.pos_std_x * args.pos_std_y],
        [args.pos_corr * args.pos_std_x * args.pos_std_y, args.pos_std_y**2]
    ])
    
    # Build position bounds (only if specified)
    pos_bounds = None
    if args.pos_min_x is not None and args.pos_max_x is not None and args.pos_min_y is not None and args.pos_max_y is not None:
        pos_bounds = ((args.pos_min_x, args.pos_max_x), (args.pos_min_y, args.pos_max_y))
    
    init = bivariate_normal_position_uniform_depth(
        pos_mean=np.array([args.pos_mean_x, args.pos_mean_y]),
        pos_cov=pos_cov,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        velocity_dist=velocity_dist,
        pos_bounds=pos_bounds,
    )
    times, traj = simulate(
        n_targets=args.n_targets,
        total_time=args.total_time,
        dt=args.dt,
        init=init,
        behavior=constant_velocity,
        seed=args.seed,
    )
    fig = make_heatmap_animation(times, traj, grid_size=args.grid_size)
    fig.show()


