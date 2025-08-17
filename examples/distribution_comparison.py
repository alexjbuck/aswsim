#!/usr/bin/env python3
"""
Example script demonstrating different velocity distributions and their impact.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aswsim import (
    simulate, bivariate_normal_position_uniform_depth,
    uniform_speed, rayleigh_speed, beta_speed, bivariate_normal_velocity
)


def compare_velocity_distributions():
    """Compare different velocity distributions and their impact."""

    # Visualization parameters
    grid_size = 250
    
    # Common simulation parameters
    n_targets = 100_000
    total_time = 60.0 # minutes
    dt = 1.0 # minutes
    seed = 42
    std_dev = 2000.0 # yards
    
    # Common position distribution
    pos_mean = np.array([0.0, 0.0])
    pos_cov = np.array([[std_dev**2, 0.0], [0.0, std_dev**2]])
    depth_min, depth_max = -100.0, -10.0
    pos_bounds = None

    # Common speed distribution
    speed_min = 67.5  # yards/minute = 2 knots
    speed_max = 337.5 # yards/minute = 10 knots
    speed_mean = (speed_min + speed_max) / 2
    speed_mode = speed_mean
    speed_variance = speed_mean**2
    
    # Define different velocity distributions
    distributions = {
        "Uniform Speed (2-10 knots)": uniform_speed(speed_min, speed_max), # in yards per minute
        "Rayleigh Speed (mode=6 knots)": rayleigh_speed(speed_mode),
        "Beta Speed (a=2, b=5, 2-10 knots)": beta_speed(2.0, 5.0, speed_min, speed_max+2),
        "Bivariate Normal (vx=0±10 knots, vy=0±10 knots)": bivariate_normal_velocity(
            np.array([0.0, 0.0]),
            np.array([[speed_variance, 0.0], [0.0, speed_variance]])
        ),
    }
    
    # Run simulations
    results = {}
    for name, vel_dist in distributions.items():
        print(f"Running {name}...")
        init = bivariate_normal_position_uniform_depth(
            pos_mean, pos_cov, depth_min, depth_max, vel_dist, pos_bounds
        )
        times, trajectories = simulate(n_targets, total_time, dt, init, seed=seed)
        
        # Calculate speed statistics
        speeds = np.sqrt(trajectories[:, :, 3]**2 + trajectories[:, :, 4]**2)
        results[name] = {
            'times': times,
            'trajectories': trajectories,
            'speeds': speeds,
            'mean_speed': speeds.mean(),
            'std_speed': speeds.std(),
            'min_speed': speeds.min(),
            'max_speed': speeds.max(),
        }
    
    # Print statistics
    print("\nSpeed Statistics:")
    print("-" * 80)
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Mean: {data['mean_speed']*60/2000:.2f} knots")
        print(f"  Std:  {data['std_speed']*60/2000:.2f} knots")
        print(f"  Range: {data['min_speed']*60/2000:.2f} - {data['max_speed']*60/2000:.2f} knots")
        print()
    
    # Create comparison plots with global time slider
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(distributions.keys()),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # Get global bounds for consistent axes across all time steps
    all_x = np.concatenate([data['trajectories'][:, :, 0].flatten() for data in results.values()])
    all_y = np.concatenate([data['trajectories'][:, :, 1].flatten() for data in results.values()])
    crop_factor = 0.7
    x_min, x_max = float(np.min(all_x)) * crop_factor, float(np.max(all_x)) * crop_factor
    y_min, y_max = float(np.min(all_y)) * crop_factor, float(np.max(all_y)) * crop_factor
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, grid_size)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, grid_size)
    
    # Create initial heatmaps (t=0)
    frames = []
    for i, (name, data) in enumerate(results.items()):
        row = i // 2 + 1
        col = i % 2 + 1
        
        x_all = data['trajectories'][0, :, 0]
        y_all = data['trajectories'][0, :, 1]
        z, _, _ = np.histogram2d(x_all, y_all, bins=[x_edges, y_edges], density=True)
        
        heatmap = go.Heatmap(
            z=z.T, x=x_edges, y=y_edges,
            colorscale="Viridis", zsmooth="best",
            showscale=(i == 0)  # Only show colorbar for first plot
        )
        
        fig.add_trace(heatmap, row=row, col=col)
    
    # Create frames for all time steps
    for t_idx in range(len(times)):
        frame_data = []
        for i, (name, data) in enumerate(results.items()):
            x_all = data['trajectories'][t_idx, :, 0]
            y_all = data['trajectories'][t_idx, :, 1]
            z, _, _ = np.histogram2d(x_all, y_all, bins=[x_edges, y_edges], density=True)
            
            heatmap = go.Heatmap(
                z=z.T, x=x_edges, y=y_edges,
                colorscale="Viridis", zsmooth="best",
                showscale=(i == 0)
            )
            frame_data.append(heatmap)
        
        frames.append(go.Frame(data=frame_data, name=str(t_idx)))
    
    fig.frames = frames
    
    fig.update_layout(
        title="Target Distribution Comparison",
        height=800,
        width=1000,
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
                        "label": f"t={times[i]:.1f}s",
                        "method": "animate",
                        "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                    }
                    for i in range(len(times))
                ],
            }
        ],
    )
    
    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="x", row=i, col=j)
            fig.update_yaxes(title_text="y", row=i, col=j)
    
    fig.show()
    
    return results


if __name__ == "__main__":
    compare_velocity_distributions()
