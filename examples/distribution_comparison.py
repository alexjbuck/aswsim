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
    grid_size = 150 
    
    # Common simulation parameters
    n_targets = 50_000
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
    a = 3.0
    b = 5.0
    speed_min = 2 * 2000/60 # yards/minute = 2 knots
    speed_max = 12 * 2000/60 # yards/minute = 10 knots
    speed_max_beta = 12 * 2000/60 # yards/minute = 12 knots
    # speed_mean = (speed_min + speed_max) / 2
    speed_mean = 5.5 * 2000/60
    speed_variance = speed_mean**2
    
    # Define different velocity distributions
    distributions = {
        f"Uniform Speed ({speed_min*60/2000:.2f}-{speed_max*60/2000:.2f} knots)": uniform_speed(speed_min, speed_max), # in yards per minute
        f"Normal (vx=0±{speed_mean*60/2000:.2f} knots, vy=0±{speed_mean*60/2000:.2f} knots)": bivariate_normal_velocity(
            np.array([0.0, 0.0]),
            np.array([[speed_variance, 0.0], [0.0, speed_variance]])
        ),
        # f"Rayleigh Speed (mode={speed_mode*60/2000:.2f} knots)": rayleigh_speed(speed_mode),
        f"Beta Speed (a=1.8, b=4, {2:.2f}-{20:.2f} knots)": beta_speed(1.8, 4.0, 2*2000/60, 20*2000/60),
        f"Beta Speed (a=3, b=5, {3:.2f}-{12:.2f} knots)": beta_speed(3, 5, 3*2000/60, 12*2000/60),
        f"Beta Speed (a=5, b=2.5, {1:.2f}-{12:.2f} knots)": beta_speed(5, 2.5, 1*2000/60, 12*2000/60),
        f"Beta Speed (a=.6, b=.4, {2:.2f}-{12:.2f} knots)": beta_speed(0.6, 0.4, 2*2000/60, 12*2000/60),
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
    
    # Create comparison plots with shared axes
    n_distributions = len(distributions)
    
    # Create subplots with shared axes for position plots (left column)
    # and shared axes for velocity plots (right column)
    fig = make_subplots(
        rows=n_distributions, cols=2,
        subplot_titles=[f"{name}" for name in np.repeat(list(distributions.keys()), 2)],
        specs=[[{"type": "heatmap"}, {"type": "histogram"}]] * n_distributions,
        vertical_spacing=0.04,
        horizontal_spacing=0.1,
        shared_xaxes=True,  # Share x-axes within each column
        # shared_yaxes=True   # Share y-axes within each column
    )
    
    # Get global bounds for consistent axes across all time steps
    all_x = np.concatenate([data['trajectories'][:, :, 0].flatten() for data in results.values()])
    all_y = np.concatenate([data['trajectories'][:, :, 1].flatten() for data in results.values()])
    crop_factor = 0.5
    x_min, x_max = float(np.min(all_x)) * crop_factor, float(np.max(all_x)) * crop_factor
    y_min, y_max = float(np.min(all_y)) * crop_factor, float(np.max(all_y)) * crop_factor
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, grid_size)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, grid_size)
    
    # Create initial heatmaps (t=0) and velocity histograms
    frames = []
    for i, (name, data) in enumerate(results.items()):
        row = i + 1  # Each distribution gets its own row
        
        # Position heatmap (left column)
        x_all = data['trajectories'][0, :, 0]
        y_all = data['trajectories'][0, :, 1]
        z, _, _ = np.histogram2d(x_all, y_all, bins=[x_edges, y_edges], density=True)
        
        heatmap = go.Heatmap(
            z=z.T, x=x_edges, y=y_edges,
            colorscale="Viridis", zsmooth="best",
            showscale=False
        )
        fig.add_trace(heatmap, row=row, col=1)
        
        # Velocity histogram (right column)
        speeds_knots = data['speeds'][0, :] * 60 / 2000  # Convert to knots
        histogram = go.Histogram(
            x=speeds_knots,
            xbins=dict(size=0.5, start=0, end=20),
            autobinx=False,
            name=f"{name}",
            opacity=0.7,
            showlegend=False
        )
        fig.add_trace(histogram, row=row, col=2)
    
    # Create frames for all time steps
    frames = []
    for t_idx in range(len(times)):
        frame_data = []
        for i, (name, data) in enumerate(results.items()):
            # Position heatmap (left column)
            x_all = data['trajectories'][t_idx, :, 0]
            y_all = data['trajectories'][t_idx, :, 1]
            z, _, _ = np.histogram2d(x_all, y_all, bins=[x_edges, y_edges], density=True)
            
            heatmap = go.Heatmap(
                z=z.T, x=x_edges, y=y_edges,
                colorscale="Viridis", zsmooth="best",
                showscale=False
            )
            frame_data.append(heatmap)
            
            # Velocity histogram (right column)
            speeds_knots = data['speeds'][t_idx, :] * 60 / 2000  # Convert to knots
            histogram = go.Histogram(
                x=speeds_knots,
                xbins=dict(size=0.5, start=0, end=20),
                autobinx=False,
                name=f"{name}",
                opacity=0.7,
                showlegend=False,
            )
            frame_data.append(histogram)
        
        frames.append(go.Frame(data=frame_data, name=str(t_idx)))
    
    fig.frames = frames
    
    # Update main layout
    fig.update_layout(
        height=400 * n_distributions,
        width=1000,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 100, "redraw": True}, "transition": {"duration": 0}}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
                ],
                "x": 0.0,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Time: ", "suffix": " min"},
                "steps": [
                    {
                        "label": f"t={times[i]:.1f}",
                        "method": "animate",
                        "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                    }
                    for i in range(len(times))
                ],
                "x": 0.10,
                "y": 1.1,
                "len": 0.9,
            }
        ],
    )
    
    # Update axes labels and set equal aspect ratio for position plots
    for i in range(1, n_distributions + 1):
        # Position heatmap axes (left column) - equal aspect ratio
        fig.update_xaxes(title_text="x (yards)", row=i, col=1)
        fig.update_yaxes(
            title_text="y (yards)", 
            row=i, 
            col=1,
            scaleanchor="x",
            scaleratio=1
        )
        
        # Velocity histogram axes (right column)
        fig.update_xaxes(title_text="Speed (knots)", row=i, col=2)
        fig.update_yaxes(title_text="Count", row=i, col=2)
    
    fig.show()
    
    return results


if __name__ == "__main__":
    compare_velocity_distributions()
