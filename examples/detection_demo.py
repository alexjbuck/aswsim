#!/usr/bin/env python3
"""Demo script for sensor detection system."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aswsim import (
    simulate, bivariate_normal_position_uniform_depth, uniform_speed,
    create_default_search_pattern, load_search_pattern
)
from pathlib import Path


def run_detection_demo():
    """Run a demonstration of the sensor detection system."""
    
    print("Running Sensor Detection Demo...")
    print("=" * 50)
    
    # Simulation parameters
    n_targets = 10_000
    total_time = 120.0  # minutes
    dt = 1.0  # minutes
    
    # Create initial distribution
    pos_mean = np.array([0.0, 0.0])
    pos_cov = np.array([[500.0, 0.0], [0.0, 500.0]])  # 500 yard std dev
    depth_min, depth_max = -100.0, -10.0
    velocity_dist = uniform_speed(2.0 * 2000/60, 8.0 * 2000/60)  # 2-8 knots
    
    init_dist = bivariate_normal_position_uniform_depth(
        pos_mean, pos_cov, depth_min, depth_max, velocity_dist
    )
    
    # Create search pattern
    search_pattern = create_default_search_pattern()
    
    print(f"Search Pattern: {search_pattern.name}")
    print(f"Description: {search_pattern.description}")
    print(f"Number of sensors: {len(search_pattern.sensors)}")
    print()
    
    for sensor in search_pattern.sensors:
        print(f"Sensor '{sensor.name}':")
        print(f"  Position: ({sensor.x:.0f}, {sensor.y:.0f}) yards")
        print(f"  Range: {sensor.range:.0f} yards")
        print(f"  Rate: {sensor.rate:.1f} detections/minute")
        print(f"  Activation time: {sensor.time:.1f} minutes")
        print()
    
    # Run simulation with detection tracking
    print("Running simulation...")
    times, trajectories, detection_stats = simulate(
        n_targets, total_time, dt, init_dist, search_pattern=search_pattern
    )
    
    print("Simulation complete!")
    print()
    
    # Display detection statistics
    if detection_stats:
        pattern_stats = detection_stats['pattern']
        sensor_stats = detection_stats['sensors']
        
        print("Detection Statistics Summary:")
        print("-" * 30)
        print(f"Total cumulative detections: {pattern_stats['cumulative_detections'][-1]:.1f}")
        print(f"Peak detection rate: {np.max(pattern_stats['total_detection_rates']):.1f} detections/minute")
        print(f"Average coverage: {np.mean(pattern_stats['average_coverage']):.1%}")
        print()
        
        print("Individual Sensor Performance:")
        print("-" * 30)
        for sensor_name, stats in sensor_stats.items():
            total_detections = stats['cumulative_detections'][-1]
            peak_rate = np.max(stats['detection_rates'])
            avg_coverage = np.mean(stats['coverage'])
            active_time = np.sum(stats['active']) * dt
            
            print(f"{sensor_name}:")
            print(f"  Total detections: {total_detections:.1f}")
            print(f"  Peak rate: {peak_rate:.1f} detections/minute")
            print(f"  Average coverage: {avg_coverage:.1%}")
            print(f"  Active time: {active_time:.1f} minutes")
            print()
    
    # Create visualization
    create_detection_visualization(times, trajectories, detection_stats, search_pattern)
    
    return detection_stats


def create_detection_visualization(times, trajectories, detection_stats, search_pattern):
    """Create visualization of detection results."""
    
    if detection_stats is None:
        print("No detection statistics to visualize")
        return
    
    pattern_stats = detection_stats['pattern']
    sensor_stats = detection_stats['sensors']
    
    # Create subplots: position heatmap, detection rate, cumulative detections
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Target Distribution (t=0)", 
            "Detection Rate Over Time",
            "Cumulative Detections", 
            "Sensor Coverage"
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Target distribution heatmap (initial time)
    x_all = trajectories[0, :, 0]
    y_all = trajectories[0, :, 1]
    
    # Create grid for heatmap
    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)
    pad_x = 0.1 * (x_max - x_min)
    pad_y = 0.1 * (y_max - y_min)
    
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, 50)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, 50)
    
    z, _, _ = np.histogram2d(x_all, y_all, bins=[x_edges, y_edges], density=True)
    
    heatmap = go.Heatmap(
        z=z.T, x=x_edges, y=y_edges,
        colorscale="Viridis", zsmooth="best",
        showscale=True,
        name="Target Density"
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # Add sensor positions
    for sensor in search_pattern.sensors:
        fig.add_trace(
            go.Scatter(
                x=[sensor.x], y=[sensor.y],
                mode='markers',
                marker=dict(size=15, color='red', symbol='diamond'),
                name=f"{sensor.name} (t={sensor.time:.0f}min)",
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Detection rate over time
    fig.add_trace(
        go.Scatter(
            x=times,
            y=pattern_stats['total_detection_rates'],
            mode='lines',
            name='Total Detection Rate',
            line=dict(color='blue', width=2)
        ),
        row=1, col=2
    )
    
    # Add individual sensor rates
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (sensor_name, stats) in enumerate(sensor_stats.items()):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=stats['detection_rates'],
                mode='lines',
                name=f'{sensor_name} Rate',
                line=dict(color=color, width=1, dash='dash'),
                opacity=0.7
            ),
            row=1, col=2
        )
    
    # 3. Cumulative detections
    fig.add_trace(
        go.Scatter(
            x=times,
            y=pattern_stats['cumulative_detections'],
            mode='lines',
            name='Total Cumulative',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    # Add individual sensor cumulative
    for i, (sensor_name, stats) in enumerate(sensor_stats.items()):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=stats['cumulative_detections'],
                mode='lines',
                name=f'{sensor_name} Cumulative',
                line=dict(color=color, width=1, dash='dash'),
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # 4. Sensor coverage over time
    for i, (sensor_name, stats) in enumerate(sensor_stats.items()):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=stats['coverage'] * 100,  # Convert to percentage
                mode='lines',
                name=f'{sensor_name} Coverage',
                line=dict(color=color, width=2)
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title="Sensor Detection System Demo",
        height=800,
        width=1200,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="x (yards)", row=1, col=1)
    fig.update_yaxes(title_text="y (yards)", row=1, col=1)
    
    fig.update_xaxes(title_text="Time (minutes)", row=1, col=2)
    fig.update_yaxes(title_text="Detection Rate (detections/min)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Detections", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (minutes)", row=2, col=2)
    fig.update_yaxes(title_text="Coverage (%)", row=2, col=2)
    
    fig.show()


def test_toml_loading():
    """Test loading search pattern from TOML file."""
    config_path = Path("search_patterns.toml")
    
    if config_path.exists():
        print("Testing TOML configuration loading...")
        search_pattern = load_search_pattern(config_path)
        
        print(f"Loaded pattern: {search_pattern.name}")
        print(f"Description: {search_pattern.description}")
        print(f"Sensors: {len(search_pattern.sensors)}")
        
        for sensor in search_pattern.sensors:
            print(f"  {sensor.name}: ({sensor.x:.0f}, {sensor.y:.0f}) at t={sensor.time:.1f}min")
        
        return search_pattern
    else:
        print("TOML configuration file not found, using default pattern")
        return create_default_search_pattern()


if __name__ == "__main__":
    # Test TOML loading
    search_pattern = test_toml_loading()
    print()
    
    # Run detection demo
    detection_stats = run_detection_demo()
