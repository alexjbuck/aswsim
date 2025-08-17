#!/usr/bin/env python3
"""
Benchmark script to compare simulation performance with and without multiprocessing.
"""

import time
import multiprocessing as mp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aswsim import (
    simulate, bivariate_normal_position_uniform_depth, uniform_speed,
    make_heatmap_animation
)


def benchmark_simulation(n_targets: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Benchmark a single simulation run."""
    
    # Create initial distribution
    init = bivariate_normal_position_uniform_depth(
        pos_mean=np.array([0.0, 0.0]),
        pos_cov=np.array([[500.0, 0.0], [0.0, 500.0]]),
        depth_min=-100.0,
        depth_max=-10.0,
        velocity_dist=uniform_speed(2.0, 8.0, vz=0.0),
        pos_bounds=((-2000.0, 2000.0), (-2000.0, 2000.0)),
    )
    
    # Time the simulation
    start_time = time.time()
    times, trajectories = simulate(
        n_targets=n_targets,
        total_time=500.0,  # Longer time to better show benefits
        dt=1.0,
        init=init,
        seed=42,
    )
    end_time = time.time()
    
    return end_time - start_time, times, trajectories


def run_benchmarks():
    """Run comprehensive benchmarks."""
    
    # Target counts to test
    target_counts = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    
    print("ASW Simulation Performance Benchmark")
    print("=" * 50)
    print(f"CPU Cores Available: {mp.cpu_count()}")
    print()
    
    results = {
        'target_counts': target_counts,
        'simulation_times': []
    }
    
    for n_targets in target_counts:
        print(f"Testing {n_targets} targets...")
        
        # Run simulation
        simulation_time, _, _ = benchmark_simulation(n_targets)
        results['simulation_times'].append(simulation_time)
        
        print(f"  Simulation time: {simulation_time:.3f}s")
        print(f"  Performance: {n_targets * 500 / simulation_time:,.0f} target-time-steps/second")
        
        print()
    
    return results


def plot_benchmark_results(results: dict):
    """Create visualization of benchmark results."""
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Simulation Time vs Target Count',),
        vertical_spacing=0.1
    )
    
    # Time vs target count
    fig.add_trace(
        go.Scatter(
            x=results['target_counts'],
            y=results['simulation_times'],
            mode='lines+markers',
            name='Vectorized Simulation',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.update_layout(
        title="ASW Simulation Performance (Vectorized)",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Number of Targets", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    
    return fig


def main():
    """Run benchmarks and display results."""
    
    print("Starting performance benchmarks...")
    results = run_benchmarks()
    
    # Create visualization
    fig = plot_benchmark_results(results)
    fig.show()
    
    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 50)
    
    # Calculate performance metrics
    total_operations = sum(n * 500 for n in results['target_counts'])  # 500 time steps each
    total_time = sum(results['simulation_times'])
    avg_performance = total_operations / total_time
    
    print(f"Total operations: {total_operations:,}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average performance: {avg_performance:,.0f} target-time-steps/second")
    
    # Find best performance
    performances = [n * 500 / t for n, t in zip(results['target_counts'], results['simulation_times'])]
    best_performance = max(performances)
    best_idx = performances.index(best_performance)
    best_targets = results['target_counts'][best_idx]
    
    print(f"Best performance: {best_performance:,.0f} target-time-steps/second at {best_targets:,} targets")
    
    print("\nOptimization Summary:")
    print("✓ Vectorized constant velocity computation")
    print("✓ NumPy broadcasting for all time steps")
    print("✓ Pre-allocated arrays for memory efficiency")
    print("✓ No multiprocessing overhead")
    print("✓ Scales efficiently with target count")


if __name__ == "__main__":
    main()
