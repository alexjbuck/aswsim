#!/usr/bin/env python3
"""
Build script to create standalone executables for ASW simulator examples.
"""

import sys
import subprocess
from pathlib import Path


def build_example_executable(example_name: str, entry_point: str):
    """Build standalone executable for an example script using PyInstaller."""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Single executable file
        f"--name={example_name}",  # Name of the executable
        "--add-data", f"{src_dir}:aswsim",  # Include source code
        "--hidden-import=plotly.graph_objs",  # Ensure plotly is included
        "--hidden-import=plotly.subplots",
        "--hidden-import=numpy",
        "--hidden-import=scipy",  # In case numpy uses scipy
        "--collect-all=plotly",  # Include all plotly components
        "--collect-all=numpy",
        "--exclude-module=matplotlib",  # Exclude optional dependencies
        "--exclude-module=pytest",
        "--exclude-module=IPython",
        "--exclude-module=jupyter",
        entry_point  # Entry point
    ]
    
    print(f"Building {example_name} executable...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run PyInstaller
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Build successful!")
        print(f"Executable created at: {project_root}/dist/{example_name}")
        
        # Show file size
        exe_path = project_root / "dist" / example_name
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"File size: {size_mb:.1f} MB")
        return True
    else:
        print("Build failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False


def build_distribution_comparison():
    """Build the distribution comparison example."""
    return build_example_executable(
        "distribution-comparison",
        "examples/distribution_comparison.py"
    )


def build_detection_demo():
    """Build the detection demo example."""
    return build_example_executable(
        "detection-demo",
        "examples/detection_demo.py"
    )


def build_all_examples():
    """Build all example executables."""
    examples = [
        ("distribution-comparison", "examples/distribution_comparison.py"),
        ("detection-demo", "examples/detection_demo.py"),
    ]
    
    success_count = 0
    for example_name, entry_point in examples:
        print(f"\n{'='*50}")
        print(f"Building {example_name}...")
        print(f"{'='*50}")
        
        if build_example_executable(example_name, entry_point):
            success_count += 1
        else:
            print(f"Failed to build {example_name}")
    
    print(f"\n{'='*50}")
    print(f"Build Summary: {success_count}/{len(examples)} examples built successfully")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build standalone ASW simulator example executables")
    parser.add_argument("--example", choices=["distribution-comparison", "detection-demo"], 
                       help="Build specific example")
    parser.add_argument("--all", action="store_true", help="Build all examples")
    
    args = parser.parse_args()
    
    if args.example == "distribution-comparison":
        build_distribution_comparison()
    elif args.example == "detection-demo":
        build_detection_demo()
    elif args.all:
        build_all_examples()
    else:
        print("Please specify --example <name> or --all")
        print("Available examples: distribution-comparison, detection-demo")
        sys.exit(1)
