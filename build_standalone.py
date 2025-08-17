#!/usr/bin/env python3
"""
Build script to create standalone executable for ASW simulator.
"""

import os
import sys
import subprocess
from pathlib import Path


def build_executable():
    """Build standalone executable using PyInstaller."""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Single executable file
        "--name=aswsim-cli",  # Name of the executable
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
        "src/aswsim/__main__.py"  # Entry point
    ]
    
    print("Building standalone executable...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run PyInstaller
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Build successful!")
        print(f"Executable created at: {project_root}/dist/aswsim-cli")
        
        # Show file size
        exe_path = project_root / "dist" / "aswsim-cli"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"File size: {size_mb:.1f} MB")
    else:
        print("Build failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build standalone ASW simulator executable")
    parser.add_argument("--console", action="store_true", help="Build console version (default)")
    
    args = parser.parse_args()
    
    # Always build console version
    build_executable()
