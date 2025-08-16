#!/usr/bin/env python3
"""
Create distribution package for ASW simulator.
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime


def create_distribution():
    """Create a distribution package with executable and documentation."""
    
    project_root = Path(__file__).parent
    dist_dir = project_root / "dist"
    package_dir = project_root / "distribution"
    
    # Create distribution directory
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # Copy executable
    exe_path = dist_dir / "aswsim-cli"
    if exe_path.exists():
        shutil.copy2(exe_path, package_dir / "aswsim")
        print(f"✅ Copied executable: {exe_path.name}")
    else:
        print(f"❌ Executable not found: {exe_path}")
        return
    
    # Copy documentation
    readme_path = project_root / "README.md"
    if readme_path.exists():
        shutil.copy2(readme_path, package_dir / "README.md")
        print(f"✅ Copied README: {readme_path.name}")
    
    # Create quick start guide
    quick_start = package_dir / "QUICK_START.txt"
    with open(quick_start, 'w') as f:
        f.write("""ASW Target Distribution Simulator - Quick Start Guide

This is a standalone executable that simulates submarine target distributions.
No Python installation required!

USAGE:
  ./aswsim [options]

EXAMPLES:
  # Run with default settings
  ./aswsim

  # Run with custom parameters
  ./aswsim --n-targets 1000 --total-time 100 --velocity-type rayleigh

  # Run with custom position distribution
  ./aswsim --pos-mean-x 100 --pos-mean-y 200 --pos-std-x 50 --pos-std-y 30

  # Run with position bounds
  ./aswsim --pos-min-x -100 --pos-max-x 100 --pos-min-y -100 --pos-max-y 100

  # Get help
  ./aswsim --help

VELOCITY TYPES:
  uniform: Uniform speed between min and max
  rayleigh: Rayleigh speed distribution
  beta: Beta distribution with min/max bounds
  bivariate-normal: Correlated velocity components
  independent-normal: Independent x/y velocities

For detailed documentation, see README.md
""")
    print(f"✅ Created quick start guide: {quick_start.name}")
    
    # Create platform-specific instructions
    platform_guide = package_dir / "PLATFORM_INSTRUCTIONS.txt"
    with open(platform_guide, 'w') as f:
        f.write("""Platform-Specific Instructions

macOS:
  - Double-click the executable or run: ./aswsim
  - If you get a security warning, go to System Preferences > Security & Privacy
  - Click "Allow Anyway" for the aswsim executable

Linux:
  - Make executable: chmod +x aswsim
  - Run: ./aswsim

Windows:
  - Double-click aswsim.exe or run from command prompt
  - If you get a security warning, click "More info" then "Run anyway"

TROUBLESHOOTING:
  - If the executable doesn't run, try running from terminal/command prompt
  - Check that you have sufficient disk space and memory
  - The executable requires internet connection for plotly visualizations
""")
    print(f"✅ Created platform guide: {platform_guide.name}")
    
    # Create zip file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"aswsim_standalone_{timestamp}.zip"
    zip_path = project_root / zip_name
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)
    
    print(f"✅ Created distribution package: {zip_path}")
    print(f"📦 Package size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Show contents
    print("\n📁 Distribution package contents:")
    for file_path in package_dir.rglob('*'):
        if file_path.is_file():
            print(f"  - {file_path.name}")
    
    print(f"\n🎉 Distribution ready! Share: {zip_path.name}")


if __name__ == "__main__":
    create_distribution()
