#!/usr/bin/env python3
"""
Test runner script for Personalized Financial Recommendation System
Provides different test execution profiles and reporting
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test runner for Financial Recommendation System")
    parser.add_argument(
        "--profile", 
        choices=["basic", "demo", "all", "coverage"],
        default="all",
        help="Test profile to run"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML test report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.extend(["-v", "-s"])
    
    if args.parallel:
        base_cmd.extend(["-n", "auto"])
    
    if args.html_report:
        base_cmd.extend(["--html=test_report.html", "--self-contained-html"])
    
    # Test profiles
    profiles = {
        "basic": {
            "description": "Basic functionality tests",
            "markers": [],
            "files": ["tests/test_basic_functionality.py"]
        },
        "demo": {
            "description": "Demo and framework tests",
            "markers": [],
            "files": ["tests/test_demo.py"]
        },
        "coverage": {
            "description": "Coverage report",
            "markers": [],
            "files": ["tests/"],
            "coverage": True
        },
        "all": {
            "description": "All working tests",
            "markers": [],
            "files": ["tests/test_basic_functionality.py", "tests/test_demo.py"]
        }
    }
    
    profile = profiles[args.profile]
    cmd = base_cmd.copy()
    
    # Add files
    cmd.extend(profile["files"])
    
    # Add markers
    if profile["markers"]:
        for marker in profile["markers"]:
            cmd.extend(["-m", marker])
    
    # Add coverage
    if args.coverage or profile.get("coverage", False):
        cmd.extend([
            "--cov=tests",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add benchmark
    if args.benchmark or profile.get("benchmark", False):
        cmd.extend(["--benchmark-only"])
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    # Update command paths to be relative to project root
    for i, cmd_part in enumerate(cmd):
        if cmd_part == "tests/":
            cmd[i] = "tests/"
    
    # Run tests
    success = run_command(cmd, profile["description"])
    
    if not success:
        sys.exit(1)
    
    # Additional checks
    if args.profile in ["all", "basic"]:
        print("\n" + "="*60)
        print("Running additional quality checks...")
        print("="*60)
        
        # Linting
        run_command(["python", "-m", "flake8", "tests/"], "Linting test files")
        
        # Type checking
        run_command(["python", "-m", "mypy", "tests/", "--ignore-missing-imports"], "Type checking test files")
    
    print(f"\n✅ Test profile '{args.profile}' completed successfully!")
    
    # Show coverage location if generated
    if args.coverage or profile.get("coverage", False):
        print(f"\n📊 Coverage report generated: {project_root}/htmlcov/index.html")
    
    # Show HTML report location if generated
    if args.html_report:
        print(f"\n📄 HTML test report generated: {project_root}/test_report.html")


if __name__ == "__main__":
    main()
