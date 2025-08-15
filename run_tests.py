#!/usr/bin/env python3
"""
Test runner script for RAG system tests
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run RAG system tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "performance", "all"], 
        default="unit",
        help="Type of tests to run (default: unit)"
    )
    parser.add_argument(
        "--service", 
        choices=["query", "retrieval", "generation", "memory", "monitoring", "orchestrator", "all"],
        default="all",
        help="Specific service to test (default: all)"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--html-report", 
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test markers based on type
    if args.type == "unit":
        cmd.extend(["-m", "unit"])
    elif args.type == "integration":
        cmd.extend(["-m", "integration"])
    elif args.type == "performance":
        cmd.extend(["-m", "performance"])
    elif args.type == "all":
        # Run all tests
        pass
    
    # Add service-specific tests
    if args.service != "all":
        service_map = {
            "query": "tests/services/test_query_service.py",
            "retrieval": "tests/services/test_retrieval_service.py", 
            "generation": "tests/services/test_generation_service.py",
            "memory": "tests/services/test_memory_service.py",
            "monitoring": "tests/services/test_monitoring_service.py",
            "orchestrator": "tests/services/test_rag_orchestrator.py"
        }
        if args.service in service_map:
            cmd.append(service_map[args.service])
    else:
        cmd.append("tests/services/")
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
        if args.html_report:
            cmd.extend(["--cov-report=html"])
    
    # Add verbose output
    if args.verbose:
        cmd.append("-v")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Run the tests
    success = run_command(cmd, f"Running {args.type} tests for {args.service} service(s)")
    
    if success:
        print(f"\n✅ Tests completed successfully!")
        
        # Show coverage report location if generated
        if args.coverage and args.html_report:
            html_path = Path("htmlcov/index.html").absolute()
            print(f"📊 HTML coverage report: {html_path}")
        
    else:
        print(f"\n❌ Tests failed!")
        sys.exit(1)


def run_quick_tests():
    """Run a quick subset of tests for development"""
    print("Running quick development tests...")
    
    quick_tests = [
        (["python", "-m", "pytest", "tests/services/", "-m", "unit", "--maxfail=5", "-x"], 
         "Quick unit tests"),
    ]
    
    all_passed = True
    for cmd, description in quick_tests:
        if not run_command(cmd, description):
            all_passed = False
            break
    
    if all_passed:
        print("\n✅ Quick tests passed!")
    else:
        print("\n❌ Quick tests failed!")
        sys.exit(1)


def run_ci_tests():
    """Run tests suitable for CI/CD pipeline"""
    print("Running CI/CD tests...")
    
    ci_tests = [
        (["python", "-m", "pytest", "tests/services/", "-m", "unit", "--cov=src", 
          "--cov-report=xml", "--cov-fail-under=80", "--junitxml=test-results.xml"], 
         "CI Unit tests with coverage"),
        (["python", "-m", "pytest", "tests/services/", "-m", "integration", "--maxfail=3"], 
         "CI Integration tests"),
    ]
    
    all_passed = True
    for cmd, description in ci_tests:
        if not run_command(cmd, description):
            all_passed = False
            # Continue with other tests even if one fails in CI
    
    if all_passed:
        print("\n✅ All CI tests passed!")
    else:
        print("\n❌ Some CI tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Check if pytest is available
    try:
        subprocess.run(["python", "-m", "pytest", "--version"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("ERROR: pytest is not installed. Please install it with:")
        print("pip install pytest pytest-cov pytest-xdist")
        sys.exit(1)
    
    # Special commands
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "ci":
        run_ci_tests()
    else:
        main()