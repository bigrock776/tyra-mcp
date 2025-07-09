#!/usr/bin/env python3
"""
MCP Integration Test Runner

Runs comprehensive tests for MCP tools and server functionality.
Includes performance benchmarks and safety validation.
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest


class MCPTestRunner:
    """Comprehensive MCP test runner with reporting."""
    
    def __init__(self, verbose: bool = False, benchmark: bool = False):
        self.verbose = verbose
        self.benchmark = benchmark
        self.test_results = {}
        self.start_time = None
        
    def run_tests(self, test_patterns: Optional[List[str]] = None) -> Dict:
        """Run MCP tests with optional pattern filtering."""
        self.start_time = datetime.utcnow()
        
        # Default test patterns if none provided
        if not test_patterns:
            test_patterns = [
                "tests/test_mcp_integration.py",
                "tests/test_mcp_server.py", 
                "tests/test_mcp_trading_safety.py"
            ]
        
        print("🧪 Starting MCP Integration Test Suite")
        print("=" * 50)
        
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        for pattern in test_patterns:
            print(f"\n📋 Running tests: {pattern}")
            print("-" * 30)
            
            # Configure pytest arguments
            pytest_args = [
                pattern,
                "-v" if self.verbose else "-q",
                "--tb=short",
                "--strict-markers",
                "--disable-warnings"
            ]
            
            if self.benchmark:
                pytest_args.extend(["--benchmark-only", "--benchmark-sort=mean"])
            
            # Run tests and capture results
            result = pytest.main(pytest_args)
            
            # Parse results (simplified - in real implementation would parse pytest output)
            if result == 0:
                print(f"✅ {pattern}: PASSED")
                total_passed += 1
            else:
                print(f"❌ {pattern}: FAILED")
                total_failed += 1
        
        # Generate summary
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        summary = {
            "total_test_files": len(test_patterns),
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "duration_seconds": duration,
            "timestamp": self.start_time.isoformat(),
            "success_rate": total_passed / len(test_patterns) if test_patterns else 0
        }
        
        self.test_results = summary
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("🏁 MCP TEST SUMMARY")
        print("=" * 50)
        
        print(f"📊 Test Files:    {summary['total_test_files']}")
        print(f"✅ Passed:        {summary['passed']}")
        print(f"❌ Failed:        {summary['failed']}")
        print(f"⏭️  Skipped:       {summary['skipped']}")
        print(f"⏱️  Duration:      {summary['duration_seconds']:.2f}s")
        print(f"📈 Success Rate:  {summary['success_rate']:.1%}")
        
        if summary['failed'] > 0:
            print(f"\n⚠️  {summary['failed']} test file(s) failed!")
            print("Review the output above for details.")
        else:
            print("\n🎉 All MCP tests passed successfully!")
    
    def run_specific_tests(self, test_type: str) -> bool:
        """Run specific category of tests."""
        test_configs = {
            "integration": {
                "patterns": ["tests/test_mcp_integration.py"],
                "description": "MCP Tool Integration Tests"
            },
            "server": {
                "patterns": ["tests/test_mcp_server.py"],
                "description": "MCP Server Tests"
            },
            "trading": {
                "patterns": ["tests/test_mcp_trading_safety.py"],
                "description": "Trading Safety Tests"
            },
            "unit": {
                "patterns": [
                    "tests/test_memory_manager.py",
                    "tests/test_hallucination_detector.py", 
                    "tests/test_embeddings.py",
                    "tests/test_reranking.py",
                    "tests/test_graph_engine.py",
                    "tests/test_cache_manager.py",
                    "tests/test_circuit_breaker.py",
                    "tests/test_performance_tracker.py"
                ],
                "description": "Unit Tests for Core Components"
            },
            "mcp": {
                "patterns": [
                    "tests/test_mcp_integration.py",
                    "tests/test_mcp_server.py",
                    "tests/test_mcp_trading_safety.py"
                ],
                "description": "All MCP Tests"
            },
            "all": {
                "patterns": [
                    # MCP Tests
                    "tests/test_mcp_integration.py",
                    "tests/test_mcp_server.py",
                    "tests/test_mcp_trading_safety.py",
                    # Unit Tests
                    "tests/test_memory_manager.py",
                    "tests/test_hallucination_detector.py", 
                    "tests/test_embeddings.py",
                    "tests/test_reranking.py",
                    "tests/test_graph_engine.py",
                    "tests/test_cache_manager.py",
                    "tests/test_circuit_breaker.py",
                    "tests/test_performance_tracker.py"
                ],
                "description": "Complete Test Suite"
            }
        }
        
        if test_type not in test_configs:
            print(f"❌ Unknown test type: {test_type}")
            print(f"Available types: {list(test_configs.keys())}")
            return False
        
        config = test_configs[test_type]
        print(f"🚀 Running {config['description']}")
        
        summary = self.run_tests(config["patterns"])
        return summary["failed"] == 0
    
    def validate_environment(self) -> bool:
        """Validate that environment is ready for MCP tests."""
        print("🔍 Validating test environment...")
        
        checks = []
        
        # Check test files exist
        test_files = [
            # MCP Tests
            "tests/test_mcp_integration.py",
            "tests/test_mcp_server.py",
            "tests/test_mcp_trading_safety.py",
            # Unit Tests
            "tests/test_memory_manager.py",
            "tests/test_hallucination_detector.py", 
            "tests/test_embeddings.py",
            "tests/test_reranking.py",
            "tests/test_graph_engine.py",
            "tests/test_cache_manager.py",
            "tests/test_circuit_breaker.py",
            "tests/test_performance_tracker.py"
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                checks.append(f"✅ {test_file} exists")
            else:
                checks.append(f"❌ {test_file} missing")
        
        # Check imports
        try:
            import pytest
            checks.append("✅ pytest available")
        except ImportError:
            checks.append("❌ pytest not installed")
        
        try:
            from src.mcp.server import TyraMemoryServer
            checks.append("✅ MCP server importable")
        except ImportError as e:
            checks.append(f"❌ MCP server import failed: {e}")
        
        # Print validation results
        for check in checks:
            print(f"  {check}")
        
        failed_checks = [c for c in checks if c.startswith("❌")]
        if failed_checks:
            print(f"\n⚠️  {len(failed_checks)} validation check(s) failed!")
            return False
        
        print("\n✅ Environment validation passed!")
        return True
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate detailed test report."""
        if not self.test_results:
            print("❌ No test results available. Run tests first.")
            return ""
        
        report = {
            "test_run": self.test_results,
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "timestamp": datetime.utcnow().isoformat()
            },
            "recommendations": self._generate_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Test report saved to: {output_file}")
        
        return json.dumps(report, indent=2)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not self.test_results:
            return ["Run tests to get recommendations"]
        
        success_rate = self.test_results.get("success_rate", 0)
        
        if success_rate < 0.5:
            recommendations.append("CRITICAL: Many tests failing - review implementation")
        elif success_rate < 0.8:
            recommendations.append("WARNING: Some tests failing - investigate issues")
        elif success_rate < 1.0:
            recommendations.append("GOOD: Most tests passing - fix remaining failures")
        else:
            recommendations.append("EXCELLENT: All tests passing - ready for production")
        
        if self.test_results.get("duration_seconds", 0) > 300:
            recommendations.append("Consider optimizing test performance - tests taking >5 minutes")
        
        return recommendations


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run MCP integration tests for Tyra Memory Server"
    )
    
    parser.add_argument(
        "test_type",
        choices=["integration", "server", "trading", "unit", "mcp", "all"],
        nargs="?",
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-b", "--benchmark",
        action="store_true", 
        help="Run performance benchmarks"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate environment setup"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        help="Generate JSON report to specified file"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = MCPTestRunner(verbose=args.verbose, benchmark=args.benchmark)
    
    # Validate environment first
    if not runner.validate_environment():
        print("\n❌ Environment validation failed!")
        print("Please fix the issues above before running tests.")
        sys.exit(1)
    
    if args.validate:
        print("\n✅ Environment validation completed successfully!")
        sys.exit(0)
    
    # Run tests
    print(f"\n🚀 Starting {args.test_type} tests...")
    success = runner.run_specific_tests(args.test_type)
    
    # Generate report if requested
    if args.report:
        runner.generate_report(args.report)
    
    # Exit with appropriate code
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()