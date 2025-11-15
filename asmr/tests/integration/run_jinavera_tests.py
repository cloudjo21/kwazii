#!/usr/bin/env python3
"""
Jinavera Integration Test Runner
Runs comprehensive tests for Jinavera FDE encoder with text and image inputs
"""

import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import test runner
import unittest

if __name__ == "__main__":
    print("=" * 60)
    print("JINAVERA FDE ENCODER - INTEGRATION TEST")
    print("=" * 60)
    print(f"Model path: /mnt/d/temp/user/ed/mart/llms/jina-embeddings-v4")
    print(f"Test directory: {current_dir}")
    print(f"Python path: {sys.path[:3]}...")
    print()
    
    # Set up test environment
    os.environ['PYTHONPATH'] = str(src_path)
    
    # Load and run tests
    loader = unittest.TestLoader()
    start_dir = current_dir
    suite = loader.discover(start_dir, pattern='test_jinavera_integration.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("Running Jinavera integration tests...")
    print("-" * 60)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)