#!/usr/bin/env python3
"""
Test runner for the EEG-Enhanced LLM system
"""

import unittest
import sys
import os

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('.', pattern='test_*.py')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful()) 