#!/usr/bin/env python3
"""
Test runner for the EEG-Enhanced LLM system
"""

import unittest
import sys
import os
import importlib.util
import glob

def import_module_from_file(file_path):
    """Import a module from file path"""
    module_name = os.path.basename(file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing {file_path}: {e}")
        return None

def load_tests_from_directory(directory):
    """Load tests from a directory without using unittest discovery"""
    test_suite = unittest.TestSuite()
    
    # Find all test_*.py files in the directory
    test_files = glob.glob(os.path.join(directory, "test_*.py"))
    
    print(f"Found {len(test_files)} test files in {directory}")
    
    for test_file in test_files:
        # Import the module
        module = import_module_from_file(test_file)
        if module is None:
            continue
        
        # Find all test cases in the module
        for item_name in dir(module):
            item = getattr(module, item_name)
            if isinstance(item, type) and issubclass(item, unittest.TestCase) and item != unittest.TestCase:
                test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(item))
    
    return test_suite

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Create the test suite
    test_suite = unittest.TestSuite()
    
    # Tests in the dedicated tests directory
    tests_directory = os.path.join(os.path.dirname(__file__), 'tests')
    if os.path.exists(tests_directory) and os.path.isdir(tests_directory):
        print(f"Loading tests from {tests_directory}...")
        tests_from_directory = load_tests_from_directory(tests_directory)
        test_suite.addTest(tests_from_directory)
    
    # Tests in the root directory
    root_directory = os.path.dirname(__file__)
    print(f"Loading tests from {root_directory}...")
    tests_from_root = load_tests_from_directory(root_directory)
    test_suite.addTest(tests_from_root)
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful()) 