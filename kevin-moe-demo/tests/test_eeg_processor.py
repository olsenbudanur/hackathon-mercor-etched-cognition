#!/usr/bin/env python3
"""
Unit tests for EEG Processor component
"""

import unittest
import numpy as np
from eeg_processor import EEGProcessor
import time

class TestEEGProcessor(unittest.TestCase):
    
    def setUp(self):
        # Initialize EEG processor with simulation mode and no visualization
        self.eeg = EEGProcessor(simulation_mode=True, enable_visualization=False)
        self.eeg.start()
    
    def tearDown(self):
        # Clean up
        self.eeg.stop()
    
    def test_initialization(self):
        """Test initialization of EEG processor"""
        self.assertTrue(self.eeg.simulation_mode)
        self.assertFalse(self.eeg.visualization)
        self.assertTrue(self.eeg.running)
    
    def test_attention_level(self):
        """Test that attention level is within valid bounds"""
        # Wait a bit for processing to happen
        time.sleep(0.5)
        
        # Test multiple times
        for _ in range(5):
            attention = self.eeg.get_attention_level()
            self.assertGreaterEqual(attention, 0.0)
            self.assertLessEqual(attention, 1.0)
            time.sleep(0.1)
    
    def test_attention_metrics(self):
        """Test attention metrics calculation"""
        # Wait a bit for processing to happen
        time.sleep(1.0)
        
        metrics = self.eeg.get_attention_metrics()
        
        # Basic validation of metrics
        self.assertIn("mean", metrics)
        self.assertIn("std", metrics)
        self.assertIn("min", metrics)
        self.assertIn("max", metrics)
        
        # Values should be within valid ranges
        self.assertGreaterEqual(metrics["mean"], 0.0)
        self.assertLessEqual(metrics["mean"], 1.0)
        
    def test_simulation_data(self):
        """Test that simulated data is being generated"""
        # Call the simulation method directly
        simulated_data = self.eeg._simulate_eeg_data()
        
        # Should return an array of 4 values (4 EEG channels)
        self.assertEqual(len(simulated_data), 4)
        
        # Each value should be a float
        for value in simulated_data:
            self.assertIsInstance(value, float)

if __name__ == "__main__":
    unittest.main() 