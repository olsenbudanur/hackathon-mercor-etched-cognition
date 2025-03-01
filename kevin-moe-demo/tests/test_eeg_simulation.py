#!/usr/bin/env python3
"""
Test script for different EEG attention patterns
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from eeg_processor import EEGProcessor
import os
import threading

class ControlledEEGProcessor(EEGProcessor):
    """Extends EEGProcessor to provide controlled attention patterns for testing"""
    
    def __init__(self, pattern_type='sine', period=10, enable_visualization=False):
        """
        Initialize the controlled EEG processor
        
        Args:
            pattern_type: Type of attention pattern to generate
                          ('sine', 'step', 'random', 'increasing')
            period: Period of the pattern in seconds
            enable_visualization: Whether to enable visualization
        """
        super().__init__(simulation_mode=True, enable_visualization=enable_visualization)
        self.pattern_type = pattern_type
        self.period = period
        self.pattern_start_time = None
        
        # Visualization data
        self.times = []
        self.values = []
    
    def start(self):
        """Start the processor"""
        self.pattern_start_time = time.time()
        super().start()
    
    def _simulate_eeg_data(self):
        """Override the simulation method to provide controlled patterns"""
        # We'll control the attention level directly, but still need to provide EEG data
        # for the parent class to process
        
        # Generate basic simulated data
        eeg_data = [np.random.uniform(0, 1) for _ in range(4)]
        
        # Apply attention pattern to alpha channel (index 1)
        if self.pattern_start_time is not None:
            elapsed = time.time() - self.pattern_start_time
            attention = self._get_attention_from_pattern(elapsed)
            
            # Record for visualization
            self.times.append(elapsed)
            self.values.append(attention)
            
            # Use the controlled attention level
            self.attention_level = attention
        
        return eeg_data
    
    def _get_attention_from_pattern(self, elapsed):
        """
        Get attention level based on the specified pattern
        
        Args:
            elapsed: Elapsed time since start
        
        Returns:
            Attention level (0.0 to 1.0)
        """
        if self.pattern_type == 'sine':
            # Sine wave oscillating between 0.1 and 0.9
            return 0.5 + 0.4 * np.sin(2 * np.pi * elapsed / self.period)
        
        elif self.pattern_type == 'step':
            # Step function alternating between low and high
            step = int((elapsed % self.period) / (self.period/2))
            return 0.2 if step == 0 else 0.8
        
        elif self.pattern_type == 'random':
            # Random values, but with temporal coherence
            phase = elapsed % self.period
            seed = int(elapsed / self.period)
            np.random.seed(seed)
            base = np.random.uniform(0.1, 0.9)
            np.random.seed(None)  # Reset seed
            variation = 0.1 * np.sin(2 * np.pi * phase / (self.period/5))
            return max(0.1, min(0.9, base + variation))
        
        elif self.pattern_type == 'increasing':
            # Gradually increasing from 0.1 to 0.9
            cycle = elapsed % self.period
            return 0.1 + 0.8 * (cycle / self.period)
        
        # Default
        return 0.5
    
    def visualize_pattern(self):
        """Visualize the attention pattern"""
        if not self.times:
            print("No data to visualize yet")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.times, self.values, 'b-')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Attention Level')
        plt.title(f'Attention Pattern: {self.pattern_type}')
        plt.ylim(0, 1)
        plt.grid(True)
        
        # Create results directory if it doesn't exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        plt.savefig(f'{results_dir}/attention_pattern_{self.pattern_type}.png')
        print(f"Pattern visualization saved to {results_dir}/attention_pattern_{self.pattern_type}.png")
        
        plt.close()

def test_pattern(pattern_type, duration=15):
    """
    Test a specific attention pattern
    
    Args:
        pattern_type: Type of pattern to test
        duration: Test duration in seconds
    """
    print(f"\nTesting attention pattern: {pattern_type}")
    
    # Create controlled EEG processor
    eeg = ControlledEEGProcessor(pattern_type=pattern_type, enable_visualization=False)
    eeg.start()
    
    # Run for specified duration
    try:
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Get current attention
            attention = eeg.get_attention_level()
            
            # Display current status
            elapsed = time.time() - start_time
            print(f"\rTime: {elapsed:.1f}s | Attention: {attention:.2f}", end="")
            
            # Sleep briefly
            time.sleep(0.1)
        
        print("\nTest completed.")
        
        # Visualize the pattern
        eeg.visualize_pattern()
    
    finally:
        eeg.stop()

def main():
    """Test all patterns"""
    patterns = ['sine', 'step', 'random', 'increasing']
    test_duration = 15  # seconds per pattern
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    for pattern in patterns:
        test_pattern(pattern, test_duration)
        time.sleep(1)  # Pause between tests

if __name__ == "__main__":
    main() 