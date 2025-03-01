#!/usr/bin/env python3
"""
Test script for examining MoE routing with different EEG attention patterns
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from eeg_processor import EEGProcessor
from moe_control import AttentionBasedMoEController
import os

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.vocab_size = 10000
    
    def encode(self, text, add_special_tokens=True):
        return [100, 200, 300]

class ControlledEEGProcessor(EEGProcessor):
    """Extended EEG processor with controlled attention patterns for testing"""
    
    def __init__(self, pattern_type='sine', period=10.0, enable_visualization=False):
        """
        Initialize with controlled attention patterns
        
        Args:
            pattern_type: Type of pattern ('sine', 'step', 'random', 'increasing')
            period: Time period for pattern cycle in seconds
            enable_visualization: Whether to enable visualization
        """
        super().__init__(simulation_mode=True, enable_visualization=enable_visualization)
        self.pattern_type = pattern_type
        self.period = period
        self.start_time = None
        
        # For validation
        self.attention_values = []
        self.timestamps = []
    
    def start(self):
        """Start EEG processing with controlled pattern"""
        super().start()
        self.start_time = time.time()
    
    def _simulate_eeg_data(self):
        """Override simulation with controlled patterns"""
        t = time.time()
        
        if self.start_time is None:
            self.start_time = t
        
        elapsed = t - self.start_time
        
        # Generate controlled attention values based on pattern type
        if self.pattern_type == 'sine':
            # Sinusoidal pattern oscillating between 0.1 and 0.9
            self.attention_level = 0.5 + 0.4 * np.sin(2 * np.pi * elapsed / self.period)
        
        elif self.pattern_type == 'step':
            # Step function between low, medium, and high attention
            phase = (elapsed % self.period) / self.period
            if phase < 0.33:
                self.attention_level = 0.2  # Low
            elif phase < 0.67:
                self.attention_level = 0.5  # Medium
            else:
                self.attention_level = 0.8  # High
        
        elif self.pattern_type == 'random':
            # Random changes with persistence
            if not hasattr(self, 'last_change_time'):
                self.last_change_time = t
                self.random_target = 0.5
            
            # Change target occasionally
            if t - self.last_change_time > 2.0:
                self.last_change_time = t
                self.random_target = np.random.uniform(0.1, 0.9)
            
            # Move toward target
            self.attention_level += (self.random_target - self.attention_level) * 0.1
        
        elif self.pattern_type == 'increasing':
            # Steadily increasing from 0.1 to 0.9 over the period
            cycle_position = (elapsed % self.period) / self.period
            self.attention_level = 0.1 + 0.8 * cycle_position
        
        # Save for validation
        self.attention_values.append(self.attention_level)
        self.timestamps.append(elapsed)
        
        # Generate simulated EEG data
        simulated_data = []
        for i in range(4):
            alpha = (1.0 - self.attention_level) * np.sin(2 * np.pi * 10 * t + i)
            beta = self.attention_level * np.sin(2 * np.pi * 20 * t + i)
            noise = 0.1 * np.random.normal()
            signal = alpha + beta + noise
            simulated_data.append(signal)
        
        return simulated_data

def test_routing_with_pattern(pattern_type, duration=20):
    """
    Test MoE routing with a specific EEG pattern
    
    Args:
        pattern_type: Type of pattern to test
        duration: Test duration in seconds
    """
    print(f"\nTesting MoE routing with {pattern_type} attention pattern...")
    
    # Create controlled EEG processor
    eeg = ControlledEEGProcessor(pattern_type=pattern_type, enable_visualization=False)
    eeg.start()
    
    # Create MoE controller
    tokenizer = MockTokenizer()
    controller = AttentionBasedMoEController(tokenizer, visualization=False)
    
    # Track data for visualization
    timestamps = []
    attention_values = []
    expert_weights = {
        "simple": [],
        "balanced": [],
        "complex": []
    }
    dominant_experts = []
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run test for specified duration
    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            # Get attention from EEG processor
            attention = eeg.get_attention_level()
            
            # Update MoE controller with current attention
            weights = controller.update(attention)
            
            # Get current timestamp
            current_time = time.time() - start_time
            
            # Record data for visualization
            timestamps.append(current_time)
            attention_values.append(attention)
            
            for expert in expert_weights:
                expert_weights[expert].append(weights.get(expert, 0.0))
            
            # Determine which expert has highest weight
            dominant_expert = max(weights.items(), key=lambda x: x[1])[0]
            dominant_experts.append(dominant_expert)
            
            # Sleep briefly to simulate token generation time
            time.sleep(0.1)
    finally:
        # Clean up
        eeg.stop()
    
    # Visualize results
    plt.figure(figsize=(14, 10))
    
    # Plot attention pattern
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, attention_values, 'k-', linewidth=2)
    plt.title(f"{pattern_type.capitalize()} Attention Pattern")
    plt.ylabel("Attention Level")
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    # Plot expert weights
    plt.subplot(2, 1, 2)
    for expert, values in expert_weights.items():
        plt.plot(timestamps, values, linewidth=2, label=f"{expert.capitalize()} Expert")
    
    plt.title("Expert Weights")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Weight")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/moe_routing_{pattern_type}.png')
    print(f"Visualization saved to {results_dir}/moe_routing_{pattern_type}.png")
    
    # Calculate and print expert usage statistics
    total_samples = len(dominant_experts)
    expert_usage = {}
    for expert in set(dominant_experts):
        expert_usage[expert] = dominant_experts.count(expert) / total_samples * 100
    
    print("\nExpert Usage Statistics:")
    for expert, percentage in expert_usage.items():
        print(f"{expert.capitalize()}: {percentage:.1f}%")
    
    return expert_usage

def main():
    """Test routing with different patterns"""
    patterns = ['sine', 'step', 'random', 'increasing']
    test_duration = 15  # seconds per pattern
    
    for pattern in patterns:
        test_routing_with_pattern(pattern, test_duration)
        plt.close('all')
        time.sleep(1)  # Pause between tests

if __name__ == "__main__":
    main() 