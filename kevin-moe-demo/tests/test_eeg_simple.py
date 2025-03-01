#!/usr/bin/env python3
"""
Simple test for EEG processor with fixed attention levels
"""

import time
from eeg_processor import EEGProcessor

def test_eeg_fixed_attention():
    """Test EEG processor with manually set attention levels"""
    print("Testing EEG processor with fixed attention levels...")
    
    # Create EEG processor without visualization
    eeg = EEGProcessor(simulation_mode=True, enable_visualization=False)
    eeg.start()
    
    try:
        # Test low attention
        eeg.attention_level = 0.2
        print("\nLow attention (0.2):")
        for _ in range(5):
            attention = eeg.get_attention_level()
            print(f"- Current attention: {attention:.2f}")
            time.sleep(0.5)
        
        # Test medium attention
        eeg.attention_level = 0.5
        print("\nMedium attention (0.5):")
        for _ in range(5):
            attention = eeg.get_attention_level()
            print(f"- Current attention: {attention:.2f}")
            time.sleep(0.5)
        
        # Test high attention
        eeg.attention_level = 0.8
        print("\nHigh attention (0.8):")
        for _ in range(5):
            attention = eeg.get_attention_level()
            print(f"- Current attention: {attention:.2f}")
            time.sleep(0.5)
        
        # Get overall metrics
        metrics = eeg.get_attention_metrics()
        print("\nAttention Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"- {key}: {value:.2f}")
            else:
                print(f"- {key}: {value}")
            
    finally:
        # Clean up
        eeg.stop()
        print("\nTest completed.")

if __name__ == "__main__":
    test_eeg_fixed_attention() 