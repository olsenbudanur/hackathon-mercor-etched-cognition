#!/usr/bin/env python3
"""
Simple test for MoE controller with fixed attention levels
"""

import torch
from moe_control import AttentionBasedMoEController

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.vocab_size = 10000
    
    def encode(self, text, add_special_tokens=True):
        return [100, 200, 300]

def test_expert_routing():
    """Test expert routing for different attention levels"""
    
    # Create controller with mock tokenizer
    tokenizer = MockTokenizer()
    controller = AttentionBasedMoEController(tokenizer, visualization=False)
    
    # Test attention levels
    test_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    
    print("Testing MoE Controller expert routing:")
    print("-------------------------------------")
    
    for attention in test_levels:
        # Update controller with current attention
        weights = controller.update(attention)
        
        # Get current dominant expert
        dominant_expert = controller.get_current_expert()
        
        # Get generation parameters
        params = controller.get_generation_params()
        
        # Print results
        print(f"\nAttention level: {attention:.1f}")
        print(f"Dominant expert: {dominant_expert}")
        print("Expert weights:")
        for expert, weight in weights.items():
            print(f"- {expert}: {weight:.3f}")
        print("Generation parameters:")
        for param, value in params.items():
            print(f"- {param}: {value}")
        
        # Create mock logits and test biasing
        mock_logits = torch.ones((1, 10000))  # Match tokenizer.vocab_size
        biased_logits = controller.apply_moe_logit_biasing(mock_logits)
        
        # Check if biasing had an effect
        diff = (biased_logits - mock_logits).abs().sum().item()
        print(f"Logit biasing effect: {diff:.2f}")

def main():
    test_expert_routing()

if __name__ == "__main__":
    main() 