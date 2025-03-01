#!/usr/bin/env python3
"""
Unit tests for Mixture-of-Experts Controller
"""

import unittest
import torch
import numpy as np
from moe_control import AttentionBasedMoEController

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.vocab_size = 10000
    
    def encode(self, text, add_special_tokens=True):
        # Mock encoding - just return some random IDs
        return [100, 200, 300]

class TestMoEController(unittest.TestCase):
    
    def setUp(self):
        # Initialize controller with mock tokenizer and no visualization
        self.tokenizer = MockTokenizer()
        self.controller = AttentionBasedMoEController(self.tokenizer, visualization=False)
    
    def test_initialization(self):
        """Test initialization of MoE controller"""
        # Check that experts are properly defined
        self.assertIn("simple", self.controller.experts)
        self.assertIn("balanced", self.controller.experts)
        self.assertIn("complex", self.controller.experts)
        
        # Check initial weights
        for expert in self.controller.current_weights:
            self.assertEqual(self.controller.current_weights[expert], 0.0)
    
    def test_update_low_attention(self):
        """Test expert weighting for low attention level"""
        # Update with low attention (0.1)
        weights = self.controller.update(0.1)
        
        # Simple expert should have highest weight
        self.assertGreater(weights["simple"], weights["balanced"])
        self.assertGreater(weights["simple"], weights["complex"])
    
    def test_update_medium_attention(self):
        """Test expert weighting for medium attention level"""
        # Update with medium attention (0.5)
        weights = self.controller.update(0.5)
        
        # Balanced expert should have highest weight
        self.assertGreater(weights["balanced"], weights["simple"])
        self.assertGreater(weights["balanced"], weights["complex"])
    
    def test_update_high_attention(self):
        """Test expert weighting for high attention level"""
        # Update with high attention (0.9)
        weights = self.controller.update(0.9)
        
        # Complex expert should have highest weight
        self.assertGreater(weights["complex"], weights["simple"])
        self.assertGreater(weights["complex"], weights["balanced"])
    
    def test_logit_biasing(self):
        """Test that logit biasing modifies logits"""
        # Create mock logits
        mock_logits = torch.ones((1, 1000))
        original_sum = mock_logits.sum().item()
        
        # Update weights first
        self.controller.update(0.5)
        
        # Apply biasing
        biased_logits = self.controller.apply_moe_logit_biasing(mock_logits)
        
        # Logits should be modified (not equal to original)
        self.assertNotEqual(biased_logits.sum().item(), original_sum)
    
    def test_generation_params(self):
        """Test getting generation parameters"""
        # Test for different attention levels
        attention_levels = [0.1, 0.5, 0.9]
        
        for attn in attention_levels:
            self.controller.update(attn)
            params = self.controller.get_generation_params()
            
            # Params should include key generation parameters
            self.assertIn("temperature", params)
            self.assertIn("top_k", params)
            self.assertIn("repetition_penalty", params)
            
            # Params should be reasonable values
            self.assertGreater(params["temperature"], 0)
            self.assertGreater(params["top_k"], 0)
            self.assertGreater(params["repetition_penalty"], 1.0)
    
    def test_get_current_expert(self):
        """Test getting current dominant expert"""
        # For low attention, simple should be dominant
        self.controller.update(0.1)
        self.assertEqual(self.controller.get_current_expert(), "simple")
        
        # For high attention, complex should be dominant
        self.controller.update(0.9)
        self.assertEqual(self.controller.get_current_expert(), "complex")

if __name__ == "__main__":
    unittest.main() 