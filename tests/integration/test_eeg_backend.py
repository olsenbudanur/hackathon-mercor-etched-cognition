"""
Integration tests for the interaction between the EEG processing and backend components.

These tests verify that the EEG processing and backend components work together correctly,
focusing on attention level detection, expert selection, and token generation based on EEG signals.
"""

import pytest
import numpy as np
import json
import time
from unittest.mock import patch, MagicMock
import sys
import os

# Add the necessary directories to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../eeg')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))

# Import the EEG processing functions
from eeg import compute_band_power, process_data, compute_focus_index, BANDS, FS

# Import the backend app
from main import app, token_queue, add_tokens, TokenData, TokenStreamData

# Import the test client
from fastapi.testclient import TestClient

# Create a test client
client = TestClient(app)


class TestEEGBackendIntegration:
    """Test class for EEG-backend integration."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear the token queue before each test
        token_queue.clear()

    def _create_synthetic_eeg_data(self, theta_amp=0.5, alpha_amp=0.5, beta_amp=0.5, gamma_amp=0.5, duration=1.0, channels=4):
        """Create synthetic EEG data with specified frequency band amplitudes."""
        # Define parameters
        fs = FS  # Sampling frequency
        t = np.arange(0, duration, 1/fs)  # Time vector
        n_samples = len(t)
        
        # Create frequency components
        theta_wave = theta_amp * np.sin(2 * np.pi * 6 * t)  # 6 Hz (theta band)
        alpha_wave = alpha_amp * np.sin(2 * np.pi * 10 * t)  # 10 Hz (alpha band)
        beta_wave = beta_amp * np.sin(2 * np.pi * 20 * t)  # 20 Hz (beta band)
        gamma_wave = gamma_amp * np.sin(2 * np.pi * 40 * t)  # 40 Hz (gamma band)
        
        # Combine frequency components
        eeg_signal = theta_wave + alpha_wave + beta_wave + gamma_wave
        
        # Add some noise
        noise = 0.1 * np.random.randn(n_samples)
        eeg_signal += noise
        
        # Replicate for multiple channels
        eeg_data = np.tile(eeg_signal, (channels, 1))
        
        return eeg_data

    @patch('eeg.inlet')
    def test_eeg_attention_to_expert_mapping(self, mock_inlet):
        """Test mapping of EEG attention levels to expert selection."""
        # 1. Create synthetic EEG data with known frequency components
        # Create data for different attention levels
        
        # Low attention: high theta/alpha, low beta
        low_attention_data = self._create_synthetic_eeg_data(
            theta_amp=1.0,
            alpha_amp=1.0,
            beta_amp=0.2
        )
        
        # Medium attention: balanced theta/alpha/beta
        medium_attention_data = self._create_synthetic_eeg_data(
            theta_amp=0.5,
            alpha_amp=0.5,
            beta_amp=0.5
        )
        
        # High attention: low theta/alpha, high beta
        high_attention_data = self._create_synthetic_eeg_data(
            theta_amp=0.2,
            alpha_amp=0.2,
            beta_amp=1.0
        )
        
        # 2. Process the EEG data to extract attention metrics
        # Mock the inlet.pull_chunk method to return the synthetic data
        mock_inlet.pull_chunk.side_effect = [
            (low_attention_data.T.tolist(), [0.0] * low_attention_data.shape[1]),
            (medium_attention_data.T.tolist(), [0.0] * medium_attention_data.shape[1]),
            (high_attention_data.T.tolist(), [0.0] * high_attention_data.shape[1]),
            ([], [])  # End of data
        ]
        
        # Import the EEG processing function that would normally be called in a separate process
        from eeg import collect_data
        
        # Collect and process the data for each attention level
        low_attention_eeg, _ = collect_data(recording_time=0.1)
        low_attention_bands = process_data(low_attention_eeg)
        low_attention_focus = compute_focus_index(
            low_attention_bands["theta"],
            low_attention_bands["alpha"],
            low_attention_bands["beta"]
        )
        
        medium_attention_eeg, _ = collect_data(recording_time=0.1)
        medium_attention_bands = process_data(medium_attention_eeg)
        medium_attention_focus = compute_focus_index(
            medium_attention_bands["theta"],
            medium_attention_bands["alpha"],
            medium_attention_bands["beta"]
        )
        
        high_attention_eeg, _ = collect_data(recording_time=0.1)
        high_attention_bands = process_data(high_attention_eeg)
        high_attention_focus = compute_focus_index(
            high_attention_bands["theta"],
            high_attention_bands["alpha"],
            high_attention_bands["beta"]
        )
        
        # 3. Map attention levels to expert selection
        # Define thresholds for attention levels
        LOW_THRESHOLD = 0.5
        HIGH_THRESHOLD = 2.0
        
        # Function to map focus index to expert
        def map_focus_to_expert(focus_index):
            if focus_index < LOW_THRESHOLD:
                return "simple"
            elif focus_index < HIGH_THRESHOLD:
                return "balanced"
            else:
                return "complex"
        
        # Map each attention level to an expert
        low_attention_expert = map_focus_to_expert(np.mean(low_attention_focus))
        medium_attention_expert = map_focus_to_expert(np.mean(medium_attention_focus))
        high_attention_expert = map_focus_to_expert(np.mean(high_attention_focus))
        
        # 4. Check that the mapping is correct
        # Low attention should map to the simple expert
        assert low_attention_expert == "simple"
        
        # Medium attention should map to the balanced expert
        assert medium_attention_expert == "balanced"
        
        # High attention should map to the complex expert
        assert high_attention_expert == "complex"
        
        # 5. Generate tokens based on the selected experts
        # Create tokens for each attention level
        tokens = [
            TokenData(token="Low", expert=low_attention_expert),
            TokenData(token=" attention", expert=low_attention_expert),
            TokenData(token=" maps", expert=low_attention_expert),
            TokenData(token=" to", expert=low_attention_expert),
            TokenData(token=" simple", expert=low_attention_expert),
            TokenData(token=". ", expert=medium_attention_expert),
            TokenData(token="Medium", expert=medium_attention_expert),
            TokenData(token=" attention", expert=medium_attention_expert),
            TokenData(token=" maps", expert=medium_attention_expert),
            TokenData(token=" to", expert=medium_attention_expert),
            TokenData(token=" balanced", expert=medium_attention_expert),
            TokenData(token=". ", expert=high_attention_expert),
            TokenData(token="High", expert=high_attention_expert),
            TokenData(token=" attention", expert=high_attention_expert),
            TokenData(token=" maps", expert=high_attention_expert),
            TokenData(token=" to", expert=high_attention_expert),
            TokenData(token=" complex", expert=high_attention_expert),
            TokenData(token=".", expert=high_attention_expert)
        ]
        
        # Add the tokens to the backend
        token_stream_data = TokenStreamData(tokens=tokens)
        add_tokens(token_stream_data)
        
        # 6. Check that the tokens were added with the correct expert colors
        # Get the tokens from the debug endpoint
        response = client.get("/debug-tokens")
        assert response.status_code == 200
        debug_tokens = response.json()["tokens"]
        
        # Check that the tokens have the correct expert colors
        # Simple expert (low attention) should have color 1
        for i in range(5):
            assert debug_tokens[i]["number"] == 1
        
        # Balanced expert (medium attention) should have color 2
        for i in range(5, 11):
            assert debug_tokens[i]["number"] == 2
        
        # Complex expert (high attention) should have color 3
        for i in range(11, 18):
            assert debug_tokens[i]["number"] == 3

    def test_expert_color_mapping(self):
        """Test mapping of experts to colors in the backend."""
        # Add tokens with different experts
        tokens = [
            TokenData(token="Simple", expert="simple"),
            TokenData(token="Balanced", expert="balanced"),
            TokenData(token="Complex", expert="complex")
        ]
        add_tokens(TokenStreamData(tokens=tokens))
        
        # Check that the tokens were added with the correct colors
        response = client.get("/debug-tokens")
        assert response.status_code == 200
        debug_tokens = response.json()["tokens"]
        
        # Check that we have 3 tokens
        assert len(debug_tokens) == 3
        
        # Check that the tokens have the correct colors
        assert debug_tokens[0]["word"] == "Simple"
        assert debug_tokens[0]["number"] == 1  # simple expert color
        
        assert debug_tokens[1]["word"] == "Balanced"
        assert debug_tokens[1]["number"] == 2  # balanced expert color
        
        assert debug_tokens[2]["word"] == "Complex"
        assert debug_tokens[2]["number"] == 3  # complex expert color

    def test_eeg_attention_simulation(self):
        """Test simulation of EEG attention levels and their effect on token generation."""
        # This test simulates how the EEG attention levels would affect token generation
        # in the Kevin-MOE demo.
        
        # 1. Define the attention levels
        attention_levels = ["low", "medium", "high"]
        
        # 2. Define the expert weights for each attention level
        expert_weights = {
            "low": {"simple": 0.7, "balanced": 0.2, "complex": 0.1},
            "medium": {"simple": 0.2, "balanced": 0.6, "complex": 0.2},
            "high": {"simple": 0.1, "balanced": 0.2, "complex": 0.7}
        }
        
        # 3. Define a function to select an expert based on weights
        def select_expert(weights):
            experts = list(weights.keys())
            probabilities = list(weights.values())
            return np.random.choice(experts, p=probabilities)
        
        # 4. Generate tokens for each attention level
        np.random.seed(42)  # For reproducibility
        tokens = []
        
        # Generate 10 tokens for each attention level
        for attention in attention_levels:
            for i in range(10):
                expert = select_expert(expert_weights[attention])
                token = f"{attention}-{i}"
                tokens.append(TokenData(token=token, expert=expert))
        
        # 5. Add the tokens to the backend
        token_stream_data = TokenStreamData(tokens=tokens)
        add_tokens(token_stream_data)
        
        # 6. Check that the tokens were added with the correct expert colors
        # Get the tokens from the debug endpoint
        response = client.get("/debug-tokens")
        assert response.status_code == 200
        debug_tokens = response.json()["tokens"]
        
        # Count the number of tokens for each expert and attention level
        expert_counts = {
            "low": {"simple": 0, "balanced": 0, "complex": 0},
            "medium": {"simple": 0, "balanced": 0, "complex": 0},
            "high": {"simple": 0, "balanced": 0, "complex": 0}
        }
        
        for i, token in enumerate(debug_tokens):
            attention = token["word"].split("-")[0]
            expert_number = token["number"]
            
            if expert_number == 1:
                expert_counts[attention]["simple"] += 1
            elif expert_number == 2:
                expert_counts[attention]["balanced"] += 1
            elif expert_number == 3:
                expert_counts[attention]["complex"] += 1
        
        # Check that the expert distribution roughly matches the weights
        # (with some tolerance due to randomness)
        for attention in attention_levels:
            total = sum(expert_counts[attention].values())
            for expert, weight in expert_weights[attention].items():
                expected = weight * 10
                actual = expert_counts[attention][expert]
                # Allow for some deviation due to randomness
                assert abs(actual - expected) <= 5, f"Expected {expected} {expert} tokens for {attention} attention, got {actual}"

    def test_token_streaming(self):
        """Test streaming of tokens from the backend."""
        # Add tokens to the queue
        tokens = [
            TokenData(token="Hello", expert="simple"),
            TokenData(token=" world", expert="complex")
        ]
        add_tokens(TokenStreamData(tokens=tokens))
        
        # Check that the tokens are in the queue
        response = client.get("/debug-tokens")
        assert response.status_code == 200
        debug_tokens = response.json()["tokens"]
        
        # Check that we have 2 tokens
        assert len(debug_tokens) == 2
        
        # Check that the tokens have the correct values
        assert debug_tokens[0]["word"] == "Hello"
        assert debug_tokens[0]["number"] == 1  # simple expert color
        
        assert debug_tokens[1]["word"] == " world"
        assert debug_tokens[1]["number"] == 3  # complex expert color


class TestMOEIntegration:
    """Test class for Mixture-of-Experts integration."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear the token queue before each test
        token_queue.clear()

    def test_moe_routing(self):
        """Test routing of tokens through different experts based on attention."""
        # Define attention levels and corresponding experts
        attention_levels = {
            "low": "simple",
            "medium": "balanced",
            "high": "complex"
        }
        
        # Generate tokens for each attention level
        tokens = []
        for attention, expert in attention_levels.items():
            for i in range(5):
                token = f"{attention}-token-{i}"
                tokens.append(TokenData(token=token, expert=expert))
        
        # Add the tokens to the backend
        token_stream_data = TokenStreamData(tokens=tokens)
        add_tokens(token_stream_data)
        
        # Check that the tokens were added with the correct expert colors
        response = client.get("/debug-tokens")
        assert response.status_code == 200
        debug_tokens = response.json()["tokens"]
        
        # Check that we have the expected number of tokens
        assert len(debug_tokens) == 15
        
        # Check that the tokens have the correct expert colors
        for i in range(5):
            # Low attention tokens should have color 1 (simple expert)
            assert debug_tokens[i]["word"].startswith("low")
            assert debug_tokens[i]["number"] == 1
            
            # Medium attention tokens should have color 2 (balanced expert)
            assert debug_tokens[i+5]["word"].startswith("medium")
            assert debug_tokens[i+5]["number"] == 2
            
            # High attention tokens should have color 3 (complex expert)
            assert debug_tokens[i+10]["word"].startswith("high")
            assert debug_tokens[i+10]["number"] == 3

    def test_moe_token_generation(self):
        """Test generation of tokens by different experts."""
        # Define the experts and their token generation characteristics
        experts = {
            "simple": {"vocabulary": ["simple", "easy", "basic"], "color": 1},
            "balanced": {"vocabulary": ["balanced", "moderate", "average"], "color": 2},
            "complex": {"vocabulary": ["complex", "advanced", "sophisticated"], "color": 3}
        }
        
        # Generate tokens from each expert
        tokens = []
        for expert, properties in experts.items():
            for word in properties["vocabulary"]:
                tokens.append(TokenData(token=word, expert=expert))
        
        # Add the tokens to the backend
        token_stream_data = TokenStreamData(tokens=tokens)
        add_tokens(token_stream_data)
        
        # Check that the tokens were added with the correct expert colors
        response = client.get("/debug-tokens")
        assert response.status_code == 200
        debug_tokens = response.json()["tokens"]
        
        # Check that we have the expected number of tokens
        assert len(debug_tokens) == 9
        
        # Check that the tokens have the correct expert colors
        for i, token in enumerate(debug_tokens):
            word = token["word"]
            number = token["number"]
            
            # Find which expert this word belongs to
            for expert, properties in experts.items():
                if word in properties["vocabulary"]:
                    expected_color = properties["color"]
                    assert number == expected_color, f"Expected color {expected_color} for word '{word}', got {number}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
