"""
Unit tests for focus index calculation in EEG processing.

These tests cover the computation of focus/attention metrics from EEG frequency band powers.
"""

import pytest
import numpy as np
import sys
import os

# Add the eeg directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../eeg')))

# Import the focus index computation function
from eeg import compute_focus_index


class TestFocusIndexComputation:
    """Test class for focus index computation."""

    def test_compute_focus_index_basic(self):
        """Test basic focus index computation."""
        # Create test data
        theta = [1.0, 2.0, 3.0]
        alpha = [2.0, 3.0, 4.0]
        beta = [6.0, 9.0, 12.0]
        
        # Compute focus index
        focus_index = compute_focus_index(theta, alpha, beta)
        
        # Check that the focus index has the expected length
        assert len(focus_index) == 3
        
        # Check that the focus index values are correct
        # focus_index = beta / (alpha + theta)
        expected = [6.0 / (1.0 + 2.0), 9.0 / (2.0 + 3.0), 12.0 / (3.0 + 4.0)]
        for i in range(3):
            assert focus_index[i] == pytest.approx(expected[i])

    def test_compute_focus_index_zero_denominator(self):
        """Test focus index computation with zero denominator."""
        # Create test data with zero alpha and theta
        theta = [0.0, 0.0, 0.0]
        alpha = [0.0, 0.0, 0.0]
        beta = [1.0, 2.0, 3.0]
        
        # Compute focus index
        focus_index = compute_focus_index(theta, alpha, beta)
        
        # Check that the focus index has the expected length
        assert len(focus_index) == 3
        
        # Check that the focus index values are finite (not infinity)
        # The function should handle division by zero
        for value in focus_index:
            assert np.isfinite(value)
            assert value > 0  # Should be positive

    def test_compute_focus_index_high_focus(self):
        """Test focus index computation with high focus (high beta, low alpha/theta)."""
        # Create test data for high focus
        theta = [0.5, 0.6, 0.7]
        alpha = [0.8, 0.9, 1.0]
        beta = [10.0, 12.0, 14.0]
        
        # Compute focus index
        focus_index = compute_focus_index(theta, alpha, beta)
        
        # Check that the focus index values are high
        for value in focus_index:
            assert value > 5.0  # Arbitrary threshold for "high" focus

    def test_compute_focus_index_low_focus(self):
        """Test focus index computation with low focus (low beta, high alpha/theta)."""
        # Create test data for low focus
        theta = [5.0, 6.0, 7.0]
        alpha = [8.0, 9.0, 10.0]
        beta = [1.0, 1.2, 1.4]
        
        # Compute focus index
        focus_index = compute_focus_index(theta, alpha, beta)
        
        # Check that the focus index values are low
        for value in focus_index:
            assert value < 0.2  # Arbitrary threshold for "low" focus

    def test_compute_focus_index_numpy_arrays(self):
        """Test focus index computation with NumPy arrays."""
        # Create test data as NumPy arrays
        theta = np.array([1.0, 2.0, 3.0])
        alpha = np.array([2.0, 3.0, 4.0])
        beta = np.array([6.0, 9.0, 12.0])
        
        # Compute focus index
        focus_index = compute_focus_index(theta, alpha, beta)
        
        # Check that the focus index has the expected length
        assert len(focus_index) == 3
        
        # Check that the focus index values are correct
        expected = [6.0 / (1.0 + 2.0), 9.0 / (2.0 + 3.0), 12.0 / (3.0 + 4.0)]
        for i in range(3):
            assert focus_index[i] == pytest.approx(expected[i])

    def test_compute_focus_index_empty_input(self):
        """Test focus index computation with empty input."""
        # Create empty test data
        theta = []
        alpha = []
        beta = []
        
        # Compute focus index
        focus_index = compute_focus_index(theta, alpha, beta)
        
        # Check that the focus index is empty
        assert len(focus_index) == 0

    def test_compute_focus_index_different_lengths(self):
        """Test focus index computation with inputs of different lengths."""
        # Create test data with different lengths
        theta = [1.0, 2.0]
        alpha = [2.0, 3.0, 4.0]
        beta = [6.0, 9.0, 12.0, 15.0]
        
        # This should raise a ValueError or similar
        with pytest.raises(Exception):
            compute_focus_index(theta, alpha, beta)

    def test_compute_focus_index_negative_values(self):
        """Test focus index computation with negative values."""
        # Create test data with negative values
        theta = [-1.0, 2.0, 3.0]
        alpha = [2.0, -3.0, 4.0]
        beta = [6.0, 9.0, -12.0]
        
        # Compute focus index
        focus_index = compute_focus_index(theta, alpha, beta)
        
        # Check that the focus index has the expected length
        assert len(focus_index) == 3
        
        # The function should handle negative values
        # The exact behavior depends on the implementation
        # Here we just check that the values are finite
        for value in focus_index:
            assert np.isfinite(value)


class TestFocusIndexInterpretation:
    """Test class for focus index interpretation."""

    def test_focus_index_thresholds(self):
        """Test focus index thresholds for different attention levels."""
        # Define thresholds for different attention levels
        LOW_THRESHOLD = 0.5
        HIGH_THRESHOLD = 2.0
        
        # Create test data for different attention levels
        low_focus = 0.3
        medium_focus = 1.0
        high_focus = 3.0
        
        # Check that the thresholds correctly classify the attention levels
        assert low_focus < LOW_THRESHOLD  # Low attention
        assert LOW_THRESHOLD <= medium_focus < HIGH_THRESHOLD  # Medium attention
        assert high_focus >= HIGH_THRESHOLD  # High attention

    def test_focus_index_time_series(self):
        """Test focus index computation for a time series."""
        # Create a time series of band powers
        # Simulate increasing focus over time
        time_points = 10
        theta = np.linspace(3.0, 1.0, time_points)  # Decreasing theta
        alpha = np.linspace(4.0, 2.0, time_points)  # Decreasing alpha
        beta = np.linspace(6.0, 12.0, time_points)  # Increasing beta
        
        # Compute focus index
        focus_index = compute_focus_index(theta, alpha, beta)
        
        # Check that the focus index increases over time
        for i in range(1, time_points):
            assert focus_index[i] > focus_index[i-1]

    def test_focus_index_stability(self):
        """Test focus index stability with small variations in band powers."""
        # Create base band powers
        theta_base = 2.0
        alpha_base = 3.0
        beta_base = 9.0
        
        # Compute base focus index
        base_focus = beta_base / (theta_base + alpha_base)
        
        # Create variations of the band powers
        variations = 10
        theta_var = np.random.normal(theta_base, 0.1, variations)
        alpha_var = np.random.normal(alpha_base, 0.1, variations)
        beta_var = np.random.normal(beta_base, 0.1, variations)
        
        # Compute focus index for variations
        focus_var = compute_focus_index(theta_var, alpha_var, beta_var)
        
        # Check that the variations are within a reasonable range of the base focus
        for focus in focus_var:
            assert abs(focus - base_focus) < 0.5  # Arbitrary threshold


if __name__ == "__main__":
    pytest.main(["-v", __file__])
