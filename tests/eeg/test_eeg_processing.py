"""
Unit tests for EEG signal processing functionality.

These tests cover the EEG signal processing functions, including band power computation,
data collection, and data processing.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import sys
import os

# Add the eeg directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../eeg')))

# Import the EEG processing functions
from eeg import compute_band_power, collect_data, process_data, BANDS, FS, WINDOW_SIZE, STEP_SIZE


class TestBandPowerComputation:
    """Test class for band power computation."""

    def test_compute_band_power_theta(self):
        """Test computing power in the theta band."""
        # Create synthetic EEG data with known frequency components
        # Theta band: 4-7 Hz
        t = np.linspace(0, 1, FS)  # 1 second of data at FS sampling rate
        theta_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave (in theta band)
        
        # Compute band power
        theta_power = compute_band_power(theta_signal, FS, BANDS["theta"])
        
        # The power should be significant in the theta band
        assert theta_power > 0.1
        
        # Test with a signal outside the theta band
        alpha_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave (in alpha band)
        alpha_in_theta_power = compute_band_power(alpha_signal, FS, BANDS["theta"])
        
        # The power should be very low in the theta band
        assert alpha_in_theta_power < 0.1
        
        # The theta signal should have more power in the theta band than the alpha signal
        assert theta_power > alpha_in_theta_power

    def test_compute_band_power_alpha(self):
        """Test computing power in the alpha band."""
        # Create synthetic EEG data with known frequency components
        # Alpha band: 8-13 Hz
        t = np.linspace(0, 1, FS)  # 1 second of data at FS sampling rate
        alpha_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave (in alpha band)
        
        # Compute band power
        alpha_power = compute_band_power(alpha_signal, FS, BANDS["alpha"])
        
        # The power should be significant in the alpha band
        assert alpha_power > 0.1
        
        # Test with a signal outside the alpha band
        beta_signal = np.sin(2 * np.pi * 20 * t)  # 20 Hz sine wave (in beta band)
        beta_in_alpha_power = compute_band_power(beta_signal, FS, BANDS["alpha"])
        
        # The power should be very low in the alpha band
        assert beta_in_alpha_power < 0.1
        
        # The alpha signal should have more power in the alpha band than the beta signal
        assert alpha_power > beta_in_alpha_power

    def test_compute_band_power_beta(self):
        """Test computing power in the beta band."""
        # Create synthetic EEG data with known frequency components
        # Beta band: 14-30 Hz
        t = np.linspace(0, 1, FS)  # 1 second of data at FS sampling rate
        beta_signal = np.sin(2 * np.pi * 20 * t)  # 20 Hz sine wave (in beta band)
        
        # Compute band power
        beta_power = compute_band_power(beta_signal, FS, BANDS["beta"])
        
        # The power should be significant in the beta band
        assert beta_power > 0.1
        
        # Test with a signal outside the beta band
        gamma_signal = np.sin(2 * np.pi * 40 * t)  # 40 Hz sine wave (in gamma band)
        gamma_in_beta_power = compute_band_power(gamma_signal, FS, BANDS["beta"])
        
        # The power should be very low in the beta band
        assert gamma_in_beta_power < 0.1
        
        # The beta signal should have more power in the beta band than the gamma signal
        assert beta_power > gamma_in_beta_power

    def test_compute_band_power_gamma(self):
        """Test computing power in the gamma band."""
        # Create synthetic EEG data with known frequency components
        # Gamma band: 31-50 Hz
        t = np.linspace(0, 1, FS)  # 1 second of data at FS sampling rate
        gamma_signal = np.sin(2 * np.pi * 40 * t)  # 40 Hz sine wave (in gamma band)
        
        # Compute band power
        gamma_power = compute_band_power(gamma_signal, FS, BANDS["gamma"])
        
        # The power should be significant in the gamma band
        assert gamma_power > 0.1
        
        # Test with a signal outside the gamma band
        beta_signal = np.sin(2 * np.pi * 20 * t)  # 20 Hz sine wave (in beta band)
        beta_in_gamma_power = compute_band_power(beta_signal, FS, BANDS["gamma"])
        
        # The power should be very low in the gamma band
        assert beta_in_gamma_power < 0.1
        
        # The gamma signal should have more power in the gamma band than the beta signal
        assert gamma_power > beta_in_gamma_power

    def test_compute_band_power_mixed_signal(self):
        """Test computing band power with a mixed signal."""
        # Create a mixed signal with multiple frequency components
        t = np.linspace(0, 1, FS)  # 1 second of data at FS sampling rate
        theta_component = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz (theta)
        alpha_component = 1.0 * np.sin(2 * np.pi * 10 * t)  # 10 Hz (alpha)
        beta_component = 0.3 * np.sin(2 * np.pi * 20 * t)  # 20 Hz (beta)
        gamma_component = 0.1 * np.sin(2 * np.pi * 40 * t)  # 40 Hz (gamma)
        
        mixed_signal = theta_component + alpha_component + beta_component + gamma_component
        
        # Compute band powers
        theta_power = compute_band_power(mixed_signal, FS, BANDS["theta"])
        alpha_power = compute_band_power(mixed_signal, FS, BANDS["alpha"])
        beta_power = compute_band_power(mixed_signal, FS, BANDS["beta"])
        gamma_power = compute_band_power(mixed_signal, FS, BANDS["gamma"])
        
        # Check that the band powers reflect the signal composition
        # Alpha should have the highest power (amplitude 1.0)
        # Theta should be next (amplitude 0.5)
        # Beta should be next (amplitude 0.3)
        # Gamma should be lowest (amplitude 0.1)
        assert alpha_power > theta_power
        assert theta_power > beta_power
        assert beta_power > gamma_power


class TestDataCollection:
    """Test class for EEG data collection."""

    @patch('eeg.inlet')
    def test_collect_data(self, mock_inlet):
        """Test collecting EEG data from the LSL stream."""
        # Mock the inlet.pull_chunk method to return synthetic data
        mock_data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_timestamps = [0.0, 0.1, 0.2]
        mock_inlet.pull_chunk.return_value = (mock_data, mock_timestamps)
        
        # Call collect_data with a short recording time
        data, timestamps = collect_data(recording_time=0.1)
        
        # Check that pull_chunk was called
        mock_inlet.pull_chunk.assert_called()
        
        # Check that the data was transposed correctly
        assert data.shape[0] == 2  # 2 channels
        assert data.shape[1] == 3  # 3 samples
        assert np.array_equal(data[0], [0.1, 0.2, 0.3])
        assert np.array_equal(data[1], [0.4, 0.5, 0.6])
        
        # Check that the timestamps were returned correctly
        assert np.array_equal(timestamps, [0.0, 0.1, 0.2])

    @patch('eeg.inlet')
    def test_collect_data_empty(self, mock_inlet):
        """Test collecting EEG data when no data is available."""
        # Mock the inlet.pull_chunk method to return empty data
        mock_inlet.pull_chunk.return_value = ([], [])
        
        # Call collect_data with a short recording time
        data, timestamps = collect_data(recording_time=0.1)
        
        # Check that pull_chunk was called
        mock_inlet.pull_chunk.assert_called()
        
        # Check that empty arrays were returned
        assert data.shape == (0, 0)
        assert timestamps.shape == (0,)

    @patch('eeg.inlet')
    def test_collect_data_multiple_chunks(self, mock_inlet):
        """Test collecting EEG data with multiple chunks."""
        # Mock the inlet.pull_chunk method to return different data on each call
        mock_inlet.pull_chunk.side_effect = [
            ([[0.1, 0.2], [0.3, 0.4]], [0.0, 0.1]),
            ([[0.5, 0.6], [0.7, 0.8]], [0.2, 0.3]),
            ([], [])  # End of data
        ]
        
        # Call collect_data with a longer recording time
        data, timestamps = collect_data(recording_time=0.5)
        
        # Check that pull_chunk was called multiple times
        assert mock_inlet.pull_chunk.call_count > 1
        
        # Check that the data was combined correctly
        assert data.shape[0] == 2  # 2 channels
        assert data.shape[1] == 4  # 4 samples
        assert np.array_equal(data[0], [0.1, 0.2, 0.5, 0.6])
        assert np.array_equal(data[1], [0.3, 0.4, 0.7, 0.8])
        
        # Check that the timestamps were combined correctly
        assert np.array_equal(timestamps, [0.0, 0.1, 0.2, 0.3])


class TestDataProcessing:
    """Test class for EEG data processing."""

    @patch('eeg.compute_band_power')
    def test_process_data(self, mock_compute_band_power):
        """Test processing EEG data to extract frequency band powers."""
        # Mock the compute_band_power function to return known values
        mock_compute_band_power.side_effect = lambda data, fs, band: {
            (4, 7): 0.1,    # theta
            (8, 13): 0.2,   # alpha
            (14, 30): 0.3,  # beta
            (31, 50): 0.4,  # gamma
            (55, 65): 0.5   # sixty
        }[band]
        
        # Create synthetic EEG data
        # 2 channels, 1000 samples (enough for multiple windows)
        data = np.random.rand(2, 1000)
        
        # Process the data
        band_levels = process_data(data)
        
        # Check that compute_band_power was called
        assert mock_compute_band_power.call_count > 0
        
        # Check that band_levels has the expected structure
        assert set(band_levels.keys()) == set(BANDS.keys())
        
        # Check that each band has the expected number of values
        # Number of windows = (data.shape[1] - WINDOW_SIZE) / STEP_SIZE + 1
        expected_windows = (data.shape[1] - WINDOW_SIZE) // STEP_SIZE + 1
        for band_name in BANDS:
            assert len(band_levels[band_name]) == expected_windows
        
        # Check that the band levels have the expected values
        for band_name in BANDS:
            expected_value = {
                "theta": 0.1,
                "alpha": 0.2,
                "beta": 0.3,
                "gamma": 0.4,
                "sixty": 0.5
            }[band_name]
            
            for value in band_levels[band_name]:
                assert value == expected_value

    def test_process_data_real(self):
        """Test processing EEG data with real computation."""
        # Create synthetic EEG data with known frequency components
        t = np.linspace(0, 10, FS * 10)  # 10 seconds of data at FS sampling rate
        
        # Channel 1: Strong alpha (10 Hz)
        ch1 = np.sin(2 * np.pi * 10 * t)
        
        # Channel 2: Strong beta (20 Hz)
        ch2 = np.sin(2 * np.pi * 20 * t)
        
        # Combine channels
        data = np.vstack((ch1, ch2))
        
        # Process the data
        band_levels = process_data(data)
        
        # Check that band_levels has the expected structure
        assert set(band_levels.keys()) == set(BANDS.keys())
        
        # Check that the alpha band has higher power than other bands
        # (since we have a strong 10 Hz component)
        for i in range(len(band_levels["alpha"])):
            assert band_levels["alpha"][i] > band_levels["theta"][i]
            assert band_levels["alpha"][i] > band_levels["gamma"][i]
        
        # Check that the beta band also has significant power
        # (since we have a strong 20 Hz component)
        for i in range(len(band_levels["beta"])):
            assert band_levels["beta"][i] > band_levels["theta"][i]
            assert band_levels["beta"][i] > band_levels["gamma"][i]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
