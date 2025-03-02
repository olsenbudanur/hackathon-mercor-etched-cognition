#!/usr/bin/env python
# coding: utf-8

"""
EEG Signal Processing Module

This module provides functions for processing EEG signals, computing frequency band powers,
and calculating focus/attention metrics. It can connect to EEG devices through Lab Streaming
Layer (LSL) protocol and process real-time EEG data.

The module includes functionality for:
- Connecting to EEG devices via LSL
- Collecting and processing EEG data
- Computing frequency band powers (theta, alpha, beta, gamma)
- Calculating focus/attention metrics
- Visualizing EEG data and focus levels in real-time
- Exposing focus metrics via a FastAPI endpoint
"""

import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
import time
from scipy.signal import welch
from IPython.display import clear_output

# Parameters
FS = 256  # EEG sampling rate (update based on device)
CHUNK_LENGTH = 12
LSL_SCAN_TIMEOUT = 5
WINDOW_SIZE = FS * 2  # 2-second window
STEP_SIZE = FS  # 1-second step

# Frequency Bands
BANDS = {
    "theta": (4, 7),
    "alpha": (8, 13),
    "beta": (14, 30),
    "gamma": (31, 50),
    "sixty": (55, 65),
}

def compute_band_power(data, fs, band):
    """
    Compute the power in a specific frequency band of an EEG signal.
    
    This function calculates the power spectral density (PSD) of the input signal
    using Welch's method and then integrates the PSD over the specified frequency band.
    
    Args:
        data (numpy.ndarray): The EEG signal time series
        fs (int): Sampling frequency of the EEG signal in Hz
        band (tuple): Frequency band as a tuple of (low_freq, high_freq) in Hz
    
    Returns:
        float: The power in the specified frequency band
    
    Example:
        ```python
        # Compute alpha band power (8-13 Hz) for an EEG channel
        alpha_power = compute_band_power(eeg_data[0], 256, (8, 13))
        ```
    """
    freqs, psd = welch(data, fs, nperseg=fs)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapz(psd[band_mask], freqs[band_mask])

# Resolve EEG stream
print("Looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)
if not streams:
    raise RuntimeError("EEG Stream not found.")

print("Started acquiring data.")
inlet = StreamInlet(streams[0], max_chunklen=CHUNK_LENGTH)
info = inlet.info()
NCHAN = info.channel_count()
CH_NAMES = ['TP9', 'AF7', 'AF8', 'TP10', 'AUX'][:NCHAN]

def collect_data(recording_time=30):
    """
    Collect EEG data from the LSL stream for a specified duration.
    
    This function collects EEG data from the connected LSL stream for the specified
    recording time. It pulls data chunks from the stream and accumulates them into
    arrays for further processing.
    
    Args:
        recording_time (float, optional): Duration to collect data in seconds. Defaults to 30.
    
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Collected EEG data with shape (channels, samples)
            - numpy.ndarray: Timestamps for each sample
    
    Example:
        ```python
        # Collect 5 seconds of EEG data
        eeg_data, timestamps = collect_data(recording_time=5)
        print(f"Collected {eeg_data.shape[1]} samples from {eeg_data.shape[0]} channels")
        ```
    """
    start_time = time.time()
    all_data, all_timestamps = [], []
    while time.time() - start_time < recording_time:
        data, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=CHUNK_LENGTH)
        if timestamps:
            all_data.extend(data)
            all_timestamps.extend(timestamps)
    return np.array(all_data).T, np.array(all_timestamps)

def process_data(all_data):
    """
    Process EEG data to extract frequency band powers.
    
    This function processes the collected EEG data by dividing it into overlapping
    windows and computing the power in each frequency band (theta, alpha, beta, gamma, etc.)
    for each window.
    
    Args:
        all_data (numpy.ndarray): EEG data with shape (channels, samples)
    
    Returns:
        dict: A dictionary where keys are frequency band names and values are lists of
              band powers for each window
    
    Example:
        ```python
        # Process collected EEG data
        eeg_data, _ = collect_data(recording_time=5)
        band_powers = process_data(eeg_data)
        print(f"Alpha band powers: {band_powers['alpha']}")
        ```
    """
    band_levels = {key: [] for key in BANDS}
    for start in range(0, all_data.shape[1] - WINDOW_SIZE, STEP_SIZE):
        window_data = all_data[:, start:start + WINDOW_SIZE]
        for band_name, band_range in BANDS.items():
            band_levels[band_name].append(
                np.mean([compute_band_power(ch_data, FS, band_range) for ch_data in window_data])
            )
    return band_levels

def compute_focus_index(theta, alpha, beta):
    """
    Compute a focus/attention index based on EEG frequency band powers.
    
    This function calculates a focus index by taking the ratio of beta power to the
    sum of alpha and theta powers. Higher values indicate higher focus/attention levels.
    
    Args:
        theta (list or numpy.ndarray): Theta band powers
        alpha (list or numpy.ndarray): Alpha band powers
        beta (list or numpy.ndarray): Beta band powers
    
    Returns:
        list: Focus index values for each input data point
    
    Example:
        ```python
        # Compute focus index from band powers
        eeg_data, _ = collect_data(recording_time=5)
        band_powers = process_data(eeg_data)
        focus_index = compute_focus_index(
            band_powers['theta'], 
            band_powers['alpha'], 
            band_powers['beta']
        )
        print(f"Focus index: {focus_index}")
        ```
    """
    epsilon = 1e-6  # Avoid division by zero
    return (np.array(beta) / (np.array(alpha) + np.array(theta) + epsilon)).tolist()

# Thresholds (adjust as needed)
THRESHOLD_GAMMA = 8
THRESHOLD_SIXTY = 250

# Live Plotting
timestamps, focus_levels, target_focus_levels = [], [], []
start_time = time.time()
focus_or_not = True
while True:
    focus_or_not = not focus_or_not
    for _ in range(15):
        target_focus = 14 if focus_or_not else 5
        current_time = time.time() - start_time
        
        # Collect and process data
        new_data, new_timestamps = collect_data(2)
        band_levels = process_data(new_data)
        focus_level = np.mean(band_levels["gamma"])
        
        # Determine shared variable
        focus_val = 2 if focus_level > THRESHOLD_GAMMA * 2 else 1 if focus_level > THRESHOLD_GAMMA else 0
        print(focus_val)
        
        # Update plot data
        timestamps.append(current_time)
        focus_levels.append(focus_level)
        target_focus_levels.append(target_focus)
        
        # Clear and redraw plot
        clear_output(wait=True)
        fig, axes = plt.subplots(2, 1, figsize=(10, 4))
        
        # Focus Level Plot
        axes[0].plot(timestamps, focus_levels, label="Focus Level")
        axes[0].plot(timestamps, target_focus_levels, label="Target Focus Level", linestyle='dashed')
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Focus Level")
        axes[0].legend()
        axes[0].set_title("Focus Level Over Time")
        
        # EEG Data Plot
        for i in range(NCHAN):
            axes[1].plot(new_timestamps - new_timestamps[0], new_data[i] + i * 100, label=CH_NAMES[i])
        
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("EEG Signal")
        axes[1].set_title("EEG Data Over 10 Seconds")
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()


# In[ ]:


import nest_asyncio
import uvicorn
import threading
from fastapi import FastAPI
import asyncio

# Apply nest_asyncio to allow running FastAPI inside Jupyter
nest_asyncio.apply()

app = FastAPI()

# Global variable to be updated
shared_var = 0

# Async function to simulate real-time updates
async def update_var():
    """
    Asynchronous function to update shared variables in real-time.
    
    This function runs in the background and periodically updates shared variables
    that can be accessed by FastAPI endpoints. It's designed to run as a background
    task in the FastAPI application.
    
    Note:
        The function is currently configured to sleep for 1 second between updates,
        but the actual update logic is commented out.
    
    Example:
        ```python
        # Start the update task in the background
        asyncio.create_task(update_var())
        ```
    """
    global shared_var
    while True:
        await asyncio.sleep(1)  # Update every second
        #shared_var += 1

@app.get("/latest_value")
async def get_latest_value():
    """
    FastAPI endpoint to return the latest focus value.
    
    This endpoint returns the current focus value calculated from the EEG data.
    It can be accessed via HTTP GET request to /latest_value.
    
    Returns:
        dict: A dictionary containing the current focus value
              - shared_var (int): The current focus value (0, 1, or 2)
    
    Example:
        ```python
        import requests
        
        response = requests.get("http://127.0.0.1:8000/latest_value")
        print(response.json())  # {"shared_var": 1}
        ```
    """
    return {"shared_var": focus_val}

@app.on_event("startup")
async def startup():
    """
    Function that runs when the FastAPI application starts up.
    
    This function is called when the FastAPI application starts. It initializes
    background tasks like the update_var function.
    
    Example:
        This function is automatically called by FastAPI when the application starts.
        It doesn't need to be called manually.
    """
    asyncio.create_task(update_var())

def run_server():
    """
    Function to run the FastAPI server.
    
    This function starts the Uvicorn server to serve the FastAPI application.
    It's designed to be run in a separate thread to avoid blocking the main thread.
    
    Example:
        ```python
        # Run the server in a separate thread
        import threading
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        ```
    """
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Run the server in a separate thread so it doesn't block Jupyter
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

