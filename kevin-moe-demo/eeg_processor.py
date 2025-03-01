"""
EEG Signal Processor
-------------------
This module provides EEG signal processing capabilities for the language model demo.
It can either process real EEG signals from a Muse headset or simulate attention
signals for demonstration purposes.
"""

import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import random  # For simulation mode
import requests
import json

# Try to import the Muse libraries, but provide fallback if not available
try:
    from muselsl import stream, list_muses
    from pylsl import StreamInlet, resolve_byprop
    MUSE_AVAILABLE = True
except ImportError:
    print("Muse libraries not found, running in simulation mode")
    MUSE_AVAILABLE = False

class HTTPEEGProcessor:
    """
    Process EEG signals from an HTTP endpoint rather than directly from the Muse headset.
    Provides the same interface as the EEGProcessor class for compatibility.
    """
    
    def __init__(self, api_url="https://f424-216-201-226-138.ngrok-free.app/latest_value", 
                 debug_output=False, smoothing_factor=0.8):
        """
        Initialize the HTTP EEG processor.
        
        Args:
            api_url (str): URL to fetch attention values from
            debug_output (bool): Whether to print debug information
            smoothing_factor (float): Factor for exponential moving average (0-1)
        """
        self.api_url = api_url
        self.debug_output = debug_output
        self.smoothing_factor = smoothing_factor
        
        # State variables
        self.running = False
        self.current_attention = 0.5  # Default middle attention
        self.smoothed_attention = 0.5
        
        # Metrics tracking
        self.attention_history = deque(maxlen=1000)
        self.fetch_times = deque(maxlen=100)
        self.metrics = {
            "mean_attention": 0.5,
            "std_attention": 0.0,
            "num_low_attention_periods": 0,
            "num_high_attention_periods": 0,
            "successful_fetches": 0,
            "failed_fetches": 0,
            "avg_fetch_time_ms": 0
        }
        
        # Thread for continuous fetching
        self.fetch_thread = None
        self.lock = threading.Lock()
    
    def start(self):
        """Start the EEG processor fetch thread"""
        if self.running:
            return
            
        self.running = True
        self.fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.fetch_thread.start()
        
        if self.debug_output:
            print(f"HTTP EEG processor started, fetching from {self.api_url}")
    
    def stop(self):
        """Stop the EEG processor fetch thread"""
        self.running = False
        if self.fetch_thread:
            self.fetch_thread.join(timeout=1.0)
        
        if self.debug_output:
            print("HTTP EEG processor stopped")
    
    def _fetch_loop(self):
        """Background thread to continuously fetch attention values"""
        last_high = False
        last_low = False
        
        while self.running:
            try:
                start_time = time.time()
                
                # Fetch attention value from API
                headers = {"ngrok-skip-browser-warning": "69420"}
                response = requests.get(self.api_url, headers=headers, timeout=2.0)
                
                if response.status_code == 200:
                    # Try to parse as JSON first
                    try:
                        data = response.json()
                        
                        # Handle specific format from ngrok API: {"shared_var": X}
                        if isinstance(data, dict) and "shared_var" in data:
                            expert_value = int(data["shared_var"])
                            # Map expert values to attention levels:
                            # 0 = simple/low attention (0.2)
                            # 1 = balanced/medium attention (0.5) 
                            # 2 = complex/high attention (0.8)
                            attention_map = {0: 0.2, 1: 0.5, 2: 0.8}
                            attention = attention_map.get(expert_value, 0.5)
                            
                            if self.debug_output:
                                print(f"HTTP EEG: Mapped expert value {expert_value} to attention {attention:.2f}")
                        
                        # Check for other known formats
                        elif isinstance(data, dict) and "attention" in data:
                            attention = float(data["attention"])
                        elif isinstance(data, dict) and "value" in data:
                            attention = float(data["value"])
                        else:
                            # Try to get the first numeric value
                            for key, value in data.items():
                                if isinstance(value, (int, float)):
                                    attention = float(value)
                                    # If value is between 0-2, assume it's an expert indicator
                                    if 0 <= value <= 2:
                                        attention_map = {0: 0.2, 1: 0.5, 2: 0.8}
                                        attention = attention_map.get(int(value), 0.5)
                                    break
                            else:
                                attention = float(data) if isinstance(data, (int, float)) else 0.5
                    except json.JSONDecodeError:
                        # Not JSON, try to parse as plain text
                        try:
                            attention = float(response.text.strip())
                        except ValueError:
                            attention = 0.5
                    
                    # Normalize to [0, 1] if needed (skip for values we've already mapped)
                    if attention > 1.0 and not any(attention == val for val in [0.2, 0.5, 0.8]):
                        attention = attention / 100.0
                    
                    # Clamp to valid range
                    attention = max(0.0, min(1.0, attention))
                    
                    # Update metrics
                    self.metrics["successful_fetches"] += 1
                    
                    # Apply smoothing
                    with self.lock:
                        if len(self.attention_history) == 0:
                            self.smoothed_attention = attention
                        else:
                            self.smoothed_attention = (
                                self.smoothing_factor * self.smoothed_attention + 
                                (1 - self.smoothing_factor) * attention
                            )
                        self.current_attention = self.smoothed_attention
                        self.attention_history.append(self.current_attention)
                    
                    # Track high/low attention periods
                    if self.current_attention > 0.7 and not last_high:
                        self.metrics["num_high_attention_periods"] += 1
                        last_high = True
                    else:
                        last_high = False
                        
                    if self.current_attention < 0.3 and not last_low:
                        self.metrics["num_low_attention_periods"] += 1
                        last_low = True
                    else:
                        last_low = False
                    
                    # Calculate attention statistics
                    if len(self.attention_history) > 0:
                        attention_array = np.array(list(self.attention_history))
                        self.metrics["mean_attention"] = float(np.mean(attention_array))
                        self.metrics["std_attention"] = float(np.std(attention_array))
                    
                    if self.debug_output:
                        print(f"HTTP EEG: Attention = {self.current_attention:.2f}")
                
                else:
                    self.metrics["failed_fetches"] += 1
                    if self.debug_output:
                        print(f"HTTP EEG: Failed to fetch data, status code {response.status_code}")
            
            except Exception as e:
                self.metrics["failed_fetches"] += 1
                if self.debug_output:
                    print(f"HTTP EEG: Error fetching data: {str(e)}")
            
            # Calculate fetch time
            fetch_time = (time.time() - start_time) * 1000  # in ms
            self.fetch_times.append(fetch_time)
            if len(self.fetch_times) > 0:
                self.metrics["avg_fetch_time_ms"] = sum(self.fetch_times) / len(self.fetch_times)
            
            # Sleep to avoid overwhelming the API
            time.sleep(0.5)
    
    def get_attention_level(self):
        """Get the current attention level (0-1)"""
        with self.lock:
            return self.current_attention
    
    def get_raw_attention(self):
        """Get the current raw attention level (identical to smoothed in HTTP version)"""
        return self.get_attention_level()
    
    def get_attention_metrics(self):
        """Get metrics about attention levels over time"""
        return self.metrics
    
    def get_latest_eeg_data(self):
        """Placeholder for backwards compatibility with EEGProcessor"""
        return None

class EEGProcessor:
    """Process EEG signals from Muse headset and calculate attention metrics"""
    
    def __init__(self, simulation_mode=False, buffer_size=500, enable_visualization=True, debug_output=False):
        """
        Initialize EEG processor
        
        Args:
            simulation_mode: If True, generate simulated EEG data instead of using a real headset
            buffer_size: Size of signal buffer (number of samples)
            enable_visualization: Whether to show a real-time visualization of the attention level
            debug_output: Whether to print debug information to console
        """
        self.simulation_mode = simulation_mode or not MUSE_AVAILABLE
        self.buffer_size = buffer_size
        self.raw_eeg_buffer = [deque(maxlen=buffer_size) for _ in range(4)]  # 4 EEG channels
        self.alpha_power = [deque(maxlen=buffer_size) for _ in range(4)]
        self.beta_power = [deque(maxlen=buffer_size) for _ in range(4)]
        self.attention_buffer = deque(maxlen=buffer_size)
        self.running = False
        self.inlet = None
        self.process_thread = None
        self.attention_level = 0.5  # Default mid-level attention
        self.attention_history = deque(maxlen=100)  # For visualization
        self.timestamps = deque(maxlen=100)  # For tracking timing
        self.vis_thread = None
        self.visualization = enable_visualization
        self.last_update = time.time()
        self.start_time = None
        self.debug_output = debug_output  # Store debug output preference
        
        # Metrics for analysis
        self.attention_metrics = {
            "mean": 0.5,
            "std": 0.1,
            "min": 0.5,
            "max": 0.5,
            "low_periods": 0,  # Periods of low attention
            "high_periods": 0   # Periods of high attention
        }
        
        # Set up visualization if requested
        if self.visualization:
            self.fig, self.ax = plt.subplots(figsize=(10, 4))
            self.ax.set_ylim(0, 1)
            self.ax.set_xlim(0, 100)
            self.ax.set_title('Real-time Attention Level')
            self.ax.set_xlabel('Time (samples)')
            self.ax.set_ylabel('Attention Level')
            self.line, = self.ax.plot([], [], 'r-')
            
            # Add threshold regions with labels
            self.ax.axhspan(0, 0.3, alpha=0.2, color='blue', label='Low Attention')
            self.ax.axhspan(0.3, 0.7, alpha=0.2, color='green', label='Medium Attention')
            self.ax.axhspan(0.7, 1.0, alpha=0.2, color='red', label='High Attention')
            
            # Add moving average
            self.ma_line, = self.ax.plot([], [], 'k--', alpha=0.7, label='Moving Avg')
            
            self.ax.legend(loc='upper right')
    
    def start(self):
        """Start EEG processing"""
        if self.running:
            return
            
        self.running = True
        self.start_time = time.time()
        
        if not self.simulation_mode:
            # Connect to Muse headset
            self._connect_to_muse()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_eeg)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Start visualization if requested
        if self.visualization:
            self.vis_thread = threading.Thread(target=self._run_visualization)
            self.vis_thread.daemon = True
            self.vis_thread.start()
    
    def stop(self):
        """Stop EEG processing"""
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
        if self.vis_thread:
            self.vis_thread.join(timeout=1.0)
        plt.close('all')
    
    def get_attention_level(self):
        """Get current attention level (0.0 to 1.0)"""
        # Smoothed attention value (exponential moving average)
        return self.attention_level
    
    # Maintain backwards compatibility
    def get_attention(self):
        """Legacy method for backward compatibility"""
        return self.get_attention_level()
    
    def get_attention_metrics(self):
        """Get detailed metrics about attention levels"""
        # Update metrics before returning
        if len(self.attention_buffer) > 0:
            attention_array = np.array(self.attention_buffer)
            self.attention_metrics.update({
                "mean": float(np.mean(attention_array)),
                "std": float(np.std(attention_array)),
                "min": float(np.min(attention_array)),
                "max": float(np.max(attention_array)),
                "low_percentage": float(np.mean(attention_array < 0.3)),
                "medium_percentage": float(np.mean((attention_array >= 0.3) & (attention_array <= 0.7))),
                "high_percentage": float(np.mean(attention_array > 0.7))
            })
        
        return self.attention_metrics
    
    def _connect_to_muse(self):
        """Connect to Muse headset via LSL"""
        try:
            # Find available Muse devices
            muses = list_muses()
            if not muses:
                print("No Muse devices found. Running in simulation mode.")
                self.simulation_mode = True
                return
                
            print(f"Found Muse device: {muses[0]['name']}")
            
            # Start streaming from the first available Muse
            stream(muses[0]['address'])
            
            # Create LSL inlet for EEG data
            eeg_streams = resolve_byprop('type', 'EEG', timeout=5)
            if not eeg_streams:
                print("No EEG stream found. Running in simulation mode.")
                self.simulation_mode = True
                return
                
            print("EEG stream found, creating inlet...")
            self.inlet = StreamInlet(eeg_streams[0])
            print("Connected to Muse headset")
            
        except Exception as e:
            print(f"Error connecting to Muse: {e}")
            print("Running in simulation mode")
            self.simulation_mode = True
    
    def _process_eeg(self):
        """Process EEG data stream and calculate attention metrics"""
        # For tracking attention state changes
        last_attention_state = "medium"  # "low", "medium", "high"
        
        while self.running:
            # Get EEG data (real or simulated)
            if self.simulation_mode:
                # Generate simulated EEG data
                raw_data = self._simulate_eeg_data()
                time.sleep(0.01)  # Simulate 100Hz sampling rate
            else:
                # Get real data from Muse
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                if sample is None:
                    continue
                raw_data = sample[:4]  # Just take the 4 EEG channels
            
            # Store raw data
            for i, val in enumerate(raw_data):
                self.raw_eeg_buffer[i].append(val)
            
            # Calculate frequency band powers (if enough data)
            if len(self.raw_eeg_buffer[0]) >= 100:  # Need at least 1 second of data at 100Hz
                for channel in range(4):
                    # Calculate alpha (8-13 Hz) and beta (13-30 Hz) power
                    # In a real implementation, this would use proper filtering and FFT
                    # For this demo, we'll use the simulated data directly
                    alpha = np.mean(np.abs(raw_data[channel])) 
                    beta = np.std(raw_data[channel])
                    
                    self.alpha_power[channel].append(alpha)
                    self.beta_power[channel].append(beta)
                
                # Calculate attention metric
                # Higher beta/alpha ratio generally indicates higher attention/focus
                if len(self.alpha_power[0]) > 0 and len(self.beta_power[0]) > 0:
                    # Average the beta/alpha ratio across channels
                    ratios = []
                    for channel in range(4):
                        alpha_avg = np.mean(self.alpha_power[channel])
                        beta_avg = np.mean(self.beta_power[channel])
                        if alpha_avg > 0:
                            ratios.append(beta_avg / alpha_avg)
                    
                    if ratios:
                        # Normalize to 0-1 range (sigmoid function)
                        ratio = np.mean(ratios)
                        attention = 1 / (1 + np.exp(-0.5 * (ratio - 1.5)))
                        
                        # Apply smoothing (exponential moving average)
                        alpha_smooth = 0.05  # Smoothing factor
                        self.attention_level = alpha_smooth * attention + (1 - alpha_smooth) * self.attention_level
                        self.attention_buffer.append(self.attention_level)
                        self.attention_history.append(self.attention_level)
                        self.timestamps.append(time.time() - self.start_time)
                        
                        # Track attention state changes for metrics
                        current_state = "medium"
                        if self.attention_level < 0.3:
                            current_state = "low"
                        elif self.attention_level > 0.7:
                            current_state = "high"
                            
                        if current_state != last_attention_state:
                            if current_state == "low":
                                self.attention_metrics["low_periods"] += 1
                            elif current_state == "high":
                                self.attention_metrics["high_periods"] += 1
                            last_attention_state = current_state
                        
                        # Print debug info at most once per second
                        if time.time() - self.last_update > 1.0 and self.debug_output:
                            print(f"Attention level: {self.attention_level:.2f}")
                            self.last_update = time.time()
    
    def _simulate_eeg_data(self):
        """Generate simulated EEG data for testing"""
        # Create a mixture of frequencies to simulate EEG
        # Add in some random walk to simulate changes in attention
        t = time.time()
        
        # Add a slow-changing random influence to simulate changing attention
        if not hasattr(self, 'random_walk'):
            self.random_walk = 0.5
            self.trend_direction = random.choice([-1, 1])
            self.trend_duration = random.uniform(3.0, 10.0)
            self.trend_start = t
        
        # Occasionally change trend direction (more realistic simulation)
        if t - self.trend_start > self.trend_duration:
            self.trend_direction = -self.trend_direction
            self.trend_duration = random.uniform(3.0, 10.0)
            self.trend_start = t
        
        # Update random walk with some bounds and bias
        trend_influence = 0.01 * self.trend_direction
        self.random_walk += random.uniform(-0.02, 0.02) + trend_influence
        self.random_walk = max(0.1, min(0.9, self.random_walk))
        
        # Generate simulated data with:
        # - Alpha waves (8-13 Hz) stronger when relaxed (low attention)
        # - Beta waves (13-30 Hz) stronger when focused (high attention)
        simulated_data = []
        for i in range(4):  # 4 EEG channels
            # Balance of alpha vs beta based on random walk
            alpha_amp = 1.0 - self.random_walk
            beta_amp = self.random_walk
            
            # Generate alpha and beta components
            alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + i)  # 10 Hz alpha
            beta = beta_amp * np.sin(2 * np.pi * 20 * t + i)    # 20 Hz beta
            
            # Add noise
            noise = 0.1 * np.random.normal()
            
            # Combine signals
            signal = alpha + beta + noise
            simulated_data.append(signal)
        
        return simulated_data
    
    def _run_visualization(self):
        """Run the real-time visualization of attention levels"""
        def update(frame):
            # Update the line data
            x = list(self.timestamps) if self.timestamps else list(range(len(self.attention_history)))
            y = list(self.attention_history)
            
            if len(y) > 0:
                self.line.set_data(x, y)
                
                # Update moving average (window size of 10)
                if len(y) >= 10:
                    ma_y = []
                    for i in range(len(y)):
                        if i < 5:
                            # For the first few points, use available data
                            ma_y.append(np.mean(y[:i+6]))
                        elif i > len(y) - 6:
                            # For the last few points, use available data
                            ma_y.append(np.mean(y[i-5:]))
                        else:
                            # Use a centered 11-point moving average
                            ma_y.append(np.mean(y[i-5:i+6]))
                    
                    self.ma_line.set_data(x, ma_y)
                    
                # Update axes if needed
                if x:
                    min_x, max_x = min(x), max(x)
                    padding = (max_x - min_x) * 0.1 if max_x > min_x else 1.0
                    self.ax.set_xlim(min_x - padding, max_x + padding)
                
            return self.line, self.ma_line
        
        # Create animation
        ani = FuncAnimation(self.fig, update, interval=100, blit=True)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    print("Starting EEG processor...")
    eeg = EEGProcessor(simulation_mode=True, enable_visualization=True, debug_output=True)
    eeg.start()
    
    try:
        # Run for 30 seconds
        for _ in range(30):
            attention = eeg.get_attention_level()
            print(f"Current attention: {attention:.2f}")
            
            # Every 10 seconds, print detailed metrics
            if _ % 10 == 0 and _ > 0:
                print("\nAttention Metrics:")
                metrics = eeg.get_attention_metrics()
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                print()
                
            time.sleep(1)
    finally:
        eeg.stop()
        print("EEG processor stopped")