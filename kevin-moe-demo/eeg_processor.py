import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import random  # For simulation mode

# Try to import the Muse libraries, but provide fallback if not available
try:
    from muselsl import stream, list_muses
    from pylsl import StreamInlet, resolve_byprop
    MUSE_AVAILABLE = True
except ImportError:
    print("Muse libraries not found, running in simulation mode")
    MUSE_AVAILABLE = False

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