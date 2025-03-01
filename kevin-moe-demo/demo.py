#!/usr/bin/env python3
"""
EEG-Enhanced Language Model Demo
--------------------------------
This script demonstrates the complete EEG-controlled language model system
with Mixture-of-Experts routing based on real-time attention signals.

It supports token streaming to the frontend for visualization and
chunk-based generation for better control over attention updates.
"""

import time
import argparse
import os
import traceback
import numpy as np
import sys

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

# Add the tests directory to the path for importing test modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests'))

try:
    from test_eeg_simulation import ControlledEEGProcessor
except ImportError:
    # Define a fallback ControlledEEGProcessor if the import fails
    from eeg_processor import EEGProcessor
    
    class ControlledEEGProcessor(EEGProcessor):
        """Fallback implementation if the real one can't be imported"""
        def __init__(self, pattern="sine", **kwargs):
            super().__init__(simulation_mode=True, **kwargs)
            self.pattern = pattern
            self.counter = 0
        
        def get_attention_level(self):
            """Get current attention level based on the pattern"""
            if self.pattern == "sine":
                # Sine wave oscillating between 0.2 and 0.8
                return 0.5 + 0.3 * np.sin(self.counter / 10)
            elif self.pattern == "step":
                # Step function alternating between 0.2 and 0.8
                return 0.2 if (self.counter // 20) % 2 == 0 else 0.8
            elif self.pattern == "random":
                # Random values with some temporal coherence
                if self.counter % 10 == 0:
                    self._random_value = np.random.uniform(0.2, 0.8)
                return self._random_value
            elif self.pattern == "increasing":
                # Gradually increasing from 0.2 to 0.8
                cycle_position = (self.counter % 100) / 100
                return 0.2 + 0.6 * cycle_position
            else:
                # Default to sine wave
                return 0.5 + 0.3 * np.sin(self.counter / 10)
        
        def get_raw_eeg(self):
            """Get simulated raw EEG data"""
            # Increment counter for time-varying patterns
            self.counter += 1
            return np.random.normal(0, 1, size=(4, 256))

from eeg_processor import EEGProcessor, HTTPEEGProcessor
from main_eeg import EEGEnhancedLLM
from token_streamer import TokenStreamer

# Demo prompts to showcase capabilities
DEMO_PROMPTS = [
    "Explain quantum computing to me.",
    "What are the key concepts in machine learning?",
    "How do neural networks work?",
    "What is the significance of EEG signals in brain-computer interfaces?",
    "Explain the concept of consciousness.",
    "What are the ethical implications of AI?",
]

# Available simulation patterns for the demo
SIMULATION_PATTERNS = {
    "sine": {
        "description": "Sine wave oscillating between low and high attention"
    },
    "step": {
        "description": "Step function alternating between low and high attention"
    },
    "random": {
        "description": "Random attention values"
    },
    "increasing": {
        "description": "Gradually increasing attention from low to high"
    }
}

def run_interactive_demo(
    model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    use_http_eeg=False, 
    eeg_url=None,
    token_chunk_size=1,
    backend_url="http://localhost:8000",
    enable_visualization=True,
    debug_eeg=False,
    max_new_tokens=150,
    simulation_pattern=None,
    non_interactive=False
):
    """
    Run the EEG-Enhanced LLM demo interactively
    
    Args:
        model_path: Path to the language model
        use_http_eeg: Whether to use HTTP EEG server
        eeg_url: URL for the HTTP EEG server
        token_chunk_size: Number of tokens to generate in each chunk
        backend_url: URL for the token streaming backend
        enable_visualization: Whether to enable visualization
        debug_eeg: Whether to output debug info from EEG processor
        max_new_tokens: Maximum number of new tokens to generate
        simulation_pattern: Pattern to use for simulated attention
        non_interactive: Whether to run in non-interactive mode (for testing)
    """
    print("\n" + "=" * 80)
    print(" " * 30 + "EEG-ENHANCED LLM DEMO")
    print("=" * 80 + "\n")
    
    print("Initializing system...")
    
    # Initialize token streamer if backend URL is provided
    token_streamer = None
    if backend_url:
        try:
            print(f"Initializing token streamer to {backend_url}")
            token_streamer = TokenStreamer(backend_url)
        except Exception as e:
            print(f"Error initializing token streamer: {e}")
            print("Token streaming will be disabled")
            token_streamer = None
    
    # Setup EEG processor
    if use_http_eeg and eeg_url:
        print(f"Using HTTP EEG processor with API URL: {eeg_url}")
        if debug_eeg:
            print("Debug output enabled for EEG processor")
        eeg_processor = HTTPEEGProcessor(api_url=eeg_url, debug_output=debug_eeg)
    elif simulation_pattern and simulation_pattern in SIMULATION_PATTERNS:
        pattern_info = SIMULATION_PATTERNS[simulation_pattern]
        print(f"Using simulated attention pattern: {simulation_pattern}")
        print(f"Pattern description: {pattern_info['description']}")
        eeg_processor = ControlledEEGProcessor(pattern_type=simulation_pattern)
    else:
        eeg_processor = None
        
    # Initialize the EEG-enhanced language model
    llm = EEGEnhancedLLM(
        model_path=model_path,
        simulation_mode=(eeg_processor is None),
        enable_visualization=enable_visualization,
        eeg_debug_output=debug_eeg,
        max_new_tokens=max_new_tokens
    )
    
    # Replace the default EEG processor if we have a custom one
    if eeg_processor:
        # Stop the default processor first
        if hasattr(llm, 'eeg_processor') and llm.eeg_processor:
            llm.eeg_processor.stop()
        
        # Replace with our custom processor
        llm.eeg_processor = eeg_processor
        llm.eeg_processor.start()
    
    # Set demo prompts
    demo_prompts = [
        "Explain quantum computing to me.",
        "What are the key concepts in machine learning?",
        "How do neural networks work?",
        "What is the significance of EEG signals in brain-computer interfaces?",
        "Explain the concept of consciousness.",
        "What are the ethical implications of AI?"
    ]
    
    print("\nSystem ready! Using token chunk size of", token_chunk_size)
    if llm.simulation_mode:
        print("Attention level is following a sine pattern.")
    else:
        print("Attention level is being read from EEG input.")
    print("The model will adapt its responses based on the detected attention level.")
    
    # If non-interactive mode is enabled, just run a demo prompt and exit
    if non_interactive:
        print("\nRunning in non-interactive mode with demo prompt...")
        prompt = demo_prompts[1]  # Use the machine learning prompt
        print(f"\nPrompt: {prompt}\n")
        response = llm.generate_with_eeg_control(
            prompt,
            max_new_tokens=max_new_tokens,
            token_chunk_size=token_chunk_size,
            token_streamer=token_streamer
        )
        print("\nFinal response:")
        print(response)
        
        # Clean up
        if llm.eeg_processor:
            print("EEG processor stopped.")
            llm.eeg_processor.stop()
        if token_streamer:
            token_streamer.stop()
        print("\nDemo session ended.")
        return
    
    # Interactive menu loop
    running = True
    while running:
        print("\nOptions:")
        print("  1. Try a demo prompt")
        print("  2. Enter your own prompt")
        print("  3. Run automated demo")
        print("  4. View system statistics")
        print("  5. Change attention simulation pattern")
        print("  q. Quit")
        
        try:
            choice = input("\nEnter your choice (1-5, q): ").strip().lower()
            
            if choice == 'q':
                running = False
            elif choice == '1':
                print("\nDemo prompts:")
                for i, prompt in enumerate(demo_prompts):
                    print(f"  {i+1}. {prompt}")
                
                try:
                    prompt_idx = int(input("\nSelect a prompt (1-6): ")) - 1
                    if 0 <= prompt_idx < len(demo_prompts):
                        prompt = demo_prompts[prompt_idx]
                        print(f"\nPrompt: {prompt}\n")
                        response = llm.generate_with_eeg_control(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            token_chunk_size=token_chunk_size,
                            token_streamer=token_streamer
                        )
                        print("\nFinal response:")
                        print(response)
                    else:
                        print("Invalid prompt selection.")
                except (ValueError, IndexError, EOFError) as e:
                    print(f"Error: {e}. Please enter a valid number.")
                
            elif choice == '2':
                try:
                    prompt = input("\nEnter your prompt: ")
                    if prompt:
                        print(f"\nPrompt: {prompt}\n")
                        response = llm.generate_with_eeg_control(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            token_chunk_size=token_chunk_size,
                            token_streamer=token_streamer
                        )
                        print("\nFinal response:")
                        print(response)
                    else:
                        print("Empty prompt. Please enter a valid prompt.")
                except EOFError:
                    print("Error reading input. Please try again.")
                
            elif choice == '3':
                print("\nRunning automated demo...")
                for i, prompt in enumerate(demo_prompts[:2]):  # Just run first two for automated demo
                    print(f"\nPrompt {i+1}: {prompt}\n")
                    response = llm.generate_with_eeg_control(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        token_chunk_size=token_chunk_size,
                        token_streamer=token_streamer
                    )
                    print("\nResponse:")
                    print(response)
                    print("\n" + "-" * 40)
                    
            elif choice == '4':
                stats = llm.get_generation_stats()
                print("\nSystem Statistics:")
                print(f"  Tokens Generated: {stats.get('tokens_generated', 0)}")
                print(f"  Average Attention: {stats.get('avg_attention', 0.0):.2f}")
                
                if 'expert_usage' in stats and stats['expert_usage']:
                    print("\nExpert Usage:")
                    for expert, count in stats['expert_usage'].items():
                        print(f"  {expert}: {count} tokens")
                
                if 'attention_distribution' in stats:
                    print("\nAttention Distribution:")
                    dist = stats['attention_distribution']
                    print(f"  Low: {dist.get('low', 0.0):.2f}")
                    print(f"  Medium: {dist.get('medium', 0.0):.2f}")
                    print(f"  High: {dist.get('high', 0.0):.2f}")
                    
            elif choice == '5':
                print("\nAvailable attention simulation patterns:")
                for i, (pattern, info) in enumerate(SIMULATION_PATTERNS.items()):
                    print(f"  {i+1}. {pattern}: {info['description']}")
                
                try:
                    pattern_idx = int(input("\nSelect a pattern (1-4): ")) - 1
                    patterns = list(SIMULATION_PATTERNS.keys())
                    
                    if 0 <= pattern_idx < len(patterns):
                        selected_pattern = patterns[pattern_idx]
                        
                        # Stop the current processor
                        if llm.eeg_processor:
                            llm.eeg_processor.stop()
                        
                        # Create a new processor with the selected pattern
                        print(f"Switching to {selected_pattern} attention pattern")
                        new_processor = ControlledEEGProcessor(pattern_type=selected_pattern)
                        
                        # Replace the processor
                        llm.eeg_processor = new_processor
                        llm.eeg_processor.start()
                        
                        print(f"Now using: {SIMULATION_PATTERNS[selected_pattern]['description']}")
                    else:
                        print("Invalid pattern selection.")
                except (ValueError, IndexError, EOFError) as e:
                    print(f"Error: {e}. Please enter a valid number.")
            else:
                print("Invalid choice. Please enter a number from 1-5 or q to quit.")
                
        except EOFError:
            # Handle the case where input is piped and comes to an end
            print("\nInput ended. Exiting demo.")
            running = False
        except KeyboardInterrupt:
            print("\nDemo interrupted by user.")
            running = False
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            # Continue running despite errors
    
    # Clean up
    if llm.eeg_processor:
        print("EEG processor stopped.")
        llm.eeg_processor.stop()
    if token_streamer:
        token_streamer.stop()
    print("\nDemo session ended.")

def parse_args():
    """Parse command line arguments for the demo"""
    parser = argparse.ArgumentParser(description="Run the EEG-Enhanced LLM Demo")
    
    # Model arguments
    parser.add_argument("--model-path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                      help="Path to the language model")
    
    # EEG input options
    parser.add_argument("--http-eeg", action="store_true", 
                      help="Use HTTP EEG server instead of direct Muse connection")
    parser.add_argument("--eeg-url", default=None, 
                      help="URL for the HTTP EEG server, required if --http-eeg is set")
    parser.add_argument("--debug-eeg", action="store_true",
                      help="Enable debug output from EEG processor")
    
    # Simulation pattern
    parser.add_argument("--simulation-pattern", choices=list(SIMULATION_PATTERNS.keys()), default=None,
                      help="Pattern to use for simulated attention")
    
    # Generation parameters
    parser.add_argument("--token-chunk-size", type=int, default=1,
                      help="Number of tokens to generate in each chunk")
    parser.add_argument("--max-tokens", type=int, default=150,
                      help="Maximum number of new tokens to generate")
    
    # Output options
    parser.add_argument("--backend-url", default="http://localhost:8000",
                      help="URL for the token streaming backend")
    parser.add_argument("--no-visualization", action="store_true",
                      help="Disable attention visualization (helps with threading issues)")
    
    # Demo mode
    parser.add_argument("--non-interactive", action="store_true",
                      help="Run in non-interactive mode with a demo prompt (for testing)")
    
    return parser.parse_args()

def main():
    """Main function to run the demo"""
    args = parse_args()
    
    # Set default simulation pattern if none specified
    if not args.simulation_pattern and not args.http_eeg:
        args.simulation_pattern = "sine"
    
    # Check if HTTP EEG URL is provided when HTTP EEG is requested
    if args.http_eeg and not args.eeg_url:
        print("Error: --eeg-url is required when using --http-eeg")
        return 1
    
    try:
        run_interactive_demo(
            model_path=args.model_path,
            use_http_eeg=args.http_eeg,
            eeg_url=args.eeg_url,
            token_chunk_size=args.token_chunk_size,
            backend_url=args.backend_url,
            enable_visualization=not args.no_visualization,
            debug_eeg=args.debug_eeg,
            max_new_tokens=args.max_tokens,
            simulation_pattern=args.simulation_pattern,
            non_interactive=args.non_interactive
        )
        return 0
    except Exception as e:
        print(f"Error in demo: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main() 