#!/usr/bin/env python3
"""
EEG-Enhanced Language Model Demo with Token Streaming
----------------------------------------------------
This script provides a unified demo of the EEG-controlled language model system
with real-time token streaming to the backend API.
"""

import time
import argparse
import os
import matplotlib.pyplot as plt
from main_eeg import EEGEnhancedLLM, SYSTEM_PROMPT
import numpy as np
import threading
import keyboard
import json
import logging
import token_streamer  # Import our token streamer module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('unified_demo')

# Demo prompts to showcase capabilities
DEMO_PROMPTS = [
    "Explain quantum computing to me.",
    "What are the key concepts in machine learning?",
    "How do neural networks work?",
    "What is the significance of EEG signals in brain-computer interfaces?",
    "Explain the concept of consciousness.",
    "What are the ethical implications of AI?",
]

class UnifiedEEGDemo:
    """
    Unified EEG-Enhanced Language Model Demo with Token Streaming
    
    This class provides a complete demo experience for the EEG-controlled
    language model system with real-time token streaming to the backend API.
    """
    
    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        use_real_eeg: bool = False,
        enable_visualization: bool = True,
        api_url: str = "http://localhost:8000", 
        verbose: bool = True
    ):
        """
        Initialize the demo.
        
        Args:
            model_path: Path to the language model
            use_real_eeg: Whether to use real EEG data from Muse headset
            enable_visualization: Whether to enable visualization
            api_url: URL of the backend API for token streaming
            verbose: Whether to print verbose output
        """
        self.model_path = model_path
        self.use_real_eeg = use_real_eeg
        self.enable_visualization = enable_visualization
        self.api_url = api_url
        self.verbose = verbose
        
        # Set up token streamer
        self.token_streamer = token_streamer.init_streamer(
            api_url=api_url,
            batch_size=1  # Stream tokens one by one for real-time experience
        )
        
        # State tracking
        self.llm = None
        self.is_running = False
        self.attention_level = 0.5  # Default middle attention
        self.attention_history = []
        self.expert_weights_history = []
        
        # Initialize the language model
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize the EEG-enhanced language model."""
        logger.info("Initializing EEG-enhanced language model...")
        try:
            self.llm = EEGEnhancedLLM(
                model_path=self.model_path,
                simulation_mode=not self.use_real_eeg,
                enable_visualization=self.enable_visualization
            )
            logger.info("Language model initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            if self.verbose:
                print(f"Error initializing model: {e}")
                print("If this is a memory error, try using a smaller model or quantized version.")
            return False
    
    def select_prompt(self):
        """
        Allow the user to select a demo prompt or enter a custom one.
        
        Returns:
            Selected prompt text
        """
        print("\nOptions:")
        print("  1. Try a demo prompt")
        print("  2. Enter your own prompt")
        
        while True:
            choice = input("\nEnter your choice (1-2): ").strip().lower()
            
            if choice == '1':
                # Show demo prompts
                print("\nDemo prompts:")
                for i, prompt in enumerate(DEMO_PROMPTS):
                    print(f"  {i+1}. {prompt}")
                
                try:
                    prompt_idx = int(input("\nSelect a prompt (1-6): ")) - 1
                    if 0 <= prompt_idx < len(DEMO_PROMPTS):
                        user_prompt = DEMO_PROMPTS[prompt_idx]
                        print(f"\nSelected: {user_prompt}")
                        return user_prompt
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            elif choice == '2':
                # Custom prompt
                user_prompt = input("\nEnter your question: ").strip()
                if user_prompt:
                    return user_prompt
                else:
                    print("Please enter a valid prompt.")
            
            else:
                print("Invalid choice. Please try again.")
    
    def print_instructions(self):
        """Print usage instructions for the demo."""
        print("\n" + "="*80)
        print(" "*25 + "EEG-ENHANCED LLM DEMO WITH TOKEN STREAMING")
        print("="*80)
        
        if self.use_real_eeg:
            print("\nUsing REAL EEG data from Muse headset.")
            print("The model will adapt based on your actual attention levels.")
        else:
            print("\nUsing SIMULATED EEG data.")
            print("During generation, use UP/DOWN arrow keys to adjust attention level.")
            print("  UP: Increase attention level (more complex responses)")
            print("  DOWN: Decrease attention level (simpler responses)")
        
        print("\nYour tokens will be streamed in real-time to the backend API.")
        print("You can view them in the frontend visualization.")
        
        print("\nPress [ESC] at any time to stop generation.")
        print("Press [SPACE] to generate the next chunk with current attention level.")
    
    def generate_text_chunk(self, prompt, tokens_to_generate=30):
        """
        Generate a chunk of text based on the current attention level.
        
        Args:
            prompt: The input prompt
            tokens_to_generate: Number of tokens to generate
            
        Returns:
            Generated text and information about the generation
        """
        logger.info(f"Generating text chunk ({tokens_to_generate} tokens)")
        
        # Store the starting token count
        start_token_count = self.llm.get_generation_stats().get('tokens_generated', 0)
        
        # Retrieve the current attention level from the EEG processor
        # For simulated mode, this will be the manually set level
        if self.use_real_eeg:
            self.attention_level = self.llm.eeg_processor.get_attention_level()
        
        # Save attention level for history
        self.attention_history.append(self.attention_level)
        
        try:
            # Generate text using the current attention level
            result = self.llm.generate_with_eeg_control(
                prompt, 
                max_new_tokens=tokens_to_generate,
                suppress_output=True  # We'll handle the output ourselves
            )
            
            # Get the number of tokens actually generated
            end_token_count = self.llm.get_generation_stats().get('tokens_generated', 0)
            tokens_generated = end_token_count - start_token_count
            
            # Get the current expert weights for visualization
            expert_weights = self.llm.moe_controller.get_expert_weights()
            self.expert_weights_history.append(expert_weights)
            
            # Determine the dominant expert based on current weights
            dominant_expert = "unknown"
            if expert_weights:
                max_weight = 0
                for expert, weight in expert_weights.items():
                    if weight > max_weight:
                        max_weight = weight
                        dominant_expert = expert
            
            # Stream each token to the backend with its corresponding expert
            if hasattr(self.llm, 'last_tokens') and self.llm.last_tokens:
                # Make sure we have both tokens and experts
                if hasattr(self.llm, 'last_experts') and len(self.llm.last_experts) == len(self.llm.last_tokens):
                    for i, token_id in enumerate(self.llm.last_tokens):
                        # Get the text representation of the token
                        token_text = self.llm.tokenizer.decode([token_id])
                        # Get the expert used for this specific token
                        token_expert = self.llm.last_experts[i] if i < len(self.llm.last_experts) else dominant_expert
                        
                        # Stream to backend
                        token_streamer.add_token(token_text, token_expert)
                        if self.verbose:
                            print(f"Streamed token: '{token_text}' (Expert: {token_expert})")
                else:
                    # Fallback if experts weren't tracked properly
                    logger.warning("Token experts not tracked properly. Using dominant expert for all tokens.")
                    for token_id in self.llm.last_tokens:
                        token_text = self.llm.tokenizer.decode([token_id])
                        token_streamer.add_token(token_text, dominant_expert)
                        if self.verbose:
                            print(f"Streamed token: '{token_text}' (Expert: {dominant_expert})")
            
            return {
                "text": result,
                "tokens_generated": tokens_generated,
                "attention_level": self.attention_level,
                "expert_weights": expert_weights,
                "dominant_expert": dominant_expert
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return {
                "text": prompt,
                "tokens_generated": 0,
                "attention_level": self.attention_level,
                "expert_weights": {},
                "dominant_expert": "unknown",
                "error": str(e)
            }
    
    def run_demo(self):
        """Run the interactive demo."""
        self.print_instructions()
        
        # Let user select a prompt
        user_prompt = self.select_prompt()
        full_prompt = SYSTEM_PROMPT + f"\n\nUser: {user_prompt}\n\nAssistant: "
        
        # Initialize generation variables
        current_text = full_prompt
        response_text = ""
        chunk_size = 25  # Tokens to generate per chunk
        
        # Set up keyboard handling for attention control
        self.is_running = True
        generation_active = True
        self.attention_level = 0.5  # Start with middle attention
        
        print("\nGenerating response...")
        print("-" * 80)
        
        try:
            while self.is_running and generation_active:
                # Generate the next chunk of text
                result = self.generate_text_chunk(current_text, tokens_to_generate=chunk_size)
                
                # Update the current text with the generated chunk
                current_text = result["text"]
                
                # Extract just the response part (after "Assistant: ")
                if "Assistant: " in current_text:
                    response_text = current_text.split("Assistant: ")[-1]
                else:
                    response_text = current_text
                
                # Display the response and stats
                os.system('cls' if os.name == 'nt' else 'clear')
                print("\nGenerated response:")
                print("-" * 80)
                print(response_text)
                print("-" * 80)
                
                # Display generation stats
                print(f"\nAttention level: {self.attention_level:.2f}")
                print(f"Tokens generated: {result['tokens_generated']}")
                print(f"Dominant expert: {result['dominant_expert']}")
                
                if not self.use_real_eeg:
                    print("\nControls:")
                    print("  UP/DOWN arrows: Adjust attention level")
                    print("  SPACE: Generate next chunk")
                    print("  ESC: Stop generation")
                
                # Wait for user input before continuing
                if not self.use_real_eeg:
                    print("\nPress SPACE to continue or ESC to stop...")
                    
                    while True:
                        if keyboard.is_pressed('space'):
                            break
                        if keyboard.is_pressed('escape'):
                            generation_active = False
                            break
                        if keyboard.is_pressed('up'):
                            self.attention_level = min(1.0, self.attention_level + 0.1)
                            print(f"\rAttention level: {self.attention_level:.2f}", end="")
                            time.sleep(0.1)
                        if keyboard.is_pressed('down'):
                            self.attention_level = max(0.0, self.attention_level - 0.1)
                            print(f"\rAttention level: {self.attention_level:.2f}", end="")
                            time.sleep(0.1)
                        time.sleep(0.05)
                else:
                    # For real EEG mode, we just add a small delay
                    time.sleep(1)
                    # Check for escape key to stop
                    if keyboard.is_pressed('escape'):
                        generation_active = False
            
            # Offer to save the results
            if input("\nSave results to file? (y/n): ").lower().startswith('y'):
                self.save_results_to_file(user_prompt, response_text)
        
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user.")
        
        finally:
            # Clean up resources
            self.is_running = False
            if self.llm:
                self.llm.cleanup()
            
            # Stop the token streamer
            token_streamer.stop_streamer()
            
            print("\nDemo session ended.")
    
    def save_results_to_file(self, prompt, response):
        """Save the generated results and statistics to a file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/demo_results_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Collect stats
        stats = self.llm.get_generation_stats() if self.llm else {}
        
        # Prepare data to save
        data = {
            "prompt": prompt,
            "response": response,
            "timestamp": timestamp,
            "model": self.model_path,
            "use_real_eeg": self.use_real_eeg,
            "attention_history": self.attention_history,
            "expert_weights_history": [
                {expert: float(weight) for expert, weight in weights.items()} 
                for weights in self.expert_weights_history
            ],
            "stats": {
                k: (float(v) if isinstance(v, (float, np.float32, np.float64)) else v)
                for k, v in stats.items()
            },
            "streaming_stats": self.token_streamer.get_stats()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {filename}")

def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(description="EEG-Enhanced Language Model Demo with Token Streaming")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Path to the language model")
    parser.add_argument("--real-eeg", action="store_true", 
                        help="Use real EEG data from Muse headset instead of simulation")
    parser.add_argument("--no-visualization", action="store_true",
                        help="Disable visualization")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                        help="URL of the backend API for token streaming")
    
    args = parser.parse_args()
    
    # Create and run the demo
    demo = UnifiedEEGDemo(
        model_path=args.model,
        use_real_eeg=args.real_eeg,
        enable_visualization=not args.no_visualization,
        api_url=args.api_url
    )
    
    demo.run_demo()

if __name__ == "__main__":
    main() 