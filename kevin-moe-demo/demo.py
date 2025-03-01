#!/usr/bin/env python3
"""
Unified EEG-Enhanced Language Model Demo

This demo combines the robustness of the original demo.py with the
user-friendly interface of streamlined_demo.py. It supports chunk-by-chunk
generation with manual attention control for an interactive experience.

Usage:
  python demo.py [--model-path MODEL_PATH] [--real-eeg]
"""

import time
import argparse
import sys
import os
import numpy as np
import threading
import traceback
from main_eeg import EEGEnhancedLLM, SYSTEM_PROMPT

class UnifiedEEGDemo:
    def __init__(self, model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", use_real_eeg=False):
        self.model_path = model_path
        self.use_real_eeg = use_real_eeg
        self.attention_level = 0.5
        self.running = True
        self.llm = None
        self.prompt = ""
        self.current_response = ""
        self.attention_history = []
        self.expert_history = {
            "simple": [],
            "balanced": [],
            "complex": []
        }
        self.suppress_output = True
        self.debug_mode = False
        
    def initialize_llm(self):
        """Initialize the language model with EEG enhancements"""
        print("Loading model... (this may take a moment)")
        
        try:
            # Create the LLM with appropriate settings
            self.llm = EEGEnhancedLLM(
                model_path=self.model_path,
                use_cuda=True,  # Try to use CUDA if available
                simulation_mode=not self.use_real_eeg,  # Use real EEG if specified
                enable_visualization=False,  # No visualization in the demo
                eeg_debug_output=self.debug_mode
            )
            
            # Check if model was initialized successfully
            if not hasattr(self.llm, 'model') or self.llm.model is None:
                print("Model initialization failed. Using a smaller model as fallback.")
                # Try with a smaller model
                self.llm = EEGEnhancedLLM(
                    model_path="facebook/opt-125m",  # Smaller model
                    use_cuda=False,
                    simulation_mode=not self.use_real_eeg,
                    enable_visualization=False,
                    eeg_debug_output=self.debug_mode
                )
            
            # Stop background threads that might interfere with the console
            self._stop_background_threads()
            
            # If not using real EEG, override the attention level with our manual control
            if not self.use_real_eeg:
                original_get_attention = self.llm.eeg_processor.get_attention_level
                
                def fixed_attention_level():
                    return self.attention_level
                    
                self.llm.eeg_processor.get_attention_level = fixed_attention_level
            
            # Silence output during generation by wrapping the generate method
            original_generate = self.llm.generate_with_eeg_control
            
            def silent_generate(prompt, max_new_tokens=None):
                if self.suppress_output:
                    # Redirect stdout to null during generation
                    import os
                    import sys
                    import contextlib
                    
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull):
                            return original_generate(prompt, max_new_tokens)
                else:
                    return original_generate(prompt, max_new_tokens)
                    
            self.llm.generate_with_eeg_control = silent_generate
            
            print("Model initialized and ready!")
            return True
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            if self.debug_mode:
                traceback.print_exc()
            print("Please make sure you have the required dependencies installed.")
            return False
    
    def _stop_background_threads(self):
        """Attempt to identify and stop any threads that might be printing to console"""
        # Get all current threads
        for thread in threading.enumerate():
            # Skip main thread
            if thread is threading.current_thread():
                continue
                
            # Skip any threads that are daemon (they'll exit when main thread ends)
            if thread.daemon:
                continue
                
            # Try to stop any thread that might be from EEG processor
            if 'EEG' in thread.name or 'eeg' in thread.name:
                if self.debug_mode:
                    print(f"Stopping thread: {thread.name}")
                # We don't have a clean way to stop it, but we can try to make it stop printing
                for attr_name in dir(thread):
                    if 'print' in attr_name.lower() or 'output' in attr_name.lower():
                        try:
                            setattr(thread, attr_name, False)
                        except:
                            pass
    
    def select_prompt(self):
        """Let the user select or enter a prompt"""
        prompts = [
            "Explain quantum computing principles.",
            "Describe how neural networks function.",
            "Explain the concept of climate change.",
            "Tell me about the history of artificial intelligence.",
            "Describe the process of photosynthesis."
        ]
        
        print("\n" + "="*80)
        print(" " * 20 + "EEG-ENHANCED LLM DEMO")
        print("="*80 + "\n")
        
        print("Available prompts:")
        for i, prompt in enumerate(prompts, 1):
            print(f"  {i}. {prompt}")
        
        while True:
            try:
                choice = input("\nSelect a prompt (1-5) or enter your own: ")
                if choice.strip() == "":
                    return None
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(prompts):
                        return prompts[idx]
                else:
                    # User entered their own prompt
                    return choice.strip()
                    
                print("Invalid selection. Please choose a number between 1 and 5 or enter your own prompt.")
            except ValueError:
                print("Please enter a valid number or text.")
            except KeyboardInterrupt:
                self.running = False
                return None
    
    def print_instructions(self):
        """Display instructions for using the demo"""
        print("\n" + "-"*80)
        print("INSTRUCTIONS:")
        print("1. The model will generate text in small chunks")
        if not self.use_real_eeg:
            print("2. After each chunk, you can control the attention level")
            print("3. Enter a number between 0.1 and 0.9 to adjust attention")
            print("   - 0.1-0.3: Low attention, simpler language")
            print("   - 0.4-0.6: Medium attention, balanced language")
            print("   - 0.7-0.9: High attention, more complex language")
            print("4. Press Enter to continue with current attention level")
        else:
            print("2. Your real-time EEG attention level will control the generation")
            print("3. Press Enter after each chunk to continue")
        print("5. Enter 'q' to quit the demo")
        print("-"*80)
    
    def generate_text_chunk(self, prompt, max_tokens=25):
        """Generate a chunk of text with the current attention level"""
        # Track tokens generated before this chunk
        before_tokens_generated = self.llm.generation_stats["tokens_generated"]
        
        # Generate text with output suppressed
        self.suppress_output = True
        try:
            response = self.llm.generate_with_eeg_control(prompt, max_new_tokens=max_tokens)
        except Exception as e:
            print(f"Error during generation: {e}")
            if self.debug_mode:
                traceback.print_exc()
            # Return empty result rather than crashing
            return "", 0, "unknown"
        self.suppress_output = False
        
        # Calculate how many tokens were actually generated
        tokens_generated = self.llm.generation_stats["tokens_generated"] - before_tokens_generated
        
        # Extract only the new part of the response
        try:
            prompt_token_count = len(self.llm.tokenizer.encode(prompt))
            response_tokens = self.llm.tokenizer.encode(response)
            
            # Add safety check for empty token arrays or when prompt_token_count is >= length
            if len(response_tokens) <= prompt_token_count:
                new_text = ""  # No new tokens generated
            else:
                new_text = self.llm.tokenizer.decode(
                    response_tokens[prompt_token_count:], 
                    skip_special_tokens=True
                )
        except Exception as e:
            print(f"Error processing response: {e}")
            if self.debug_mode:
                traceback.print_exc()
            # Fall back to a simple string replacement approach
            new_text = response.replace(prompt, "", 1) if response and response.startswith(prompt) else response
        
        # Record attention and expert weights for visualization
        current_attention = self.attention_level
        if self.use_real_eeg:
            # If using real EEG, get the actual attention level
            current_attention = self.llm.eeg_processor.get_attention_level()
            
        self.attention_history.append(current_attention)
        
        # Get expert weights and dominant expert
        dominant_expert = "unknown"
        try:
            expert_weights = self.llm.moe_controller.current_weights
            if expert_weights:
                self.expert_history["simple"].append(expert_weights.get("simple", 0))
                self.expert_history["balanced"].append(expert_weights.get("balanced", 0))
                self.expert_history["complex"].append(expert_weights.get("complex", 0))
                
                # Find the dominant expert
                max_weight = 0
                for expert, weight in expert_weights.items():
                    if weight > max_weight:
                        max_weight = weight
                        dominant_expert = expert
            else:
                # Fall back to controller's method
                dominant_expert = self.llm.moe_controller.get_current_expert() or "balanced"
        except Exception as e:
            if self.debug_mode:
                print(f"Error getting expert info: {e}")
            dominant_expert = "balanced"
        
        return new_text, tokens_generated, dominant_expert
    
    def run_demo(self):
        """Run the interactive demo"""
        try:
            if not self.initialize_llm():
                print("Could not initialize the language model. Exiting demo.")
                return
            
            self.prompt = self.select_prompt()
            if not self.prompt or not self.running:
                return
                
            full_prompt = SYSTEM_PROMPT + f"\n\nUser: {self.prompt}\n\nAssistant: "
            self.print_instructions()
            
            print(f"\nPrompt: {self.prompt}")
            if not self.use_real_eeg:
                print(f"Initial attention level: {self.attention_level:.1f}")
                
            print("\nGeneration will appear below:")
            print("-"*80)
            
            total_tokens = 0
            start_time = time.time()
            chunks_generated = 0
            
            while self.running and total_tokens < 200:
                try:
                    # Get current attention level from EEG if using real EEG
                    if self.use_real_eeg:
                        self.attention_level = self.llm.eeg_processor.get_attention_level()
                        
                    # Generate a chunk of text
                    new_chunk, tokens_generated, dominant_expert = self.generate_text_chunk(
                        full_prompt + self.current_response)
                    
                    if tokens_generated == 0:
                        # No more tokens generated, end generation
                        break
                    
                    # Update total tokens and add the new chunk to the response
                    total_tokens += tokens_generated
                    self.current_response += new_chunk
                    chunks_generated += 1
                    
                    # Determine attention category
                    attention_category = "Low" 
                    if self.attention_level > 0.3:
                        attention_category = "Medium"
                    if self.attention_level > 0.7:
                        attention_category = "High"
                    
                    # Display the new chunk with stats
                    print(f"\n[Chunk {chunks_generated}] (Attention: {self.attention_level:.1f} - {attention_category}, Expert: {dominant_expert})")
                    print(f">> {new_chunk}")
                    
                    # Ask for user input
                    if self.use_real_eeg:
                        user_input = input("\nPress Enter to continue (q to quit): ")
                    else:
                        user_input = input("\nAttention (0.1-0.9) or Enter to continue (q to quit): ")
                    
                    if user_input.lower() == 'q':
                        self.running = False
                        break
                    elif user_input.strip() == "":
                        # Continue with current attention level
                        pass
                    elif not self.use_real_eeg:
                        # Only process attention level input if not using real EEG
                        try:
                            new_level = float(user_input)
                            if 0.1 <= new_level <= 0.9:
                                self.attention_level = new_level
                                print(f"Attention set to {self.attention_level:.1f}")
                            else:
                                print("Attention must be between 0.1 and 0.9")
                        except ValueError:
                            print("Please enter a valid number")
                            
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user")
                    break
                except Exception as e:
                    print(f"\nError during generation: {e}")
                    if self.debug_mode:
                        traceback.print_exc()
                    time.sleep(1)  # Brief pause before continuing
            
            # Generation completed or stopped
            duration = time.time() - start_time
            
            print("\n" + "="*80)
            print("GENERATION COMPLETED")
            print("="*80)
            print(f"Generated {total_tokens} tokens in {duration:.2f} seconds")
            
            # Show expert usage statistics
            if self.expert_history["simple"] and len(self.expert_history["simple"]) > 0:
                simple_avg = sum(self.expert_history["simple"]) / len(self.expert_history["simple"])
                balanced_avg = sum(self.expert_history["balanced"]) / len(self.expert_history["balanced"])
                complex_avg = sum(self.expert_history["complex"]) / len(self.expert_history["complex"])
                
                print("\nExpert Usage Statistics:")
                print(f"Simple:   {simple_avg:.2%}")
                print(f"Balanced: {balanced_avg:.2%}")
                print(f"Complex:  {complex_avg:.2%}")
            
            print("\nFull Response:")
            print("-"*80)
            print(self.current_response)
            print("-"*80)
            
            # Save results to a file
            self.save_results_to_file()
            
        except KeyboardInterrupt:
            print("\n\n[Interrupted by user]")
        except Exception as e:
            print(f"\nError in demo: {e}")
            if self.debug_mode:
                traceback.print_exc()
        finally:
            if hasattr(self, 'llm') and self.llm:
                # Cleanup in a way that won't print anything
                with open(os.devnull, 'w') as devnull:
                    import contextlib
                    with contextlib.redirect_stdout(devnull):
                        try:
                            self.llm.cleanup()
                        except:
                            pass
            print("Demo completed.")
    
    def save_results_to_file(self):
        """Save generation results and statistics to a file"""
        timestamp = int(time.time())
        
        # Create results directory if it doesn't exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"{results_dir}/eeg_demo_results_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EEG-ENHANCED LLM GENERATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Prompt: {self.prompt}\n\n")
            f.write(f"Response:\n{self.current_response}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("STATISTICS\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"Tokens generated: {len(self.attention_history)}\n")
            unique_attention_levels = set()
            if self.attention_history:
                unique_attention_levels = set([round(a, 1) for a in self.attention_history])
            f.write(f"Attention levels used: {len(unique_attention_levels)}\n")
            
            if self.expert_history["simple"] and len(self.expert_history["simple"]) > 0:
                simple_avg = sum(self.expert_history["simple"]) / len(self.expert_history["simple"])
                balanced_avg = sum(self.expert_history["balanced"]) / len(self.expert_history["balanced"])
                complex_avg = sum(self.expert_history["complex"]) / len(self.expert_history["complex"])
                
                f.write("\nExpert Usage:\n")
                f.write(f"Simple:   {simple_avg:.2%}\n")
                f.write(f"Balanced: {balanced_avg:.2%}\n")
                f.write(f"Complex:  {complex_avg:.2%}\n")
            
            if self.attention_history:
                f.write("\nAttention history:\n")
                for i, att in enumerate(self.attention_history):
                    f.write(f"{i+1}: {att:.1f}\n")
        
        print(f"\nResults saved to {filename}")
        
        # Notify about visualization
        print("\nTo visualize the results, run one of the following:")
        print(f"python3 visualize_results.py {filename}")
        print("  This will show an interactive visualization")
        print(f"python3 visualize_results.py {filename} --save")
        print("  This will save visualizations including:")
        print("  - results/moe_routing_demo.png: Combined attention pattern and expert routing")
        print("  - results/attention_history.png: Detailed attention level history")
        print("  - results/expert_usage_pie.png: Expert usage distribution")

def main():
    """Main function to run the demo"""
    parser = argparse.ArgumentParser(description="EEG-Enhanced LLM Demo")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m",
                        help="Path to the language model")
    parser.add_argument("--real-eeg", action="store_true",
                        help="Use real EEG data instead of simulation")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with detailed error messages")
    
    args = parser.parse_args()
    
    demo = UnifiedEEGDemo(
        model_path=args.model_path,
        use_real_eeg=args.real_eeg
    )
    demo.debug_mode = args.debug
    demo.run_demo()

if __name__ == "__main__":
    main() 