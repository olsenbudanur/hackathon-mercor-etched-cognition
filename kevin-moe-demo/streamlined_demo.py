#!/usr/bin/env python3
"""
Streamlined demo for EEG-enhanced language model with clean interface for attention control.
This version eliminates all background prints for a smooth user experience.
"""

import time
import argparse
import sys
import os
import numpy as np
import threading
from main_eeg import EEGEnhancedLLM, SYSTEM_PROMPT

class StreamlinedDemo:
    def __init__(self, model_path="facebook/opt-350m", use_real_eeg=False):
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
        
    def initialize_llm(self):
        print("Loading model... (this may take a moment)")
        
        # Create the LLM
        self.llm = EEGEnhancedLLM(
            model_path=self.model_path,
            use_cuda=False,  # Force CPU usage for stability
            simulation_mode=True,  # Always use simulation mode for demo
            enable_visualization=False,  # No visualization
            eeg_debug_output=False  # Disable EEG debug prints
        )
        
        # Hack: Kill any threads that might be printing to console
        self._stop_background_threads()
        
        # Prevent EEG processor from updating attention levels continuously
        # Just use our manually set level
        original_get_attention = self.llm.eeg_processor.get_attention_level
        
        def fixed_attention_level():
            return self.attention_level
            
        self.llm.eeg_processor.get_attention_level = fixed_attention_level
        
        # Stop any print statements from generation
        original_generate = self.llm.generate_with_eeg_control
        
        def silent_generate(prompt, max_new_tokens=None):
            if self.suppress_output:
                # Completely suppress all output during generation
                import os
                import sys
                import contextlib
                
                # Redirect stdout to null
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        return original_generate(prompt, max_new_tokens)
            else:
                return original_generate(prompt, max_new_tokens)
                
        self.llm.generate_with_eeg_control = silent_generate
        
        print("Model initialized and ready!")
        
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
                print(f"Stopping thread: {thread.name}")
                # We don't have a clean way to stop it, but we can try to make it stop printing
                # By setting attributes that might control its printing behavior
                for attr_name in dir(thread):
                    if 'print' in attr_name.lower() or 'output' in attr_name.lower():
                        try:
                            setattr(thread, attr_name, False)
                        except:
                            pass
    
    def select_prompt(self):
        prompts = [
            "Explain quantum computing principles.",
            "Describe how neural networks function.",
            "Explain the concept of climate change.",
            "Tell me about the history of artificial intelligence.",
            "Describe the process of photosynthesis."
        ]
        
        print("\n" + "="*80)
        print(" " * 20 + "STREAMLINED EEG-ENHANCED LLM DEMO")
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
        print("\n" + "-"*80)
        print("INSTRUCTIONS:")
        print("1. The model will generate text in small chunks")
        print("2. After each chunk, you can control the attention level")
        print("3. Enter a number between 0.1 and 0.9 to adjust attention")
        print("   - 0.1-0.3: Low attention, simpler language")
        print("   - 0.4-0.6: Medium attention, balanced language")
        print("   - 0.7-0.9: High attention, more complex language")
        print("4. Press Enter to continue with current attention level")
        print("5. Enter 'q' to quit the demo")
        print("-"*80)
    
    def generate_text_chunk(self, prompt, max_tokens=25):
        """Generate a chunk of text with the current attention level"""
        # We've patched the EEG processor to use our manually set level
        
        # Track tokens generated before this chunk
        before_tokens_generated = self.llm.generation_stats["tokens_generated"]
        
        # Generate text with output suppressed
        self.suppress_output = True
        response = self.llm.generate_with_eeg_control(prompt, max_new_tokens=max_tokens)
        self.suppress_output = False
        
        # Calculate how many tokens were actually generated
        tokens_generated = self.llm.generation_stats["tokens_generated"] - before_tokens_generated
        
        # Extract only the new part of the response
        prompt_token_count = len(self.llm.tokenizer.encode(prompt))
        new_text = self.llm.tokenizer.decode(
            self.llm.tokenizer.encode(response)[prompt_token_count:], 
            skip_special_tokens=True
        )
        
        # Record attention and expert weights for visualization
        self.attention_history.append(self.attention_level)
        
        # Get expert weights
        expert_weights = self.llm.moe_controller.current_weights
        if expert_weights:
            self.expert_history["simple"].append(expert_weights.get("simple", 0))
            self.expert_history["balanced"].append(expert_weights.get("balanced", 0))
            self.expert_history["complex"].append(expert_weights.get("complex", 0))
        
        # Get the dominant expert
        dominant_expert = "unknown"
        if expert_weights:
            max_weight = 0
            for expert, weight in expert_weights.items():
                if weight > max_weight:
                    max_weight = weight
                    dominant_expert = expert
        
        return new_text, tokens_generated, dominant_expert
    
    def run_demo(self):
        try:
            self.initialize_llm()
            
            self.prompt = self.select_prompt()
            if not self.prompt or not self.running:
                return
                
            full_prompt = SYSTEM_PROMPT + f"\n\nUser: {self.prompt}\n\nAssistant: "
            self.print_instructions()
            
            print(f"\nPrompt: {self.prompt}")
            print(f"Initial attention level: {self.attention_level:.1f}")
            print("\nGeneration will appear below:")
            print("-"*80)
            
            total_tokens = 0
            start_time = time.time()
            chunks_generated = 0
            
            while self.running and total_tokens < 200:
                # Generate a chunk of text
                new_chunk, tokens_generated, dominant_expert = self.generate_text_chunk(full_prompt + self.current_response)
                
                if tokens_generated == 0:
                    # No more tokens generated, end generation
                    break
                
                # Update total tokens and add the new chunk to the response
                total_tokens += tokens_generated
                self.current_response += new_chunk
                chunks_generated += 1
                
                # Display the new chunk with stats
                # Use attention level ranges to show range
                attention_category = "Low" if self.attention_level < 0.3 else "High" if self.attention_level > 0.7 else "Medium"
                
                print(f"\n[Chunk {chunks_generated}] (Attention: {self.attention_level:.1f} - {attention_category}, Expert: {dominant_expert})")
                print(f">> {new_chunk}")
                
                # Ask for attention adjustment with clean prompt
                user_input = input("\nAttention (0.1-0.9) or Enter to continue (q to quit): ")
                
                if user_input.lower() == 'q':
                    self.running = False
                    break
                elif user_input.strip() == "":
                    # Continue with current attention level
                    pass
                else:
                    # Try to parse as a float
                    try:
                        new_level = float(user_input)
                        if 0.1 <= new_level <= 0.9:
                            self.attention_level = new_level
                            print(f"Attention set to {self.attention_level:.1f}")
                        else:
                            print("Attention must be between 0.1 and 0.9")
                    except ValueError:
                        print("Please enter a valid number")
            
            # Generation completed or stopped
            duration = time.time() - start_time
            
            print("\n" + "="*80)
            print("GENERATION COMPLETED")
            print("="*80)
            print(f"Generated {total_tokens} tokens in {duration:.2f} seconds")
            
            # Show expert usage statistics
            if self.expert_history["simple"]:
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
            f.write(f"Attention levels used: {len(set([round(a, 1) for a in self.attention_history]))}\n")
            
            if self.expert_history["simple"]:
                simple_avg = sum(self.expert_history["simple"]) / len(self.expert_history["simple"])
                balanced_avg = sum(self.expert_history["balanced"]) / len(self.expert_history["balanced"])
                complex_avg = sum(self.expert_history["complex"]) / len(self.expert_history["complex"])
                
                f.write("\nExpert Usage:\n")
                f.write(f"Simple:   {simple_avg:.2%}\n")
                f.write(f"Balanced: {balanced_avg:.2%}\n")
                f.write(f"Complex:  {complex_avg:.2%}\n")
            
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
        print("  - results/moe_routing_demo.png: Combined attention pattern and expert routing (like sine wave demo)")
        print("  - results/attention_history.png: Detailed attention level history")
        print("  - results/expert_usage_pie.png: Expert usage distribution")

def main():
    parser = argparse.ArgumentParser(description="Streamlined EEG-Enhanced LLM Demo")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m",
                        help="Path to the language model")
    
    args = parser.parse_args()
    
    demo = StreamlinedDemo(
        model_path=args.model_path
    )
    demo.run_demo()

if __name__ == "__main__":
    main() 