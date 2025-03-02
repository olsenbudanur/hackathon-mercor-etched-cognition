#!/usr/bin/env python3
"""
Simplified EEG-Enhanced LLM Demo for Testing
-------------------------------------------
This script provides a simplified version of the demo without GUI elements
for easier testing of the integrated system.
"""

import time
import argparse
from main_eeg import EEGEnhancedLLM, SYSTEM_PROMPT

# Sample prompts for testing
TEST_PROMPTS = [
    "Explain quantum computing in one paragraph.",
    "What is machine learning? Keep it simple.",
    "Define neural networks technically."
]

def run_simplified_demo(model_path, simulation_mode=True):
    """
    Run a simplified demo without visualization for testing
    
    Args:
        model_path: Path to the model
        simulation_mode: Whether to use simulated EEG data
    """
    print("\n" + "="*80)
    print(" "*20 + "EEG-ENHANCED LLM TEST DEMO (NO VISUALIZATION)")
    print("="*80)
    print("\nInitializing system...")
    
    # Initialize the system with visualization disabled
    llm = EEGEnhancedLLM(
        model_path=model_path,
        simulation_mode=simulation_mode,
        enable_visualization=False
    )
    
    print("\nSystem ready! Testing with sample prompts...")
    
    try:
        # Run test on each prompt
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n[Test {i+1}/{len(TEST_PROMPTS)}] Prompt: {prompt}")
            
            # Format the prompt
            full_prompt = SYSTEM_PROMPT + f"\n\nUser: {prompt}\n\nAssistant: "
            
            # Set attention levels for testing
            # This simulates different attention levels for each prompt
            if i == 0:
                print("Using simulated HIGH attention level")
                # Manually set high attention in simulator
                llm.eeg_processor.attention_level = 0.85
            elif i == 1:
                print("Using simulated LOW attention level")
                # Manually set low attention in simulator
                llm.eeg_processor.attention_level = 0.15
            else:
                print("Using simulated MEDIUM attention level")
                # Manually set medium attention in simulator
                llm.eeg_processor.attention_level = 0.5
            
            start_time = time.time()
            result = llm.generate_with_eeg_control(full_prompt, max_new_tokens=100)
            end_time = time.time()
            
            print(f"\nGeneration time: {end_time - start_time:.2f} seconds")
            print("\nGenerated response:")
            print("-" * 80)
            print(result.split("Assistant: ")[-1])
            print("-" * 80)
            
            # Print generation stats for this run
            stats = llm.get_generation_stats()
            print("\nGeneration Stats:")
            print(f"- Tokens generated: {stats['tokens_generated']}")
            print(f"- Average attention: {stats['avg_attention']:.2f}")
            
            # Print expert usage for this run
            print("\nExpert Usage:")
            for expert, count in stats.get("expert_usage", {}).items():
                if stats['tokens_generated'] > 0:
                    percentage = count/stats['tokens_generated']*100
                    print(f"- {expert}: {count} tokens ({percentage:.1f}%)")
            
            # Wait between tests
            if i < len(TEST_PROMPTS) - 1:
                print("\nMoving to next test in 2 seconds...")
                time.sleep(2)
        
        # Final summary
        print("\n" + "="*80)
        print(" "*30 + "TEST SUMMARY")
        print("="*80)
        print("\nAll test prompts processed successfully.")
        
        # Get overall EEG metrics
        eeg_metrics = llm.eeg_processor.get_attention_metrics()
        print("\nOverall EEG Attention Metrics:")
        for key, value in eeg_metrics.items():
            if isinstance(value, float):
                print(f"- {key}: {value:.2f}")
            else:
                print(f"- {key}: {value}")
    
    finally:
        # Clean up
        llm.cleanup()
        print("\nTest demo completed.")

def main():
    parser = argparse.ArgumentParser(description="EEG-Enhanced LLM Test Demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="Path to the language model")
    parser.add_argument("--real-eeg", action="store_true", 
                        help="Use real EEG data from Muse headset instead of simulation")
    
    args = parser.parse_args()
    
    run_simplified_demo(
        model_path=args.model,
        simulation_mode=not args.real_eeg
    )

if __name__ == "__main__":
    main() 