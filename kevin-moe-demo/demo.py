#!/usr/bin/env python3
"""
EEG-Enhanced Language Model Demo
--------------------------------
This script demonstrates the complete EEG-controlled language model system
with Mixture-of-Experts routing based on real-time attention signals.
"""

import time
import argparse
import os
import matplotlib.pyplot as plt
from main_eeg import EEGEnhancedLLM, SYSTEM_PROMPT
import numpy as np

# Demo prompts to showcase capabilities
DEMO_PROMPTS = [
    "Explain quantum computing to me.",
    "What are the key concepts in machine learning?",
    "How do neural networks work?",
    "What is the significance of EEG signals in brain-computer interfaces?",
    "Explain the concept of consciousness.",
    "What are the ethical implications of AI?",
]

def run_interactive_demo(model_path, simulation_mode=True, enable_visualization=True):
    """
    Run an interactive demo of the EEG-enhanced language model
    
    Args:
        model_path: Path to the model
        simulation_mode: Whether to use simulated EEG data
        enable_visualization: Whether to enable visualization
    """
    print("\n" + "="*80)
    print(" "*30 + "EEG-ENHANCED LLM DEMO")
    print("="*80)
    print("\nInitializing system...")
    
    # Initialize the system
    llm = EEGEnhancedLLM(
        model_path=model_path,
        simulation_mode=simulation_mode,
        enable_visualization=enable_visualization
    )
    
    print("\nSystem ready! In this demo, your attention level is being simulated.")
    print("The model will adapt its responses based on the detected attention level.")
    print("\nOptions:")
    print("  1. Try a demo prompt")
    print("  2. Enter your own prompt")
    print("  3. Run automated demo")
    print("  4. View system statistics")
    print("  q. Quit")
    
    try:
        while True:
            choice = input("\nEnter your choice (1-4, q): ").strip().lower()
            
            if choice == 'q':
                break
                
            elif choice == '1':
                # Show demo prompts
                print("\nDemo prompts:")
                for i, prompt in enumerate(DEMO_PROMPTS):
                    print(f"  {i+1}. {prompt}")
                
                try:
                    prompt_idx = int(input("\nSelect a prompt (1-6): ")) - 1
                    if 0 <= prompt_idx < len(DEMO_PROMPTS):
                        user_prompt = DEMO_PROMPTS[prompt_idx]
                        print(f"\nSelected: {user_prompt}")
                        
                        full_prompt = SYSTEM_PROMPT + f"\n\nUser: {user_prompt}\n\nAssistant: "
                        print("\nGenerating response...")
                        result = llm.generate_with_eeg_control(full_prompt)
                        
                        print("\nGenerated response:")
                        print("-" * 80)
                        print(result.split("Assistant: ")[-1])
                        print("-" * 80)
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            elif choice == '2':
                # Custom prompt
                user_prompt = input("\nEnter your question: ").strip()
                if user_prompt:
                    full_prompt = SYSTEM_PROMPT + f"\n\nUser: {user_prompt}\n\nAssistant: "
                    print("\nGenerating response...")
                    result = llm.generate_with_eeg_control(full_prompt)
                    
                    print("\nGenerated response:")
                    print("-" * 80)
                    print(result.split("Assistant: ")[-1])
                    print("-" * 80)
            
            elif choice == '3':
                # Automated demo - runs through all demo prompts
                print("\nStarting automated demo...")
                print("This will generate responses for all demo prompts.")
                
                for i, prompt in enumerate(DEMO_PROMPTS):
                    print(f"\n[Demo {i+1}/{len(DEMO_PROMPTS)}] {prompt}")
                    
                    full_prompt = SYSTEM_PROMPT + f"\n\nUser: {prompt}\n\nAssistant: "
                    result = llm.generate_with_eeg_control(full_prompt)
                    
                    print("\nGenerated response:")
                    print("-" * 80)
                    print(result.split("Assistant: ")[-1])
                    print("-" * 80)
                    
                    if i < len(DEMO_PROMPTS) - 1:
                        print("\nMoving to next demo in 3 seconds...")
                        time.sleep(3)
            
            elif choice == '4':
                # Display system statistics
                stats = llm.get_generation_stats()
                
                print("\nSystem Statistics:")
                print("-" * 80)
                print(f"Total tokens generated: {stats['tokens_generated']}")
                print(f"Average attention level: {stats['avg_attention']:.2f}")
                
                if "attention_distribution" in stats:
                    att_dist = stats["attention_distribution"]
                    print("\nAttention Distribution:")
                    print(f"  Low attention: {att_dist['low']*100:.1f}%")
                    print(f"  Medium attention: {att_dist['medium']*100:.1f}%")
                    print(f"  High attention: {att_dist['high']*100:.1f}%")
                
                print("\nExpert Usage:")
                for expert, count in stats.get("expert_usage", {}).items():
                    if stats['tokens_generated'] > 0:
                        percentage = count/stats['tokens_generated']*100
                        print(f"  {expert}: {count} tokens ({percentage:.1f}%)")
                
                # Get EEG processor metrics
                eeg_metrics = llm.eeg_processor.get_attention_metrics()
                print("\nEEG Attention Metrics:")
                for key, value in eeg_metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
                
                # Create a simple bar chart for expert usage
                if stats.get("expert_usage") and stats['tokens_generated'] > 0:
                    plt.figure(figsize=(10, 4))
                    experts = list(stats["expert_usage"].keys())
                    usage = [stats["expert_usage"][e]/stats['tokens_generated']*100 for e in experts]
                    colors = ['blue', 'green', 'red']
                    
                    plt.bar(experts, usage, color=colors[:len(experts)])
                    plt.title('Expert Usage Distribution')
                    plt.ylabel('Usage Percentage (%)')
                    plt.ylim(0, 100)
                    
                    for i, v in enumerate(usage):
                        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
                    
                    plt.tight_layout()
                    plt.show(block=False)
            
            else:
                print("Invalid choice. Please try again.")
    
    finally:
        # Clean up
        llm.cleanup()
        print("\nDemo session ended.")

def main():
    parser = argparse.ArgumentParser(description="EEG-Enhanced Language Model Demo")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Path to the language model")
    parser.add_argument("--real-eeg", action="store_true", 
                        help="Use real EEG data from Muse headset instead of simulation")
    parser.add_argument("--no-visualization", action="store_true",
                        help="Disable visualization")
    
    args = parser.parse_args()
    
    run_interactive_demo(
        model_path=args.model,
        simulation_mode=not args.real_eeg,
        enable_visualization=not args.no_visualization
    )

if __name__ == "__main__":
    main() 