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
import matplotlib.pyplot as plt
from main_eeg import EEGEnhancedLLM, SYSTEM_PROMPT
import numpy as np
from eeg_processor import HTTPEEGProcessor
from token_streamer import TokenStreamer, init_streamer

# Demo prompts to showcase capabilities
DEMO_PROMPTS = [
    "Explain quantum computing to me.",
    "What are the key concepts in machine learning?",
    "How do neural networks work?",
    "What is the significance of EEG signals in brain-computer interfaces?",
    "Explain the concept of consciousness.",
    "What are the ethical implications of AI?",
]

def run_interactive_demo(model_path, 
                       use_http_eeg=False, 
                       api_url="https://f424-216-201-226-138.ngrok-free.app/latest_value",
                       enable_visualization=False,
                       enable_token_streaming=True,
                       token_chunk_size=5,
                       backend_url="http://localhost:8000"):
    """
    Run an interactive demo of the EEG-enhanced language model
    
    Args:
        model_path: Path to the model
        use_http_eeg: Whether to use HTTP API for EEG data
        api_url: URL for HTTP EEG API
        enable_visualization: Whether to enable visualization
        enable_token_streaming: Whether to enable token streaming to frontend
        token_chunk_size: Number of tokens to generate before checking attention
        backend_url: URL for backend token streaming API
    """
    print("\n" + "="*80)
    print(" "*30 + "EEG-ENHANCED LLM DEMO")
    print("="*80)
    print("\nInitializing system...")
    
    # Initialize token streamer if enabled
    token_streamer = None
    if enable_token_streaming:
        print(f"Initializing token streamer to {backend_url}")
        token_streamer = TokenStreamer(
            api_url=backend_url,
            batch_size=3,  # Send tokens in batches of 3
            send_interval=0.1
        )
    
    # Use custom EEG processor for HTTP API
    if use_http_eeg:
        print(f"Using HTTP EEG processor with API: {api_url}")
        eeg_processor = HTTPEEGProcessor(api_url=api_url, debug_output=True)
        eeg_processor.start()
        
        # Initialize the system with custom EEG processor
        llm = EEGEnhancedLLM(
            model_path=model_path,
            simulation_mode=True,  # Simulation mode is ignored when we replace the processor
            enable_visualization=enable_visualization
        )
        # Replace the EEG processor
        llm.eeg_processor = eeg_processor
    else:
        # Use regular EEG processor with simulation
        llm = EEGEnhancedLLM(
            model_path=model_path,
            simulation_mode=True,
            enable_visualization=enable_visualization
        )
    
    print(f"\nSystem ready! Using token chunk size of {token_chunk_size}")
    if use_http_eeg:
        print(f"Attention level is being pulled from HTTP API: {api_url}")
    else:
        print("Attention level is being simulated.")
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
                        
                        # Clear token queue if streaming is enabled
                        if token_streamer:
                            token_streamer.clear_queue()
                            
                        # Generate with chunk-based generation
                        result = llm.generate_with_eeg_control(
                            full_prompt, 
                            token_chunk_size=token_chunk_size,
                            token_streamer=token_streamer
                        )
                        
                        print("\nGenerated response:")
                        print("-" * 80)
                        print(result.split("Assistant: ")[-1])
                        print("-" * 80)
                        
                        if token_streamer:
                            print(f"Tokens streamed: {token_streamer.tokens_sent}")
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
                    
                    # Clear token queue if streaming is enabled
                    if token_streamer:
                        token_streamer.clear_queue()
                    
                    # Generate with chunk-based generation
                    result = llm.generate_with_eeg_control(
                        full_prompt, 
                        token_chunk_size=token_chunk_size,
                        token_streamer=token_streamer
                    )
                    
                    print("\nGenerated response:")
                    print("-" * 80)
                    print(result.split("Assistant: ")[-1])
                    print("-" * 80)
                    
                    if token_streamer:
                        print(f"Tokens streamed: {token_streamer.tokens_sent}")
            
            elif choice == '3':
                # Automated demo - runs through all demo prompts
                print("\nStarting automated demo...")
                print("This will generate responses for all demo prompts.")
                
                for i, prompt in enumerate(DEMO_PROMPTS):
                    print(f"\n[Demo {i+1}/{len(DEMO_PROMPTS)}] {prompt}")
                    
                    # Clear token queue if streaming is enabled
                    if token_streamer:
                        token_streamer.clear_queue()
                    
                    full_prompt = SYSTEM_PROMPT + f"\n\nUser: {prompt}\n\nAssistant: "
                    
                    # Generate with chunk-based generation
                    result = llm.generate_with_eeg_control(
                        full_prompt, 
                        token_chunk_size=token_chunk_size,
                        token_streamer=token_streamer
                    )
                    
                    print("\nGenerated response:")
                    print("-" * 80)
                    print(result.split("Assistant: ")[-1])
                    print("-" * 80)
                    
                    if token_streamer:
                        print(f"Tokens streamed: {token_streamer.tokens_sent}")
                    
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
                
                # Display token streamer stats if enabled
                if token_streamer:
                    streamer_stats = token_streamer.get_stats()
                    print("\nToken Streamer Statistics:")
                    print(f"  Tokens sent: {streamer_stats['tokens_sent']}")
                    print(f"  Batches sent: {streamer_stats['batches_sent']}")
                    print(f"  Errors: {streamer_stats['errors']}")
                    print(f"  Queue size: {streamer_stats['queue_size']}")
                    print(f"  API URL: {streamer_stats['api_url']}")
                
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
        if token_streamer:
            token_streamer.stop()
        print("\nDemo session ended.")

def main():
    parser = argparse.ArgumentParser(description="EEG-Enhanced Language Model Demo")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Path to the language model")
    parser.add_argument("--http-eeg", action="store_true", 
                        help="Use HTTP endpoint for EEG data")
    parser.add_argument("--eeg-url", type=str, 
                        default="https://f424-216-201-226-138.ngrok-free.app/latest_value",
                        help="URL for EEG HTTP API")
    parser.add_argument("--no-visualization", action="store_true",
                        help="Disable visualization")
    parser.add_argument("--no-streaming", action="store_true", 
                        help="Disable token streaming to frontend")
    parser.add_argument("--backend-url", type=str, 
                        default="http://localhost:8000",
                        help="URL for backend API")
    parser.add_argument("--token-chunk-size", type=int, default=5,
                        help="Number of tokens to generate before checking attention")
    
    args = parser.parse_args()
    
    run_interactive_demo(
        model_path=args.model,
        use_http_eeg=args.http_eeg,
        api_url=args.eeg_url,
        enable_visualization=not args.no_visualization,
        enable_token_streaming=not args.no_streaming,
        token_chunk_size=args.token_chunk_size,
        backend_url=args.backend_url
    )

if __name__ == "__main__":
    main() 