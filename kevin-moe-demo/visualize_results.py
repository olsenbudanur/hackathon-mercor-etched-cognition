#!/usr/bin/env python3
"""
Visualize results from EEG-enhanced LLM interactive demo.
This script reads the results saved from interactive_demo.py and generates visualizations.
"""

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_results_file(filename):
    """Parse the results file and extract data for visualization"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract prompt and response
    prompt_match = re.search(r"Prompt: (.*?)\n\n", content, re.DOTALL)
    prompt = prompt_match.group(1) if prompt_match else "Unknown prompt"
    
    response_match = re.search(r"Response:\n(.*?)\n\n-{80}", content, re.DOTALL)
    response = response_match.group(1) if response_match else "No response found"
    
    # Extract statistics
    tokens_match = re.search(r"Tokens generated: (\d+)", content)
    tokens_generated = int(tokens_match.group(1)) if tokens_match else 0
    
    # Extract expert usage
    simple_match = re.search(r"Simple:\s+(\d+\.\d+)%", content)
    balanced_match = re.search(r"Balanced:\s+(\d+\.\d+)%", content)
    complex_match = re.search(r"Complex:\s+(\d+\.\d+)%", content)
    
    expert_usage = {}
    if simple_match and balanced_match and complex_match:
        expert_usage = {
            "simple": float(simple_match.group(1)) / 100,
            "balanced": float(balanced_match.group(1)) / 100,
            "complex": float(complex_match.group(1)) / 100
        }
    
    # Extract attention history
    attention_history = []
    att_matches = re.findall(r"\d+: (0\.\d+)", content)
    for match in att_matches:
        attention_history.append(float(match))
    
    return {
        "prompt": prompt,
        "response": response,
        "tokens_generated": tokens_generated,
        "expert_usage": expert_usage,
        "attention_history": attention_history
    }

def visualize_expert_usage(expert_usage):
    """Create a pie chart of expert usage"""
    if not expert_usage:
        print("No expert usage data found.")
        return
    
    # Create pie chart
    plt.figure(figsize=(10, 6))
    labels = ['Simple', 'Balanced', 'Complex']
    sizes = [expert_usage.get("simple", 0), 
             expert_usage.get("balanced", 0), 
             expert_usage.get("complex", 0)]
    colors = ['lightblue', 'lightgreen', 'coral']
    explode = (0.1, 0, 0)  # explode the 1st slice (Simple)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Expert Usage Distribution')
    
    plt.tight_layout()
    return plt.gcf()

def visualize_attention_history(attention_history):
    """Create a line chart of attention history"""
    if not attention_history:
        print("No attention history data found.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(attention_history, 'b-o', linewidth=2, markersize=4)
    plt.ylabel('Attention Level')
    plt.xlabel('Token Number')
    plt.title('Attention Level During Generation')
    plt.ylim(0, 1.1)
    
    # Add colored background regions for different attention levels
    plt.axhspan(0, 0.3, alpha=0.2, color='blue', label='Low Attention')
    plt.axhspan(0.3, 0.7, alpha=0.2, color='green', label='Medium Attention')
    plt.axhspan(0.7, 1.0, alpha=0.2, color='red', label='High Attention')
    
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    return plt.gcf()

def simulate_expert_weights(attention_history):
    """Simulate expert weights based on attention history for visualization purposes"""
    # This is an approximation - in a real system, expert weights would be directly recorded
    simple_weights = []
    balanced_weights = []
    complex_weights = []
    
    for att in attention_history:
        # Simplified model of how attention affects expert weights
        if att < 0.3:
            simple = 0.7 - att
            balanced = 0.3 + att
            complex = att
        elif att < 0.7:
            simple = 0.3 - (att - 0.3) * 0.75
            balanced = 0.6
            complex = 0.1 + (att - 0.3) * 1.5
        else:
            simple = 0.0
            balanced = 0.9 - (att - 0.7) * 2
            complex = 0.1 + (att - 0.7) * 2
            
        # Normalize to ensure they sum to 1
        total = simple + balanced + complex
        simple_weights.append(simple / total)
        balanced_weights.append(balanced / total)
        complex_weights.append(complex / total)
    
    return simple_weights, balanced_weights, complex_weights

def visualize_expert_weights_over_time(attention_history):
    """Create a stacked area chart of expert weights over time"""
    if not attention_history:
        print("No attention history data found.")
        return
    
    # Simulate expert weights based on attention
    simple_weights, balanced_weights, complex_weights = simulate_expert_weights(attention_history)
    
    # Create stacked area chart
    plt.figure(figsize=(12, 6))
    x = range(len(attention_history))
    
    plt.stackplot(x, 
                 [simple_weights, balanced_weights, complex_weights], 
                 labels=['Simple', 'Balanced', 'Complex'],
                 colors=['lightblue', 'lightgreen', 'coral'])
    
    plt.ylabel('Expert Weight')
    plt.xlabel('Token Number')
    plt.title('Expert Weights During Generation')
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    return plt.gcf()

def visualize_combined_attention_and_experts(attention_history):
    """
    Create a combined two-panel plot with attention pattern on top
    and expert routing weights on bottom - similar to test_pattern_routing.py output
    """
    if not attention_history:
        print("No attention history data found.")
        return
    
    # Simulate expert weights based on attention
    simple_weights, balanced_weights, complex_weights = simulate_expert_weights(attention_history)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot attention pattern in the top panel
    x = range(len(attention_history))
    ax1.plot(x, attention_history, 'k-', linewidth=2, label='Attention')
    ax1.set_ylim(0, 1)
    ax1.set_title('Attention Pattern')
    ax1.set_ylabel('Attention Level')
    
    # Add colored background regions for different attention levels
    ax1.axhspan(0, 0.3, alpha=0.2, color='blue')
    ax1.axhspan(0.3, 0.7, alpha=0.2, color='green')
    ax1.axhspan(0.7, 1.0, alpha=0.2, color='red')
    ax1.legend(loc='upper right')
    
    # Plot expert weights in the bottom panel
    ax2.plot(x, simple_weights, 'b-', linewidth=2, label='Simple Expert')
    ax2.plot(x, balanced_weights, 'g-', linewidth=2, label='Balanced Expert')
    ax2.plot(x, complex_weights, 'r-', linewidth=2, label='Complex Expert')
    ax2.set_ylim(0, 1)
    ax2.set_title('Expert Routing Weights')
    ax2.set_xlabel('Time (samples)')
    ax2.set_ylabel('Expert Weight')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize EEG-Enhanced LLM Demo Results")
    parser.add_argument("results_file", type=str, 
                        help="Path to the results file generated by interactive_demo.py")
    parser.add_argument("--save", action="store_true",
                        help="Save visualizations to PNG files instead of displaying them")
    
    args = parser.parse_args()
    
    try:
        # Parse the results file
        results = parse_results_file(args.results_file)
        
        # Print basic information
        print("\n" + "="*80)
        print("EEG-ENHANCED LLM DEMO RESULTS VISUALIZATION")
        print("="*80 + "\n")
        
        print(f"Prompt: {results['prompt']}")
        print(f"Tokens Generated: {results['tokens_generated']}")
        
        if results['expert_usage']:
            print("\nExpert Usage:")
            for expert, usage in results['expert_usage'].items():
                print(f"- {expert.capitalize()}: {usage:.2%}")
        
        # Generate visualizations
        if results['attention_history']:
            # Create results directory if it doesn't exist
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate the combined visualization
            combined_fig = visualize_combined_attention_and_experts(results['attention_history'])
            
            if args.save:
                combined_fig.savefig(f"{results_dir}/moe_routing_demo.png")
                print(f"\nSaved combined visualization to {results_dir}/moe_routing_demo.png")
                
                # Still generate the individual visualizations if needed
                expert_fig = visualize_expert_usage(results['expert_usage'])
                expert_fig.savefig(f"{results_dir}/expert_usage_pie.png")
                print(f"Saved expert usage pie chart to {results_dir}/expert_usage_pie.png")
                
                att_fig = visualize_attention_history(results['attention_history'])
                att_fig.savefig(f"{results_dir}/attention_history.png")
                print(f"Saved attention history chart to {results_dir}/attention_history.png")
            else:
                plt.show()
        else:
            print("No attention history data found in the results file.")
            
    except Exception as e:
        print(f"Error visualizing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 