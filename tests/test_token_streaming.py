#!/usr/bin/env python3
"""
Test Token Streaming

This script simulates the token streaming functionality of the EEG demo.
It sends token data with associated experts to the backend API.
"""

import requests
import time
import random
import argparse
from typing import List, Dict, Any
import json

# Backend API URL
DEFAULT_API_URL = "http://localhost:8000"

# Sample texts with diverse content to simulate different expert usage
SAMPLE_TEXTS = [
    # Simple text that might use the "simple" expert
    "The sun rises in the east and sets in the west. Birds fly in the sky. Fish swim in the water.",
    
    # Balanced text that might use multiple experts
    "Neural networks are computational systems inspired by the human brain. They consist of layers of neurons that process information.",
    
    # Complex text that might favor the "complex" expert
    "Quantum entanglement is a quantum mechanical phenomenon where pairs of particles become correlated in such a way that the quantum state of each particle cannot be described independently of the others."
]

# Expert definitions
EXPERTS = ["simple", "balanced", "complex"]

def tokenize_text(text: str) -> List[str]:
    """
    Improved tokenization to better simulate a language model tokenizer.
    This provides more realistic tokens that will display properly on the frontend.
    """
    # First, handle special cases that might cause display issues
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    
    tokens = []
    # Process the text word by word, with spaces as separate tokens
    words = []
    current_word = ""
    
    for char in text:
        if char.isalnum() or char == "'" or char == "-":
            # Part of a word
            current_word += char
        elif char.isspace():
            # Handle spaces specially - preserve them as tokens
            if current_word:
                words.append(current_word)
                current_word = ""
            # Add the space as its own token (using visible space character)
            words.append(" ")
        else:
            # Not part of a word
            if current_word:
                words.append(current_word)
                current_word = ""
            
            # Add non-whitespace characters as separate tokens
            words.append(char)
    
    # Add the last word if there is one
    if current_word:
        words.append(current_word)
    
    # Simulate a more realistic tokenization by combining some words and splitting others
    i = 0
    while i < len(words):
        # Skip processing spaces - always keep them as separate tokens
        if words[i] == " ":
            tokens.append(words[i])
            i += 1
            continue
            
        # Randomly decide how to process this token
        choice = random.random()
        
        if choice < 0.7 or i == len(words) - 1:  
            # 70% chance: keep as a single token
            tokens.append(words[i])
            i += 1
        elif choice < 0.9:  
            # 20% chance: combine with next token if available and not a space
            if i + 1 < len(words) and words[i+1] != " " and len(words[i] + words[i+1]) < 20:
                # Only combine if result isn't too long
                combined = words[i] + words[i+1]
                tokens.append(combined)
                i += 2
            else:
                tokens.append(words[i])
                i += 1
        else:  
            # 10% chance: split into smaller tokens if possible
            word = words[i]
            if len(word) > 3:
                # Split somewhere in the middle
                split_point = random.randint(1, len(word)-1)
                tokens.append(word[:split_point])
                tokens.append(word[split_point:])
            else:
                tokens.append(word)
            i += 1
    
    # Filter out any empty tokens and ensure all tokens can be displayed
    tokens = [token for token in tokens if token]
    
    # Debug info
    print(f"Tokenized into {len(tokens)} tokens")
    print(f"First 10 tokens: {tokens[:10]}")
    
    return tokens

def assign_experts_to_tokens(tokens, scenario="weighted"):
    """
    Assign experts to each token using different scenarios:
    - random: completely random assignment (least realistic)
    - sequential: experts change in chunks (more realistic)
    - weighted: context-aware with momentum (most realistic)
    - alternating: cycles through phases with transitions
    
    Returns a list of tuples: (token, expert)
    """
    experts = ["simple", "balanced", "complex"]
    result = []
    
    if scenario == "random":
        # Completely random assignment - least realistic
        for token in tokens:
            expert = random.choice(experts)
            result.append((token, expert))
    
    elif scenario == "sequential":
        # Sequential chunks of tokens - more realistic than random
        chunk_size = random.randint(3, 8)  # Random chunk size
        current_expert = random.choice(experts)
        
        for i, token in enumerate(tokens):
            if i > 0 and i % chunk_size == 0:
                # Change expert randomly, avoiding the same expert twice in a row
                other_experts = [e for e in experts if e != current_expert]
                current_expert = random.choice(other_experts)
            
            result.append((token, current_expert))
    
    elif scenario == "weighted":
        # Context-aware with momentum - most realistic
        # This simulates how different types of tokens might be routed to different experts
        
        expert_weights = {
            "simple": 0.33,
            "balanced": 0.50,
            "complex": 0.17
        }
        
        # Initialize expert momentum (tendency to keep using the same expert)
        momentum = {expert: 1.0 for expert in experts}
        current_expert = "balanced"  # Start with balanced expert
        
        for token in tokens:
            # Adjust momentum based on token characteristics
            if len(token) <= 2 and token.islower():
                # Short, simple tokens favor the simple expert
                momentum["simple"] *= 1.2
            elif len(token) >= 6 or not token.isalpha():
                # Longer or non-alphabetic tokens favor complex expert
                momentum["complex"] *= 1.2
            else:
                # Medium length tokens favor balanced expert
                momentum["balanced"] *= 1.2
            
            # Apply base weights and momentum
            weighted_experts = {
                e: expert_weights[e] * momentum[e] for e in experts
            }
            
            # Normalize weights
            total_weight = sum(weighted_experts.values())
            if total_weight > 0:
                normalized_weights = {
                    e: w/total_weight for e, w in weighted_experts.items()
                }
                
                # Choose expert based on weighted probabilities
                r = random.random()
                cumulative = 0
                for expert, weight in normalized_weights.items():
                    cumulative += weight
                    if r <= cumulative:
                        current_expert = expert
                        break
            
            # Add token with selected expert
            result.append((token, current_expert))
            
            # Decay momentum for all experts
            for e in momentum:
                if e == current_expert:
                    momentum[e] = min(momentum[e] * 1.1, 3.0)  # Increase for current expert
                else:
                    momentum[e] = max(momentum[e] * 0.9, 0.5)  # Decrease for others
    
    elif scenario == "alternating":
        # Alternating phases with transitions
        phases = [
            {"simple": 0.7, "balanced": 0.3, "complex": 0.0},  # Simple phase
            {"simple": 0.3, "balanced": 0.6, "complex": 0.1},  # Balanced phase
            {"simple": 0.1, "balanced": 0.4, "complex": 0.5},  # Complex phase
        ]
        
        phase_length = len(tokens) // len(phases)
        current_phase = 0
        
        for i, token in enumerate(tokens):
            # Determine current phase
            if i > 0 and i % phase_length == 0 and current_phase < len(phases) - 1:
                current_phase += 1
            
            # Get weights for current phase
            weights = phases[current_phase]
            
            # Choose expert based on current phase weights
            r = random.random()
            cumulative = 0
            selected_expert = experts[-1]  # Default to last expert
            
            for expert, weight in weights.items():
                cumulative += weight
                if r <= cumulative:
                    selected_expert = expert
                    break
            
            result.append((token, selected_expert))
    
    return result

def send_tokens_to_backend(tokens, api_url="http://localhost:8000", batch_size=5, delay=0.2, enable_test_mode=False):
    """
    Send the tokens to the backend API in batches
    """
    # Disable test data generation first
    try:
        disable_response = requests.post(f"{api_url}/toggle-test-data", json={"enable": False})
        print(f"Disabled test data generation")
    except Exception as e:
        print(f"Warning: Failed to disable test data generation: {e}")
    
    # Clear any existing tokens
    try:
        clear_response = requests.post(f"{api_url}/clear-tokens")
        print(f"Cleared token queue at {api_url}/clear-tokens")
    except Exception as e:
        print(f"Warning: Failed to clear token queue: {e}")
    
    # Send tokens in batches
    batch_count = 0
    for i in range(0, len(tokens), batch_size):
        batch_count += 1
        batch = tokens[i:i+batch_size]
        
        # Format tokens for the TokenStreamData format that the backend expects
        token_data_list = [{"token": token, "expert": expert} for token, expert in batch]
        payload = {
            "tokens": token_data_list
        }
        
        try:
            response = requests.post(f"{api_url}/add-tokens", json=payload)
            if response.status_code == 200:
                print(f"Batch {batch_count}: Sent {len(batch)} tokens - {response.json()}")
            else:
                print(f"Error sending batch {batch_count}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Exception sending batch {batch_count}: {e}")
        
        # Add delay between batches for realistic streaming
        if i + batch_size < len(tokens):
            time.sleep(delay)
    
    print("\nTest completed successfully!")
    print("Check your frontend to see the streamed tokens with different colors")
    print("Each color represents a different expert:")
    print("  Color 1 (blue): simple expert")
    print("  Color 2 (green): balanced expert")
    print("  Color 3 (red): complex expert")
    
    # After sending tokens, enable test mode if requested
    if enable_test_mode:
        print("\nTest mode enabled - random tokens will now be generated")
        try:
            enable_response = requests.post(f"{api_url}/toggle-test-data", json={"enable": True})
            if enable_response.status_code == 200:
                print("Press Ctrl+C to stop the test mode")
                
                try:
                    # Keep the script running until interrupted
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    # Disable test mode when interrupted
                    requests.post(f"{api_url}/toggle-test-data", json={"enable": False})
                    print("Test mode disabled")
            else:
                print(f"Failed to enable test mode: {enable_response.status_code} - {enable_response.text}")
        except Exception as e:
            print(f"Exception enabling test mode: {e}")

def main():
    """
    Main entry point for the token streaming test script
    """
    parser = argparse.ArgumentParser(description="Test token streaming to backend")
    parser.add_argument("--scenario", type=str, choices=["random", "sequential", "weighted", "alternating"],
                        default="weighted",
                        help="Scenario for assigning experts to tokens (weighted is most realistic)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of tokens to send in each batch")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Delay between batches in seconds")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                        help="URL of the backend API")
    parser.add_argument("--enable-test-mode", action="store_true",
                        help="Enable test mode after sending real tokens")
    
    args = parser.parse_args()
    
    # Use a combination of sample texts for better variety
    full_text = " ".join([
        "The sun rises in the east and sets in the west.",
        "Birds fly in the sky. Fish swim in the water.",
        "Neural networks are computational systems inspired by the human brain.",
        "They consist of layers of neurons that process information.",
        "Quantum entanglement is a quantum mechanical phenomenon where pairs of particles become correlated in such a way that the quantum state of each particle cannot be described independently of the others."
    ])
    
    print(f"Using all sample texts combined: {full_text[:50]}...")
    
    # Tokenize the text
    print("\nTokenizing text...")
    tokens = tokenize_text(full_text)
    
    # Assign experts to tokens
    print(f"\nAssigning experts using '{args.scenario}' scenario:")
    print("  - random: Fully random expert selection (least realistic)")
    print("  - sequential: Experts change in chunks of tokens (more realistic)")
    print("  - weighted: Context-aware with momentum (most realistic)")
    print("  - alternating: Cycles through phases with transitions")
    
    token_experts = assign_experts_to_tokens(tokens, scenario=args.scenario)
    
    # Count the distribution of experts
    expert_counts = {expert: 0 for expert in ["simple", "balanced", "complex"]}
    for _, expert in token_experts:
        expert_counts[expert] += 1
    
    total_tokens = len(token_experts)
    print("\nExpert distribution:")
    for expert, count in expert_counts.items():
        percentage = count / total_tokens * 100 if total_tokens > 0 else 0
        print(f"  - {expert}: {count} tokens ({percentage:.1f}%)")
    
    # Show a sample of token-expert pairs
    print("\nSample token-expert pairs:")
    for i, (token, expert) in enumerate(token_experts[:8]):
        print(f"  Token {i+1}: '{token}' → Expert: {expert}")
    print("  ...")
    
    # Send the tokens to the backend
    print(f"\nSending {len(token_experts)} tokens to {args.api_url}")
    print(f"Using batch size {args.batch_size} and delay {args.delay}s...")
    
    send_tokens_to_backend(
        tokens=token_experts,
        api_url=args.api_url,
        batch_size=args.batch_size,
        delay=args.delay,
        enable_test_mode=args.enable_test_mode
    )

if __name__ == "__main__":
    main() 