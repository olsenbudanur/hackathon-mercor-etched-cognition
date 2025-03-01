#!/usr/bin/env python3
"""
Integration test for the EEG-Enhanced LLM system with a small model
"""

import time
import argparse
from main_eeg import EEGEnhancedLLM, SYSTEM_PROMPT

def test_integration(smaller_model="facebook/opt-125m"):
    """
    Run an integration test with a smaller model
    
    Args:
        smaller_model: Path to a smaller model for faster testing
    """
    print("\n" + "="*80)
    print(" "*20 + "EEG-ENHANCED LLM INTEGRATION TEST")
    print("="*80)
    print(f"\nInitializing system with model: {smaller_model}")
    
    # Initialize the system with visualization disabled
    llm = EEGEnhancedLLM(
        model_path=smaller_model,
        simulation_mode=True,
        enable_visualization=False
    )
    
    print("\nSystem initialized! Running integration test...")
    
    # Test prompt
    test_prompt = "Explain the concept of artificial intelligence in simple terms."
    full_prompt = SYSTEM_PROMPT + f"\n\nUser: {test_prompt}\n\nAssistant: "
    
    try:
        # Test with different attention levels
        attention_levels = [0.2, 0.5, 0.8]
        responses = []
        
        for i, attention in enumerate(attention_levels):
            print(f"\n[Test {i+1}/3] Using attention level: {attention:.1f}")
            
            # Set attention level manually
            llm.eeg_processor.attention_level = attention
            
            # Generate response
            start_time = time.time()
            result = llm.generate_with_eeg_control(full_prompt, max_new_tokens=50)
            end_time = time.time()
            
            # Store response
            response = result.split("Assistant: ")[-1]
            responses.append(response)
            
            # Print results
            print(f"\nGeneration time: {end_time - start_time:.2f} seconds")
            print("\nGenerated response:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            
            # Get generation stats
            stats = llm.get_generation_stats()
            print("\nGeneration Statistics:")
            print(f"- Tokens generated: {stats['tokens_generated']}")
            print(f"- Average attention: {stats['avg_attention']:.2f}")
            
            # Print expert usage
            print("\nExpert Usage:")
            for expert, count in stats.get("expert_usage", {}).items():
                if stats['tokens_generated'] > 0:
                    percentage = count/stats['tokens_generated']*100
                    print(f"- {expert}: {count} tokens ({percentage:.1f}%)")
            
            # Wait between tests
            if i < len(attention_levels) - 1:
                print("\nWaiting 2 seconds before next test...")
                time.sleep(2)
        
        # Analyze responses
        print("\n" + "="*80)
        print(" "*30 + "RESPONSE ANALYSIS")
        print("="*80)
        
        # Compare response lengths
        print("\nResponse Length Comparison:")
        for i, (attention, response) in enumerate(zip(attention_levels, responses)):
            print(f"- Attention {attention:.1f}: {len(response)} characters, {len(response.split())} words")
        
        # Other metrics could be added here (complexity analysis, etc.)
        
    finally:
        # Clean up
        llm.cleanup()
        print("\nIntegration test completed.")

def main():
    parser = argparse.ArgumentParser(description="EEG-Enhanced LLM Integration Test")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="Path to a small language model for testing")
    
    args = parser.parse_args()
    test_integration(args.model)

if __name__ == "__main__":
    main() 