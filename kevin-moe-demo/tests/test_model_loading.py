#!/usr/bin/env python3
"""
Utility script to test model loading with different configurations
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model_loading(model_path, use_weights_only=False, use_cpu=False):
    """
    Test loading a model with different configurations
    
    Args:
        model_path: HuggingFace model path or local directory
        use_weights_only: Whether to use weights_only=True for loading
        use_cpu: Force using CPU even if GPU is available
    """
    device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Testing model loading for: {model_path}")
    print(f"Device: {device}")
    print(f"weights_only: {use_weights_only}")
    print(f"{'='*80}\n")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Tokenizer loaded successfully")
        
        print("\nLoading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            weights_only=use_weights_only
        ).to(device)
        print("✓ Model loaded successfully")
        
        # Print model info
        params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters: {params:,}")
        print(f"Model type: {model.__class__.__name__}")
        
        # Test a simple inference
        print("\nTesting inference...")
        input_text = "Hello, I am a language model"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=20)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {input_text}")
        print(f"Output: {output_text}")
        
        print("\n✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Suggest solutions
        print("\nPossible solutions:")
        if "weights_only" in str(e):
            print("1. Try running again with --no-weights-only flag")
        print("2. Try a different model or model format (e.g., safetensors)")
        print("3. Check your PyTorch version and consider updating/downgrading")
        print("4. Ensure you have enough memory/disk space for this model")

def main():
    parser = argparse.ArgumentParser(description="Test Model Loading")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                      help="Path to the language model to test")
    parser.add_argument("--no-weights-only", action="store_true",
                      help="Disable weights_only mode for loading (less secure but more compatible)")
    parser.add_argument("--cpu", action="store_true",
                      help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    test_model_loading(
        model_path=args.model,
        use_weights_only=not args.no_weights_only,
        use_cpu=args.cpu
    )

if __name__ == "__main__":
    main() 