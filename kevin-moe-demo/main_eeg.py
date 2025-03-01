import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import numpy as np
import threading
from eeg_processor import EEGProcessor
from moe_control import AttentionBasedMoEController

# Enhanced system prompt
SYSTEM_PROMPT = """You are a knowledgeable AI assistant providing factual information on various topics. Your responses are enhanced with EEG-based cognitive monitoring capabilities.

As the user interacts with you, their attention level is being monitored in real-time through an EEG device.
Your responses will adapt based on their measured attention state:

- When their attention is high (0.7-0.9): Provide more detailed, technically rich information with precise terminology
- When their attention is moderate (0.3-0.7): Balance detail with clarity, using accessible explanations
- When their attention is low (0.1-0.3): Simplify your explanations using basic language and shorter sentences

For any topic, provide factual, educational content rather than personal opinions. Focus on explaining concepts clearly and accurately.

Your goal is to educate the user while maintaining their engagement by dynamically adjusting to their cognitive state.
"""

class EEGEnhancedLLM:
    def __init__(self, model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                 use_cuda=True, simulation_mode=True, enable_visualization=True, eeg_debug_output=False):
        """
        Initialize the EEG-enhanced LLM system
        
        Args:
            model_path: Path to the model
            use_cuda: Whether to use CUDA for model inference
            simulation_mode: Whether to use simulated EEG data
            enable_visualization: Whether to enable visualization
            eeg_debug_output: Whether to enable debug output from the EEG processor
        """
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        print(f"Loading model from {model_path} to {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            weights_only=False  # Fix for PyTorch 2.6+ compatibility issue
        ).to(self.device)
        
        # Initialize EEG processor
        self.eeg_processor = EEGProcessor(
            simulation_mode=simulation_mode,
            enable_visualization=enable_visualization,
            debug_output=eeg_debug_output
        )
        
        # Initialize MoE controller
        self.moe_controller = AttentionBasedMoEController(
            tokenizer=self.tokenizer,
            visualization=enable_visualization
        )
        
        # Set up generation parameters
        self.max_new_tokens = 250
        self.stopping_criteria = None
        
        # Track token generation stats
        self.generation_stats = {
            "tokens_generated": 0,
            "avg_attention": 0.0,
            "attention_samples": [],
            "expert_usage": {}
        }
        
        # Cache for faster generation
        self.cached_past_key_values = None
        self.cached_input_ids = None
        
        # Initialize the EEG processor
        self.eeg_processor.start()
        
        print("EEG-Enhanced LLM initialized and ready!")

    def generate_with_eeg_control(self, prompt, max_new_tokens=None):
        """
        Generate text with EEG-based control using HuggingFace's generate method with KV-caching
        
        Args:
            prompt: Input prompt for text generation
            max_new_tokens: Maximum number of new tokens to generate
        
        Returns:
            Generated text
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
            
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        prompt_length = input_ids.shape[1]
        
        # Initialize tracking
        attention_values = []
        current_text = ""
        
        # Process one token at a time to apply MoE control
        for i in range(max_new_tokens):
            # Get current attention level from EEG
            current_attention = self.eeg_processor.get_attention_level()
            attention_values.append(current_attention)
            
            # Update MoE controller with current attention
            self.moe_controller.update(current_attention, new_tokens=1)
            
            # Get dynamic generation parameters from MoE controller
            gen_params = self.moe_controller.get_generation_params()
            
            # Generate a single token with HF generate, leveraging KV-cache
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids if i == 0 else input_ids[:, -1:],  # Only pass the last token after first iteration
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=gen_params["temperature"],
                    top_k=gen_params["top_k"],
                    repetition_penalty=gen_params["repetition_penalty"],
                    use_cache=True,
                    past_key_values=self.cached_past_key_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            # Update the KV-cache for next iteration
            self.cached_past_key_values = outputs.past_key_values
            
            # Get the newly generated token
            next_token = outputs.sequences[:, -1:]
            
            # Apply MoE biasing to next token prediction
            if i < max_new_tokens - 1:  # No need to bias the last token
                # Get logits for next prediction
                next_token_logits = outputs.scores[0]
                
                # Apply MoE-based logit biasing
                biased_logits = self.moe_controller.apply_moe_logit_biasing(next_token_logits)
                
                # Replace original logits with biased ones for next generation
                outputs.scores = (biased_logits,) + outputs.scores[1:] if len(outputs.scores) > 1 else (biased_logits,)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
            # Track generation stats
            self.generation_stats["tokens_generated"] += 1
            
            # Get current dominant expert
            current_expert = self.moe_controller.get_current_expert()
            
            # Update expert usage stats
            if current_expert not in self.generation_stats["expert_usage"]:
                self.generation_stats["expert_usage"][current_expert] = 0
            self.generation_stats["expert_usage"][current_expert] += 1
            
            # Print debug info
            next_token_text = self.tokenizer.decode([next_token.item()])
            tokens_generated = i + 1
            print(f"\rTokens: {tokens_generated}/{max_new_tokens} | " 
                  f"Attention: {current_attention:.2f} | "
                  f"Expert: {current_expert} | "
                  f"Last token: {next_token_text}", end="")
        
        print()  # New line after generation
        
        # Update overall generation stats
        if attention_values:
            self.generation_stats["avg_attention"] = sum(attention_values) / len(attention_values)
            self.generation_stats["attention_samples"].extend(attention_values)
        
        # Decode and return the generated text
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    def get_generation_stats(self):
        """Get statistics about the generation process"""
        stats = self.generation_stats.copy()
        
        # Add MoE controller stats
        stats["expert_stats"] = self.moe_controller.get_expert_stats()
        
        # Calculate attention distribution
        if stats["attention_samples"]:
            attention_array = np.array(stats["attention_samples"])
            stats["attention_distribution"] = {
                "low": np.mean(attention_array < 0.3),
                "medium": np.mean((attention_array >= 0.3) & (attention_array <= 0.7)),
                "high": np.mean(attention_array > 0.7)
            }
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'eeg_processor'):
            self.eeg_processor.stop()
            print("EEG processor stopped.")

def main():
    """Main function to run the EEG-enhanced LLM"""
    # Initialize the system
    llm = EEGEnhancedLLM(
        model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        simulation_mode=True,
        enable_visualization=True,
        eeg_debug_output=True  # Enable debug output in standalone mode
    )
    
    try:
        # Initial system prompt
        result = llm.generate_with_eeg_control(SYSTEM_PROMPT + "\n\nUser: Explain quantum computing to me.\n\nAssistant: ")
        print("\n\nGenerated text:")
        print(result)
        
        # Print generation stats
        print("\nGeneration Statistics:")
        stats = llm.get_generation_stats()
        print(f"Total tokens generated: {stats['tokens_generated']}")
        print(f"Average attention level: {stats['avg_attention']:.2f}")
        
        if "attention_distribution" in stats:
            att_dist = stats["attention_distribution"]
            print(f"Attention distribution: Low: {att_dist['low']:.2f}, "
                  f"Medium: {att_dist['medium']:.2f}, High: {att_dist['high']:.2f}")
        
        print("\nExpert usage:")
        for expert, count in stats["expert_usage"].items():
            print(f"  {expert}: {count} tokens ({count/stats['tokens_generated']*100:.1f}%)")
        
        # Interactive mode
        while True:
            user_input = input("\nEnter your question (or 'q' to quit): ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
                
            full_prompt = SYSTEM_PROMPT + f"\n\nUser: {user_input}\n\nAssistant: "
            result = llm.generate_with_eeg_control(full_prompt)
            
            print("\nGenerated text:")
            print(result)
    
    finally:
        # Clean up
        llm.cleanup()
        print("Session ended.")

if __name__ == "__main__":
    main() 