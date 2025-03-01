import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from pynput import keyboard
import re
import numpy as np
import threading

def main():
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Enhanced system prompt for better steerability
    system_prompt = """You are DeepSeek-R1, a helpful AI assistant. 
    Write clear, informative responses.
    You will sometimes receive thought injections or guidance during your response generation.
    When this happens, incorporate this guidance naturally into your reasoning process.
    If the guidance suggests a new direction or additional details, adjust your response accordingly.
    Use <think>...</think> tags for your internal reasoning when appropriate."""
    
    # More specific user prompt
    prompt = f"System: {system_prompt}\nUser: Write a detailed essay about the American Civil War covering causes, major battles, key figures, and its lasting impact. Include at least 5 paragraphs.\nAssistant:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Simulate EEG attention levels - will be replaced with real EEG data
    eeg_simulator = EEGSimulator()
    eeg_simulator.start()
    
    try:
        generate_with_advanced_control(model, tokenizer, input_ids, eeg_simulator)
    finally:
        eeg_simulator.stop()

class EEGSimulator:
    """Simulates EEG attention signals for development and testing"""
    def __init__(self):
        self.attention_level = 0.5  # Default mid-level attention
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the simulation thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_simulation)
        self.thread.daemon = True
        self.thread.start()
        print("EEG Simulator started. Use keys 1-9 to adjust attention level (1=low, 9=high)")
    
    def stop(self):
        """Stop the simulation thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def get_attention(self):
        """Get current simulated attention level (0.0 to 1.0)"""
        return self.attention_level
    
    def _run_simulation(self):
        """Run the simulation, listening for keyboard input"""
        def on_press(key):
            try:
                if key.char in "123456789":
                    # Set attention level based on key press (1-9)
                    self.attention_level = float(key.char) / 9.0
                    print(f"\n[EEG Simulator] Attention level: {self.attention_level:.2f}")
            except AttributeError:
                pass
            return self.running
        
        # Start keyboard listener
        with keyboard.Listener(on_press=on_press) as listener:
            while self.running:
                time.sleep(0.1)
            listener.stop()

def generate_with_advanced_control(model, tokenizer, input_ids, eeg_simulator):
    """
    Generate text with advanced control including:
    - Token-by-token generation
    - Direct logit biasing based on EEG attention
    - Injection of thinking tokens
    - Handling of repetition
    """
    print("\nStreaming response:\nAssistant: ", end="", flush=True)
    
    tokens_generated = 0
    max_tokens = 1000
    current_ids = input_ids.clone()
    past_key_values = None
    
    # Define thought pivots for different attention levels
    thought_pivots = {
        "low": "<think>I notice the reader's attention is dropping. I should simplify my explanation and focus on more engaging examples or stories.</think>",
        "medium": "<think>The reader seems moderately engaged. I'll continue with balanced content but add some interesting details to maintain interest.</think>",
        "high": "<think>The reader is highly engaged! I should provide more in-depth analysis and complex details since they're following closely.</think>"
    }
    
    # Track for injection control
    last_attention_check = 0
    attention_check_interval = 10  # Check attention every N tokens
    last_injection_point = 0
    min_injection_gap = 50  # Minimum tokens between injections
    
    # EEG thresholds
    low_threshold = 0.3
    high_threshold = 0.7
    
    # Keep track of all generated text
    generated_text = ""
    
    # Track repetition
    repetition_window = []
    repetition_threshold = 10
    
    # Buffer to accumulate tokens for checking tags
    token_buffer = ""
    
    # Flag to indicate we need to inject a closing tag
    inject_closing_tag = False
    
    # Track if we're inside a thought section
    in_thought_section = False
    
    try:
        while tokens_generated < max_tokens:
            # Check if we need to evaluate attention and possibly inject thought
            if tokens_generated - last_attention_check >= attention_check_interval:
                attention = eeg_simulator.get_attention()
                print(f"\n[DEBUG] Current attention level: {attention:.2f}", end="", flush=True)
                last_attention_check = tokens_generated
                
                # Only inject if we haven't recently injected
                if tokens_generated - last_injection_point >= min_injection_gap and not in_thought_section:
                    # Determine which thought to inject based on attention
                    if attention < low_threshold:
                        print("\n\n[Injecting Thought: Low Attention]\n", end="", flush=True)
                        thought_pivot = thought_pivots["low"]
                        last_injection_point = tokens_generated
                    elif attention > high_threshold:
                        print("\n\n[Injecting Thought: High Attention]\n", end="", flush=True)
                        thought_pivot = thought_pivots["high"]
                        last_injection_point = tokens_generated
                    elif tokens_generated > 200 and tokens_generated % 200 == 0:
                        # Occasionally inject medium thought to demonstrate capability
                        print("\n\n[Injecting Thought: Medium Attention]\n", end="", flush=True)
                        thought_pivot = thought_pivots["medium"]
                        last_injection_point = tokens_generated
                    else:
                        thought_pivot = None
                    
                    # Inject the thought if applicable
                    if thought_pivot:
                        # Tokenize the thought pivot
                        thought_ids = tokenizer(thought_pivot, return_tensors="pt").input_ids[0].to(model.device)
                        
                        # Add the thought to the current input
                        current_ids = torch.cat([current_ids, thought_ids.unsqueeze(0)], dim=1)
                        
                        # Reset past key values to ensure the model processes the injected thought
                        past_key_values = None
                        in_thought_section = True
                        continue
            
            # Check if we need to inject a closing tag
            if inject_closing_tag:
                print("</think>", end="", flush=True)
                
                # Add the closing tag to the context
                closing_tag = "</think>"
                closing_tag_ids = tokenizer(closing_tag, return_tensors="pt").input_ids.to(model.device)
                current_ids = torch.cat([current_ids, closing_tag_ids[0].unsqueeze(0)], dim=-1)
                
                # Reset the flag
                inject_closing_tag = False
                in_thought_section = False
                
                # Reset past key values to ensure the model processes the injected tag
                past_key_values = None
                continue

            # Get model outputs for next token prediction
            with torch.no_grad():
                outputs = model(current_ids, past_key_values=past_key_values, return_dict=True, use_cache=True)
            
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply attention-based logit biasing
            if eeg_simulator.get_attention() < low_threshold:
                # Simplify language when attention is low (boost common tokens)
                common_token_ids = get_common_token_ids(tokenizer, 100)
                next_token_logits[:, common_token_ids] *= 1.2  # Boost common tokens
            elif eeg_simulator.get_attention() > high_threshold:
                # Encourage complex language when attention is high (boost rare tokens)
                rare_token_ids = get_rare_token_ids(tokenizer, 100)
                next_token_logits[:, rare_token_ids] *= 1.2  # Boost rare tokens
            
            # Use dynamic temperature based on attention
            attention = eeg_simulator.get_attention()
            # Higher attention -> lower temperature (more focused/deterministic)
            # Lower attention -> higher temperature (more variety/simplicity)
            temperature = 1.1 - (attention * 0.4)  # Maps 0->1.1, 1->0.7
            next_token_logits = next_token_logits / temperature
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Apply top-k sampling with dynamic k based on attention
            # Lower attention -> lower k (focus on more likely tokens)
            k = int(20 + (attention * 40))  # Maps 0->20, 1->60
            indices_to_remove = probs < torch.topk(probs, k)[0][..., -1, None]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
            
            # Sample next token
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Check if we've generated an EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                print("\n[End of sequence detected]")
                break
            
            # Decode the token
            token_text = tokenizer.decode(next_token_id[0])
            token_buffer += token_text
            
            # Check for opening think tag
            if "<think>" in token_buffer and not in_thought_section:
                # Print the opening tag
                print("<think>", end="", flush=True)
                
                # Remove the tag from the buffer
                token_buffer = token_buffer.replace("<think>", "")
                
                # Set the flag to inject a closing tag in the next iteration
                inject_closing_tag = True
                in_thought_section = True
            
            # Check for closing think tag (in case it's already there)
            if "</think>" in token_buffer and in_thought_section:
                # Print the closing tag
                print("</think>", end="", flush=True)
                
                # Remove the tag from the buffer
                token_buffer = token_buffer.replace("</think>", "")
                
                # Reset the injection flag
                inject_closing_tag = False
                in_thought_section = False
            
            # Print any remaining text in the buffer
            if token_buffer:
                print(token_buffer, end="", flush=True)
                generated_text += token_buffer
                token_buffer = ""
            
            # Enhanced repetition detection
            repetition_window.append(next_token_id.item())
            if len(repetition_window) > repetition_threshold:
                repetition_window.pop(0)
                
                # If we detect a repeating pattern, introduce variation
                if is_repetitive(repetition_window):
                    print("\n[Detected repetition, introducing variation]")
                    
                    # Attention-aware variation
                    attention = eeg_simulator.get_attention()
                    if attention < low_threshold:
                        variation_text = ". To simplify, "
                    elif attention > high_threshold:
                        variation_text = ". Diving deeper, "
                    else:
                        variation_text = ". Moving on to another aspect, "
                    
                    variation_ids = tokenizer(variation_text, return_tensors="pt").input_ids.to(model.device)
                    current_ids = torch.cat([current_ids, variation_ids[0].unsqueeze(0)], dim=-1)
                    past_key_values = None  # Reset context to break repetition
                    repetition_window = []
                    continue
            
            # Add a small delay for better visualization
            time.sleep(0.05)
            tokens_generated += 1

            # Continue normal token generation
            current_ids = torch.cat([current_ids, next_token_id], dim=-1)
            past_key_values = outputs.past_key_values
    
    finally:
        # If we still need to inject a closing tag, do it now
        if inject_closing_tag:
            print("</think>", end="", flush=True)
        print("\nfinal full text: ", tokenizer.decode(current_ids[0]))

def is_repetitive(token_list):
    """Enhanced check for repetitive patterns"""
    # Check for direct repetition (same token multiple times)
    for i in range(len(token_list) - 3):
        if token_list[i] == token_list[i+1] == token_list[i+2]:
            return True
            
    # Check for pattern repetition
    for i in range(len(token_list) - 5):
        if token_list[i] == token_list[i+2] == token_list[i+4] and token_list[i+1] == token_list[i+3]:
            return True
            
    return False

def get_common_token_ids(tokenizer, n=100):
    """Get IDs of common tokens (placeholder implementation)"""
    # This would ideally be based on token frequency in a corpus
    # For demonstration, just return some arbitrary token IDs
    return torch.tensor(list(range(100, 100+n)), device="cuda" if torch.cuda.is_available() else "cpu")

def get_rare_token_ids(tokenizer, n=100):
    """Get IDs of rare/complex tokens (placeholder implementation)"""
    # This would ideally be based on token frequency in a corpus
    # For demonstration, just return some arbitrary token IDs
    vocab_size = tokenizer.vocab_size
    return torch.tensor(list(range(vocab_size-n, vocab_size)), device="cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    main()
