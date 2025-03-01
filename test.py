import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from pynput import keyboard
import re

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

    # Simplified and more direct system prompt
    system_prompt = """You are DeepSeek-R1, a helpful AI assistant. 
    Write clear, informative responses. 
    If the user injects a new thought or feedback, acknowledge it and adjust your response accordingly."""
    
    # More specific user prompt
    prompt = f"System: {system_prompt}\nUser: Write a detailed essay about the American Civil War covering causes, major battles, key figures, and its lasting impact. Include at least 5 paragraphs.\nAssistant:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    generate_with_thought_pivot(model, tokenizer, input_ids)

def generate_with_thought_pivot(model, tokenizer, input_ids):
    print("\nStreaming response:\nAssistant: ", end="", flush=True)
    
    tokens_generated = 0
    max_tokens = 1000  # Reduced to avoid excessive generation
    current_ids = input_ids.clone()
    past_key_values = None  # Store past attention states

    # More specific thought pivots
    thought_pivot_l = "\nUser: I'm finding this very interesting! Could you add more details about Lincoln's leadership?\nAssistant:"
    thought_pivot_j = "\nUser: This is getting boring. Could you focus more on the dramatic battle scenes and personal stories?\nAssistant:"
    
    # Create a shared variable to track key presses
    key_pressed = {'key': None}
    
    def on_press(key):
        try:
            if key.char == 'l':
                key_pressed['key'] = 'l'
                return True
            elif key.char == 'j':
                key_pressed['key'] = 'j'
                return True
        except AttributeError:
            # Not a character key
            pass
        return True  # Continue listening

    # Start listening for keypresses
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Keep track of all generated text for display
    generated_text = ""
    
    # Track repetition
    repetition_window = []
    repetition_threshold = 10  # Number of tokens to check for repetition
    
    # Buffer to accumulate tokens for checking tags
    token_buffer = ""
    
    # Flag to indicate we need to inject a closing tag
    inject_closing_tag = False

    try:
        while tokens_generated < max_tokens:
            # Check if a key was pressed
            if key_pressed['key'] == 'l':
                print("\n\n[Injecting Thought: More about Lincoln]\n")
                
                # Create a new prompt with the injected thought
                full_text = tokenizer.decode(current_ids[0])
                new_prompt = full_text + thought_pivot_l
                
                # Tokenize the new prompt
                current_ids = tokenizer(new_prompt, return_tensors="pt").input_ids.to(model.device)
                past_key_values = None  # Reset past key values for the new context
                tokens_generated = 0
                generated_text = ""  # Reset generated text
                repetition_window = []  # Reset repetition detection
                token_buffer = ""  # Reset token buffer
                inject_closing_tag = False  # Reset injection flag
                key_pressed['key'] = None  # Reset the key press
            
            elif key_pressed['key'] == 'j':
                print("\n\n[Injecting Thought: More Battle Details]\n")
                
                # Create a new prompt with the injected thought
                full_text = tokenizer.decode(current_ids[0])
                new_prompt = full_text + thought_pivot_j
                
                # Tokenize the new prompt
                current_ids = tokenizer(new_prompt, return_tensors="pt").input_ids.to(model.device)
                past_key_values = None  # Reset past key values for the new context
                tokens_generated = 0
                generated_text = ""  # Reset generated text
                repetition_window = []  # Reset repetition detection
                token_buffer = ""  # Reset token buffer
                inject_closing_tag = False  # Reset injection flag
                key_pressed['key'] = None  # Reset the key press
                
            # Check if we need to inject a closing tag
            if inject_closing_tag:
                print("</think>", end="", flush=True)
                
                # Add the closing tag to the context
                closing_tag = "</think>"
                closing_tag_ids = tokenizer(closing_tag, return_tensors="pt").input_ids.to(model.device)
                current_ids = torch.cat([current_ids, closing_tag_ids[0].unsqueeze(0)], dim=-1)
                
                # Reset the flag
                inject_closing_tag = False
                
                # Reset past key values to ensure the model processes the injected tag
                past_key_values = None
                continue

            with torch.no_grad():
                outputs = model(current_ids, past_key_values=past_key_values, return_dict=True, use_cache=True)
            
            next_token_logits = outputs.logits[:, -1, :]
            
            # Use temperature sampling with higher temperature for more diversity
            temperature = 0.9  # Increased from 0.7
            next_token_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Apply top-k sampling to prevent unlikely tokens
            top_k = 40
            indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
            
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Check if we've generated an EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                print("\n[End of sequence detected]")
                break
            
            token_text = tokenizer.decode(next_token_id[0])
            token_buffer += token_text
            
            # Check for opening think tag
            if "<think>" in token_buffer:
                # Print the opening tag
                print("<think>", end="", flush=True)
                
                # Remove the tag from the buffer
                token_buffer = token_buffer.replace("<think>", "")
                
                # Set the flag to inject a closing tag in the next iteration
                inject_closing_tag = True
            
            # Check for closing think tag (in case it's already there)
            if "</think>" in token_buffer:
                # Print the closing tag
                print("</think>", end="", flush=True)
                
                # Remove the tag from the buffer
                token_buffer = token_buffer.replace("</think>", "")
                
                # Reset the injection flag since we already have a closing tag
                inject_closing_tag = False
            
            # Print any remaining text in the buffer
            if token_buffer:
                print(token_buffer, end="", flush=True)
                generated_text += token_buffer
                token_buffer = ""
            
            # Enhanced repetition detection
            repetition_window.append(next_token_id.item())
            if len(repetition_window) > repetition_threshold:
                repetition_window.pop(0)
                
                # If we detect a repeating pattern, introduce more significant variation
                if is_repetitive(repetition_window):
                    print("\n[Detected repetition, introducing variation]")
                    
                    # Add a more substantial prompt to break out of the loop
                    variation_text = ". Moving on to another aspect, "
                    variation_ids = tokenizer(variation_text, return_tensors="pt").input_ids.to(model.device)
                    current_ids = torch.cat([current_ids, variation_ids[0].unsqueeze(0)], dim=-1)
                    past_key_values = None  # Reset context to break repetition
                    repetition_window = []
                    continue
            
            time.sleep(0.05)
            tokens_generated += 1

            # Continue normal token generation
            current_ids = torch.cat([current_ids, next_token_id], dim=-1)
            past_key_values = outputs.past_key_values
    
    finally:
        # Stop the listener when done
        listener.stop()
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

if __name__ == "__main__":
    main()
