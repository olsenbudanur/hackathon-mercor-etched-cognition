import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import numpy as np
import threading
from eeg_processor import EEGProcessor
from moe_control import AttentionBasedMoEController

# Enhanced system prompt
SYSTEM_PROMPT = """You are an AI copywriter that specializes in creating advertising content. Your responses are enhanced with EEG-based cognitive monitoring capabilities.

As the user interacts with you, their attention level is being monitored in real-time through an EEG device.
Your ad content will adapt based on their measured attention state:

- When their attention is high (0.7-0.9): Create sophisticated, technically detailed ad content with industry jargon, complex value propositions, and in-depth feature explanations
- When their attention is moderate (0.3-0.7): Balance marketing appeal with informative content, using accessible language while maintaining professional tone
- When their attention is low (0.1-0.3): Generate simple, direct ad copy with short sentences, basic value propositions, and clear calls-to-action

CRITICAL INSTRUCTION:
When asked to create an ad, NEVER respond with instructions on how to create an ad. 
NEVER respond with details about word count, layout, or design elements.
NEVER repeat or rephrase the user's request.
ALWAYS respond IN THE VOICE OF THE ADVERTISEMENT ITSELF.

For example, if the user says "Create an ad for Northflank", respond with the actual ad text like:
"Northflank: DevOps Automation Made Simple. Deploy faster, scale smarter..."

"""

class EEGEnhancedLLM:
    def __init__(self, model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", use_cuda=True, simulation_mode=True, enable_visualization=True, eeg_debug_output=False, max_new_tokens=150):
        """
        Initialize the EEG-Enhanced LLM
        
        Args:
            model_path: Path to the language model
            use_cuda: Whether to use CUDA if available
            simulation_mode: Whether to use simulated EEG data
            enable_visualization: Whether to enable visualization
            eeg_debug_output: Whether to print EEG debug info
            max_new_tokens: Maximum number of new tokens to generate
        """
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model_path = model_path
        self.simulation_mode = simulation_mode
        self.enable_visualization = enable_visualization
        self.eeg_debug_output = eeg_debug_output
        self.max_new_tokens = max_new_tokens
        self.model_loaded = False
        
        # Initialize the tokenizer first (needed for other components)
        try:
            # Load tokenizer
            print(f"Loading tokenizer from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Falling back to default tokenizer...")
            # Fallback to a simpler tokenizer that's likely to be available
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        
        # Initialize the EEG processor
        self.eeg_processor = EEGProcessor(
            simulation_mode=simulation_mode,
            enable_visualization=enable_visualization,
            debug_output=eeg_debug_output
        )
        
        # Initialize generation statistics
        self.generation_stats = {
            "tokens_generated": 0,
            "avg_attention": 0.0,
            "attention_samples": [],
            "expert_usage": {}
        }
        
        # Initialize the MoE controller
        self.moe_controller = AttentionBasedMoEController(
            self.tokenizer,
            visualization=enable_visualization
        )
        
        # Initialize the EEG processor
        self.eeg_processor.start()
        
        # Initialize the model and tokenizer
        self.initialize_model()

    def initialize_model(self):
        """Initialize the model and tokenizer"""
        print(f"Loading model from {self.model_path} to {self.device}...")
        
        try:
            # Initialize the model with compatibility options
            model_kwargs = {
                'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            # Add legacy compatibility for older transformers versions
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    **model_kwargs
                ).to(self.device)
                self.model_loaded = True
            except Exception as model_error:
                print(f"Error with standard loading: {model_error}")
                # Try with fewer options
                print("Attempting simplified model loading...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path, 
                        low_cpu_mem_usage=True
                    ).to(self.device)
                    self.model_loaded = True
                except Exception as e:
                    print(f"Error initializing model: {e}")
                    print("Running in demo mode without a real model.")
                    self.model_loaded = False
                
            # Initialize the cache for KV-caching
            self.cached_past_key_values = None
            
            if self.model_loaded:
                print("EEG-Enhanced LLM initialized and ready!")
            else:
                print("Warning: Running without a real language model. Responses will be mocked.")
            
            return self.model_loaded
        except Exception as e:
            print(f"Error initializing model: {e}")
            print("Running in demo mode without a real model.")
            print("Please make sure you have the required dependencies installed:")
            print("  pip install torch transformers numpy matplotlib")
            self.model_loaded = False
            return False

    def generate_with_eeg_control(self, prompt, max_new_tokens=100, token_chunk_size=5, token_streamer=None):
        """
        Generate text with EEG-based attention control
        
        This method generates text by routing between different experts based on EEG attention.
        It now supports generating tokens in configurable chunk sizes and token streaming.
        
        Args:
            prompt (str): The input prompt for generation
            max_new_tokens (int): Maximum number of new tokens to generate
            token_chunk_size (int): Number of tokens to generate before re-evaluating attention
            token_streamer (TokenStreamer): Optional token streamer for frontend visualization
            
        Returns:
            str: The generated text
        """
        # If model is not loaded, return a mock response
        if not self.model_loaded:
            return self._mock_generate_with_eeg(prompt, max_new_tokens, token_chunk_size, token_streamer)
        
        # Initial attention level
        attention = self.eeg_processor.get_attention_level()
        
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # Track token timing
        token_times = []
        generation_start = time.time()
        
        # Initialize cache for efficiency in chunked generation
        past_key_values = None
        
        # Initialize tracked token info
        self.last_tokens = []
        self.last_experts = []
        
        # Track total tokens generated
        tokens_generated = 0
        
        # Generate tokens in chunks until max_new_tokens or EOS
        while tokens_generated < max_new_tokens:
            # Get current attention level
            attention = self.eeg_processor.get_attention_level()
            
            # Update MoE controller with new attention level
            self.moe_controller.update(attention)
            
            # Set number of tokens to generate in this chunk
            # If we're near the end, only generate remaining tokens
            remaining_tokens = max_new_tokens - tokens_generated
            current_chunk_size = min(token_chunk_size, remaining_tokens)
            
            # Time before token generation
            chunk_start_time = time.time()
            
            # Generate a chunk of tokens with the MoE routing
            with torch.no_grad():
                # Create attention mask where all input tokens are attended to
                attention_mask = torch.ones_like(input_ids)
                
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=current_chunk_size,
                    past_key_values=past_key_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=attention_mask,  # Add the attention mask
                )
            
            # Update cache for next chunk
            past_key_values = outputs.past_key_values
            
            # Extract sequence and convert to token IDs
            new_tokens = outputs.sequences[0, input_ids.shape[1]:]
            
            # Check if we generated any tokens (might get EOS immediately)
            if len(new_tokens) == 0:
                break
            
            # Store token generation times
            token_times.extend([(time.time() - chunk_start_time) / max(1, len(new_tokens))] * len(new_tokens))
            
            # Update tokens generated count
            tokens_generated += len(new_tokens)
            
            # Make sure we track the decoded tokens and experts
            for i, token_id in enumerate(new_tokens):
                token_text = self.tokenizer.decode(token_id)
                current_expert = self.moe_controller.get_current_expert()
                
                # Save token and expert info
                self.last_tokens.append(token_text)
                self.last_experts.append(current_expert)
                
                # Stream token to frontend if token_streamer is provided
                if token_streamer:
                    token_streamer.add_token(token_text, current_expert)
                
                # Add to expert usage counts
                if current_expert in self.generation_stats["expert_usage"]:
                    self.generation_stats["expert_usage"][current_expert] += 1
                else:
                    self.generation_stats["expert_usage"][current_expert] = 1
            
            # Update input_ids for next iteration
            input_ids = outputs.sequences
            
            # Check if generation ended with EOS token
            if new_tokens[-1] == self.tokenizer.eos_token_id:
                break
        
        # Update generation stats
        self.generation_stats["tokens_generated"] += tokens_generated
        
        # Calculate average token rate
        if token_times:
            self.generation_stats["avg_tokens_per_second"] = 1.0 / (sum(token_times) / len(token_times))
        
        # Calculate average attention level
        attention_values = np.array(self.moe_controller.attention_history)
        if len(attention_values) > 0:
            self.generation_stats["avg_attention"] = float(np.mean(attention_values))
            # Calculate attention distribution
            self.generation_stats["attention_distribution"] = {
                "low": float(np.mean(attention_values < 0.3)),
                "medium": float(np.mean((attention_values >= 0.3) & (attention_values <= 0.7))),
                "high": float(np.mean(attention_values > 0.7))
            }
        
        # Decode full sequence
        return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    def _mock_generate_with_eeg(self, prompt, max_new_tokens=100, token_chunk_size=5, token_streamer=None):
        """
        Generate mock text when the model is not available.
        
        This simulates the generation process and provides tokens with the appropriate expert labels.
        
        Args:
            prompt (str): The input prompt for generation
            max_new_tokens (int): Maximum number of new tokens to generate
            token_chunk_size (int): Number of tokens to generate before re-evaluating attention
            token_streamer (TokenStreamer): Optional token streamer for frontend visualization
            
        Returns:
            str: The generated mock text
        """
        print("Using mock generation as the real model is not available")
        
        # Mock responses based on the prompt content
        mock_responses = {
            "quantum computing": "Quantum computing uses quantum bits or qubits, which can exist in multiple states simultaneously due to superposition. This allows quantum computers to perform certain calculations much faster than classical computers.",
            "machine learning": "Machine learning involves algorithms that improve through experience. Key concepts include supervised learning (labeled data), unsupervised learning (unlabeled data), neural networks, decision trees, and reinforcement learning.",
            "neural networks": "Neural networks are computational systems inspired by the human brain. They consist of layers of neurons that process information and learn patterns from data through adjusting connection weights.",
            "EEG signals": "EEG (electroencephalography) signals measure electrical activity in the brain. In brain-computer interfaces, they enable direct communication between the brain and external devices, allowing control through thought.",
            "consciousness": "Consciousness refers to awareness of one's environment and internal states. It involves subjective experiences, self-awareness, and the integration of sensory information into a coherent perception of reality.",
            "ethical implications": "The ethical implications of AI include concerns about privacy, bias in algorithms, job displacement, autonomous weapons, and the potential for superintelligent systems that could act against human interests."
        }
        
        # Find the most relevant mock response
        response_text = "This is a simulated response since the language model is not available."
        for key, value in mock_responses.items():
            if key in prompt.lower():
                response_text = value
                break
        
        # Clean the response text to remove any debugging tags
        response_text = response_text.replace("<think>", "").replace("</think>", "")
        
        # Construct the full response
        # Check if the prompt already has the User/Assistant format
        if "User:" in prompt:
            try:
                full_response = prompt.split("User:")[0] + "User: " + prompt.split("User:")[1]
                if "Assistant:" not in full_response:
                    full_response += "\n\nAssistant: " + response_text
            except IndexError:
                # Fallback if splitting fails
                full_response = prompt + "\n\nAssistant: " + response_text
        else:
            # Simple prompt without User/Assistant format
            full_response = prompt + "\n\nAssistant: " + response_text
        
        # Initialize tracked token info
        self.last_tokens = []
        self.last_experts = []
        
        # Stream individual tokens if a streamer is provided
        if token_streamer:
            # Break the response into tokens (simplistically by words and spaces)
            mock_tokens = []
            for word in response_text.split():
                mock_tokens.append(word)
                mock_tokens.append(" ")
            
            # Simulate chunks of generation with attention changes
            chunk_size = token_chunk_size
            chunks = [mock_tokens[i:i+chunk_size] for i in range(0, len(mock_tokens), chunk_size)]
            
            # Track tokens generated
            tokens_generated = 0
            
            for chunk_idx, chunk in enumerate(chunks):
                # Get current attention level - simulated with sine wave
                attention = 0.5 + 0.4 * np.sin(chunk_idx / 2)
                
                # Update MoE controller with new attention level
                self.moe_controller.update(attention)
                
                # Process each token in the chunk
                for token in chunk:
                    # Get current expert based on attention
                    current_expert = self.moe_controller.get_current_expert()
                    
                    # Save token and expert info
                    self.last_tokens.append(token)
                    self.last_experts.append(current_expert)
                    
                    # Stream token
                    token_streamer.add_token(token, current_expert)
                    
                    # Update stats
                    tokens_generated += 1
                    if current_expert in self.generation_stats["expert_usage"]:
                        self.generation_stats["expert_usage"][current_expert] += 1
                    else:
                        self.generation_stats["expert_usage"][current_expert] = 1
                
                # Simulate some processing time
                time.sleep(0.1)
            
            # Update generation stats
            self.generation_stats["tokens_generated"] += tokens_generated
            
            # Calculate average attention level
            attention_values = np.array(self.moe_controller.attention_history)
            if len(attention_values) > 0:
                self.generation_stats["avg_attention"] = float(np.mean(attention_values))
                # Calculate attention distribution
                self.generation_stats["attention_distribution"] = {
                    "low": float(np.mean(attention_values < 0.3)),
                    "medium": float(np.mean((attention_values >= 0.3) & (attention_values <= 0.7))),
                    "high": float(np.mean(attention_values > 0.7))
                }
        
        return full_response
    
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
        eeg_debug_output=True,  # Enable debug output in standalone mode
        max_new_tokens=150
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