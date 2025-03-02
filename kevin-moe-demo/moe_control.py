import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

class AttentionBasedMoEController:
    """
    Implements a simulated Mixture-of-Experts approach to control model generation
    based on attention levels from EEG signals.
    
    Each "expert" is a different configuration of generation parameters optimized for
    a specific cognitive state (low attention, medium attention, high attention).
    """
    
    def __init__(self, tokenizer, num_experts=3, visualization=True):
        """
        Initialize the MoE controller
        
        Args:
            tokenizer: The tokenizer for the model
            num_experts: Number of experts to simulate
            visualization: Whether to show real-time visualization
        """
        self.tokenizer = tokenizer
        self.num_experts = num_experts
        self.visualization = visualization
        
        # Expert definitions
        self.experts = {
            "simple": {
                "name": "Simple Ad Expert",
                "description": "Optimized for low attention - creates direct, concise ad copy with clear benefits",
                "temperature": 1.2,
                "top_k": 20,
                "repetition_penalty": 1.05,
                "token_boost_ids": self._get_common_tokens(100),
                "token_boost_factor": 1.5,
                "style_tokens": ["simple", "easy", "quick", "free", "save", "now", "try", "get", "start", "join"]
            },
            "balanced": {
                "name": "Standard Ad Expert",
                "description": "Optimized for medium attention - creates professional marketing content with balanced information",
                "temperature": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.1,
                "token_boost_ids": None,
                "token_boost_factor": 1.0,
                "style_tokens": ["discover", "enhance", "improve", "quality", "professional", "effective", "trusted", "leading"]
            },
            "complex": {
                "name": "Technical Ad Expert",
                "description": "Optimized for high attention - creates sophisticated ad copy with technical details and industry terminology",
                "temperature": 0.7,
                "top_k": 60,
                "repetition_penalty": 1.2,
                "token_boost_ids": self._get_rare_tokens(100),
                "token_boost_factor": 1.8,
                "style_tokens": ["sophisticated", "revolutionary", "proprietary", "enterprise-grade", "seamless", "scalable", "robust", "infrastructure"]
            }
        }
        
        # Expert weights history for visualization
        self.max_history = 100
        self.weight_history = {expert: deque(maxlen=self.max_history) for expert in self.experts}
        self.attention_history = deque(maxlen=self.max_history)
        self.tokens_generated = deque(maxlen=self.max_history)
        self.current_weights = {expert: 0.0 for expert in self.experts}
        self.total_tokens = 0
        
        # Setup visualization
        if self.visualization:
            self._setup_visualization()
    
    def update(self, attention_level, new_tokens=1):
        """
        Update expert weights based on attention level
        
        Args:
            attention_level: Current attention level (0.0 to 1.0)
            new_tokens: Number of new tokens generated in this step
        
        Returns:
            Dict with current weights for each expert
        """
        self.total_tokens += new_tokens
        
        # Calculate expert weights based on attention level
        # Using softmax-like approach to distribute weights
        if attention_level < 0.3:
            # Low attention - strongly favor simple expert
            logits = np.array([8.0, 1.0, 0.2])  # Simple, Balanced, Complex
        elif attention_level > 0.7:
            # High attention - strongly favor complex expert
            logits = np.array([0.2, 1.0, 8.0])  # Simple, Balanced, Complex
        else:
            # Medium attention - favor balanced expert
            mid_point = (attention_level - 0.3) / 0.4  # 0.0 at attn=0.3, 1.0 at attn=0.7
            # Gradually shift from simple to complex as attention increases
            simple_wt = 3.0 - 2.8 * mid_point
            complex_wt = 0.2 + 2.8 * mid_point
            logits = np.array([simple_wt, 5.0, complex_wt])  # Simple, Balanced, Complex
        
        # Convert to probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        weights = exp_logits / exp_logits.sum()
        
        # Update current weights
        experts = list(self.experts.keys())
        for i, expert in enumerate(experts):
            self.current_weights[expert] = weights[i]
        
        # Update history for visualization
        for expert in self.experts:
            self.weight_history[expert].append(self.current_weights[expert])
        self.attention_history.append(attention_level)
        self.tokens_generated.append(self.total_tokens)
        
        # Update visualization if enabled
        if self.visualization and len(self.attention_history) % 5 == 0:
            self._update_visualization()
        
        return self.current_weights
    
    def apply_moe_logit_biasing(self, logits):
        """
        Apply MoE-based biasing to the model's output logits
        
        Args:
            logits: Model output logits tensor of shape [batch_size, vocab_size]
            
        Returns:
            Modified logits tensor
        """
        # Apply expert-specific biasing based on current weights
        experts = list(self.experts.keys())
        modified_logits = logits.clone()
        
        for expert_name in experts:
            expert = self.experts[expert_name]
            weight = self.current_weights[expert_name]
            
            # Only apply if weight is significant
            if weight > 0.2:
                # Token boosting
                if expert["token_boost_ids"] is not None:
                    boost_factor = 1.0 + (expert["token_boost_factor"] - 1.0) * weight
                    modified_logits[:, expert["token_boost_ids"]] *= boost_factor
                
                # Add style tokens (like "simply" for simple expert)
                for style_token in expert["style_tokens"]:
                    # Find token IDs for the style token
                    style_token_ids = self.tokenizer.encode(style_token, add_special_tokens=False)
                    if style_token_ids:
                        # Small boost for style tokens, proportional to weight
                        style_boost = 1.0 + 0.5 * weight
                        modified_logits[:, style_token_ids] *= style_boost
        
        return modified_logits
    
    def get_generation_params(self):
        """
        Get weighted generation parameters based on current expert weights
        
        Returns:
            Dict with generation parameters
        """
        # Initialize with zeros/defaults
        params = {
            "temperature": 0.0,
            "top_k": 0,
            "repetition_penalty": 0.0
        }
        
        # Calculate weighted parameters
        for expert_name, weight in self.current_weights.items():
            expert = self.experts[expert_name]
            params["temperature"] += expert["temperature"] * weight
            params["top_k"] += int(expert["top_k"] * weight)
            params["repetition_penalty"] += expert["repetition_penalty"] * weight
        
        return params
    
    def _get_common_tokens(self, n=100):
        """Get IDs of common tokens"""
        # This would ideally be based on token frequency in a corpus
        # For this demo, we'll use a simple approximation
        # In a real implementation, you would precompute this based on corpus statistics
        try:
            # Get the first n tokens as a proxy for common tokens
            return torch.tensor(list(range(100, 100+n)), 
                               device="cuda" if torch.cuda.is_available() else "cpu")
        except:
            # Fallback if there's an issue
            return torch.tensor(list(range(100, 100+n)))
    
    def _get_rare_tokens(self, n=100):
        """Get IDs of rare/specialized tokens"""
        # This would ideally be based on token frequency in a corpus
        # For this demo, we'll use a simple approximation
        try:
            # Get tokens from the higher range of the vocabulary
            vocab_size = self.tokenizer.vocab_size
            return torch.tensor(list(range(vocab_size-n, vocab_size)), 
                               device="cuda" if torch.cuda.is_available() else "cpu")
        except:
            # Fallback
            return torch.tensor(list(range(8000, 8000+n)))
    
    def _setup_visualization(self):
        """Set up the visualization for expert weights"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Expert weights plot
        self.ax1.set_ylim(0, 1)
        self.ax1.set_title('Real-time Expert Routing Weights')
        self.ax1.set_xlabel('Tokens Generated')
        self.ax1.set_ylabel('Expert Weight')
        
        # Initialize lines for expert weights
        self.expert_lines = {}
        colors = ['blue', 'green', 'red']
        for i, expert in enumerate(self.experts):
            self.expert_lines[expert], = self.ax1.plot(
                [], [], 
                label=self.experts[expert]["name"], 
                color=colors[i]
            )
        self.ax1.legend(loc='upper right')
        
        # Attention level plot
        self.ax2.set_ylim(0, 1)
        self.ax2.set_title('Attention Level')
        self.ax2.set_xlabel('Tokens Generated')
        self.ax2.set_ylabel('Attention Level')
        self.attention_line, = self.ax2.plot([], [], 'k-')
        
        # Add threshold markers
        self.ax2.axhspan(0, 0.3, alpha=0.2, color='blue')
        self.ax2.axhspan(0.3, 0.7, alpha=0.2, color='green')
        self.ax2.axhspan(0.7, 1.0, alpha=0.2, color='red')
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        self.fig.show()
    
    def _update_visualization(self):
        """Update the visualization with current data"""
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            return  # No visualization to update
            
        # Update x-axis limits
        if self.tokens_generated and len(self.tokens_generated) > 0:
            min_x = max(0, min(self.tokens_generated) - 10)
            max_x = max(self.tokens_generated) + 10
            self.ax1.set_xlim(min_x, max_x)
            self.ax2.set_xlim(min_x, max_x)
        
        # Update expert weight lines
        for expert in self.experts:
            if len(self.tokens_generated) > 0 and len(self.weight_history[expert]) > 0:
                self.expert_lines[expert].set_data(self.tokens_generated, self.weight_history[expert])
        
        # Update attention line
        if len(self.tokens_generated) > 0 and len(self.attention_history) > 0:
            self.attention_line.set_data(self.tokens_generated, self.attention_history)
        
        # Try to redraw
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            # Just ignore visualization errors - don't let them crash the main process
            pass
    
    def get_current_expert(self):
        """Get the current dominant expert"""
        if not self.current_weights:
            return "balanced"  # Default to balanced if no weights set
        try:
            return max(self.current_weights.items(), key=lambda x: x[1])[0]
        except ValueError:
            # In case of empty dict or other errors
            return "balanced"
    
    def get_expert_stats(self):
        """Get statistics about expert usage"""
        stats = {}
        for expert in self.experts:
            # Calculate average weight
            if self.weight_history[expert]:
                avg_weight = sum(self.weight_history[expert]) / len(self.weight_history[expert])
                stats[expert] = {
                    "name": self.experts[expert]["name"],
                    "avg_weight": avg_weight,
                    "current_weight": self.current_weights[expert]
                }
        return stats

# For testing
if __name__ == "__main__":
    # Simple mock tokenizer for testing
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 10000
        
        def encode(self, text, add_special_tokens=True):
            # Mock encoding - just return some random IDs
            return [100, 200, 300]
    
    # Create the controller
    tokenizer = MockTokenizer()
    controller = AttentionBasedMoEController(tokenizer)
    
    # Simulate changing attention levels
    try:
        for i in range(100):
            # Oscillating attention pattern for demonstration
            attention = 0.5 + 0.4 * np.sin(i / 10)
            weights = controller.update(attention)
            print(f"Token {i}: Attention={attention:.2f}, Expert weights={weights}")
            
            # Mock logits for testing
            mock_logits = torch.ones((1, 1000))
            biased_logits = controller.apply_moe_logit_biasing(mock_logits)
            
            params = controller.get_generation_params()
            print(f"  Generation params: {params}")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        plt.close('all') 