import math

class MoEController:
    """Controller for Mixture of Experts token generation"""
    
    def __init__(self, expert_config=None, debug_output=False):
        """
        Initialize the Mixture of Experts controller
        
        Args:
            expert_config: Dictionary of expert configurations
            debug_output: Whether to print debug information
        """
        # Default expert configuration if none provided
        self.expert_config = expert_config or {
            "simple": {
                "attention_range": [0.0, 0.3],
                "temperature": 0.7,
                "top_k": 50,
                "repetition_penalty": 1.05
            },
            "balanced": {
                "attention_range": [0.3, 0.7],
                "temperature": 0.5,
                "top_k": 30,
                "repetition_penalty": 1.10
            },
            "complex": {
                "attention_range": [0.7, 1.0],
                "temperature": 0.3,
                "top_k": 20,
                "repetition_penalty": 1.15
            }
        }
        
        # Debug output flag
        self.debug_output = debug_output
        
        # Current state
        self.current_attention = 0.5
        self.current_weights = {}
        self.attention_history = []
        
        # Expert metrics
        self.expert_usage = {expert: 0 for expert in self.expert_config.keys()}
        
        # Initialize with default attention
        self.update_attention(0.5)
    
    def update_attention(self, attention_level, smoothing_factor=0.3):
        """
        Update controller with new attention level
        
        Args:
            attention_level: Current attention level (0.0-1.0)
            smoothing_factor: Factor for smoothing attention changes (0.0-1.0)
        """
        # Apply smoothing for more stable transitions
        self.current_attention = (smoothing_factor * attention_level + 
                                (1 - smoothing_factor) * self.current_attention)
        
        # Keep attention in valid range
        self.current_attention = max(0.0, min(1.0, self.current_attention))
        
        # Store history
        self.attention_history.append(self.current_attention)
        
        # Calculate expert weights based on attention level
        self.current_weights = self._calculate_expert_weights(self.current_attention)
        
        if self.debug_output:
            self._print_expert_weights()
    
    # Legacy method for backward compatibility
    def update(self, attention_level, new_tokens=1):
        """Legacy update method for backward compatibility"""
        self.update_attention(attention_level)
    
    def _calculate_expert_weights(self, attention):
        """
        Calculate weights for each expert based on current attention
        
        Args:
            attention: Current attention level (0.0-1.0)
            
        Returns:
            Dictionary mapping expert names to weights (0.0-1.0)
        """
        weights = {}
        total_weight = 0.0
        
        # Calculate raw weights based on proximity to expert's attention range
        for expert, config in self.expert_config.items():
            att_range = config["attention_range"]
            
            # Fast exit if attention is directly in range
            if att_range[0] <= attention <= att_range[1]:
                weights[expert] = 1.0
                # Update expert usage stats
                self.expert_usage[expert] += 1
                return {expert: 1.0 for expert in self.expert_config.keys() if expert == expert}
            
            # Calculate distance to range
            if attention < att_range[0]:
                distance = att_range[0] - attention
            else:
                distance = attention - att_range[1]
            
            # Convert distance to weight (closer = higher weight)
            # Using exponential falloff for smoother transitions
            weight = math.exp(-5 * distance)
            weights[expert] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for expert in weights:
                weights[expert] /= total_weight
        else:
            # Fallback to balanced expert if all weights are zero
            weights = {expert: 1.0 if expert == "balanced" else 0.0 for expert in self.expert_config.keys()}
        
        return weights
    
    def get_current_expert(self):
        """
        Get the current dominant expert
        
        Returns:
            Name of the expert with highest weight
        """
        if not self.current_weights:
            return "balanced"
            
        # Find expert with highest weight
        max_weight = -1
        dominant_expert = "balanced"  # Default fallback
        
        for expert, weight in self.current_weights.items():
            if weight > max_weight:
                max_weight = weight
                dominant_expert = expert
        
        return dominant_expert
    
    def get_generation_params(self):
        """
        Get generation parameters based on current expert weights
        
        Returns:
            Dictionary of generation parameters (temperature, top_k, etc.)
        """
        # Start with balanced expert as baseline
        params = {
            "temperature": 0.5,
            "top_k": 30,
            "repetition_penalty": 1.1,
        }
        
        # If we have a single dominant expert with weight > 0.8, use its params directly
        dominant_expert = self.get_current_expert()
        max_weight = self.current_weights.get(dominant_expert, 0)
        
        if max_weight > 0.8:
            for param in ["temperature", "top_k", "repetition_penalty"]:
                params[param] = self.expert_config[dominant_expert][param]
            return params
        
        # Otherwise, blend parameters from all experts based on weights
        for param in ["temperature", "top_k", "repetition_penalty"]:
            weighted_sum = 0.0
            for expert, weight in self.current_weights.items():
                weighted_sum += self.expert_config[expert][param] * weight
            params[param] = weighted_sum
        
        # Round top_k to integer
        params["top_k"] = max(1, round(params["top_k"]))
        
        return params
    
    def _print_expert_weights(self):
        """Print current expert weights (for debugging)"""
        weight_str = " | ".join([f"{expert}: {weight:.2f}" for expert, weight in self.current_weights.items()])
        print(f"Attention: {self.current_attention:.2f} → {weight_str}")
        
        # Also print active expert
        active_expert = self.get_current_expert()
        print(f"Active expert: {active_expert}") 