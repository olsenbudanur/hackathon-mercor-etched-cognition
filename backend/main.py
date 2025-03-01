from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import Body, HTTPException
import time
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
import random
import string
import json
import asyncio
from collections import deque
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from the frontend
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow GET and POST methods
    allow_headers=["*"],  # Allow all headers
)

# Queue to store tokens and their associated experts
token_queue = deque(maxlen=1000)  # Limit queue size to prevent memory issues

# Configuration flags
enable_test_data = False  # Flag to control random word generation

# Expert to color mapping
EXPERT_COLORS = {
    "simple": 1,     # Maps to color 1 in frontend (e.g., blue)
    "balanced": 2,   # Maps to color 2 in frontend (e.g., green)
    "complex": 3,    # Maps to color 3 in frontend (e.g., red)
    "unknown": 2     # Default to balanced/middle color if expert is unknown
}

class TokenData(BaseModel):
    token: str
    expert: str

class TokenStreamData(BaseModel):
    tokens: List[TokenData]

@app.post("/add-tokens")
async def add_tokens(data: TokenStreamData):
    """
    Endpoint to receive token data from the EEG demo
    and add it to the streaming queue
    """
    for token_data in data.tokens:
        # Add each token with its expert to the queue
        token = token_data.token
        expert = token_data.expert
        
        # Validate and process token text
        # Ensure token is a non-empty string
        if not token and token != " ":  # Special check for space tokens
            continue
        
        # Ensure token is properly escaped for display
        token = token.replace('\\', '\\\\').replace('\n', ' ').replace('\t', ' ')
        
        # Limit token length for display (frontend might have size constraints)
        if len(token) > 30:
            token = token[:27] + '...'
        
        # Map expert to a color number for the frontend
        color_num = EXPERT_COLORS.get(expert.lower(), 2)
        
        token_queue.append({
            "word": token,
            "number": color_num
        })
    
    return {"status": "success", "tokens_added": len(data.tokens)}

# Fallback function for demo/testing - generates random words
def generate_random_token():
    """
    Generate a random token with an associated expert.
    Creates more realistic token samples than just random letters.
    """
    # Possible token types
    token_types = [
        "word",          # Normal word
        "punctuation",   # Punctuation mark
        "space",         # Space or whitespace
        "abbreviation",  # Abbreviation or acronym
        "number",        # Numeric token
    ]
    
    # Weight toward normal words
    token_type = random.choices(
        token_types, 
        weights=[0.7, 0.15, 0.05, 0.05, 0.05], 
        k=1
    )[0]
    
    if token_type == "word":
        # Common English words
        common_words = [
            "the", "and", "that", "have", "for", "not", "with", "you", 
            "this", "but", "from", "they", "say", "she", "will", "one", 
            "all", "would", "there", "their", "what", "out", "about", 
            "who", "get", "which", "when", "make", "can", "like", "time", 
            "just", "know", "people", "year", "take", "them", "some", 
            "into", "two", "see", "more", "look", "only", "come", "its", 
            "over", "think", "also", "back", "use", "after", "work", 
            "first", "well", "way", "even", "new", "want", "because", 
            "any", "these", "give", "day", "most", "us"
        ]
        random_word = random.choice(common_words)
        
        # Occasionally make it uppercase or capitalize
        if random.random() < 0.05:  # 5% chance
            random_word = random_word.upper()
        elif random.random() < 0.2:  # 20% chance
            random_word = random_word.capitalize()
            
    elif token_type == "punctuation":
        # Common punctuation
        random_word = random.choice([",", ".", "!", "?", ";", ":", "-", ")", "(", "\"", "'"])
        
    elif token_type == "space":
        # Space or newline
        random_word = " "
        
    elif token_type == "abbreviation":
        # Common abbreviations or acronyms
        random_word = random.choice(["AI", "ML", "Dr.", "Mr.", "PhD", "USA", "UK", "CEO", "etc.", "e.g."])
        
    else:  # number
        # Numeric token
        if random.random() < 0.7:  # 70% chance of a single digit
            random_word = str(random.randint(0, 9))
        else:  # 30% chance of a multi-digit number
            random_word = str(random.randint(10, 999))
    
    # Assign a realistic expert based on the token type
    if token_type == "word" and len(random_word) >= 7:
        # Longer words tend to use the complex expert
        expert_weights = {"simple": 0.1, "balanced": 0.3, "complex": 0.6}
    elif token_type in ["punctuation", "space"]:
        # Punctuation and spaces tend to use the simple expert
        expert_weights = {"simple": 0.7, "balanced": 0.25, "complex": 0.05}
    elif token_type == "abbreviation":
        # Abbreviations tend to use the complex expert
        expert_weights = {"simple": 0.2, "balanced": 0.3, "complex": 0.5}
    else:
        # Most tokens tend to use the balanced expert
        expert_weights = {"simple": 0.3, "balanced": 0.5, "complex": 0.2}
    
    # Select expert based on weights
    experts = list(expert_weights.keys())
    weights = list(expert_weights.values())
    random_expert = random.choices(experts, weights=weights, k=1)[0]
    
    return {
        "word": random_word,
        "number": EXPERT_COLORS[random_expert]
    }

# Streaming function to yield tokens from the queue
async def event_stream():
    """
    Stream tokens to the frontend in a controlled manner.
    Handles both real tokens from queue and random test tokens when enabled.
    """
    while True:
        if token_queue:
            # If there are tokens in the queue, send one
            data = token_queue.popleft()
            
            # Debug info
            word = data.get("word", "[empty]")
            number = data.get("number", 0)
            print(f"Streaming token: '{word}' (Expert color: {number})")
            
            # Send as server-sent event
            yield f"data: {json.dumps(data)}\n\n"
            
            # Realistic delay between tokens (varies slightly)
            # This simulates the natural rhythm of text generation
            delay = 0.1 + random.random() * 0.1  # 0.1-0.2 seconds
            await asyncio.sleep(delay)  
        else:
            # If queue is empty and test data is enabled, generate random tokens
            # Otherwise, just wait for new tokens
            if enable_test_data:
                data = generate_random_token()
                # Debug info
                word = data.get("word", "[empty]")
                number = data.get("number", 0)
                print(f"Streaming test token: '{word}' (Expert color: {number})")
                
                # Send as server-sent event
                yield f"data: {json.dumps(data)}\n\n"
                
                # Longer delay for test mode (feels more natural)
                await asyncio.sleep(1.0 + random.random() * 1.0)  # 1-2 seconds
            else:
                # Just wait for new tokens without sending anything
                await asyncio.sleep(0.2)

@app.get("/stream")
async def stream():
    """Stream tokens to the frontend"""
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Endpoint to clear the token queue (useful for testing)
@app.post("/clear-tokens")
async def clear_tokens():
    token_queue.clear()
    return {"status": "success", "message": "Token queue cleared"}

# Endpoint to toggle test data generation
@app.post("/toggle-test-data")
async def toggle_test_data(enable: bool = Body(..., embed=True)):
    """Enable or disable random test data generation"""
    global enable_test_data
    enable_test_data = enable
    return {"status": "success", "test_data_enabled": enable_test_data}

# Get current status
@app.get("/status")
async def get_status():
    """Get the current status of the server"""
    return {
        "queue_size": len(token_queue),
        "test_data_enabled": enable_test_data
    }

# Get recent tokens for debugging
@app.get("/debug-tokens")
async def debug_tokens(limit: int = 50):
    """
    Get a dump of recently processed tokens for debugging
    
    Args:
        limit: Maximum number of tokens to return (default 50)
    """
    # Create a copy of the token queue for inspection
    token_list = list(token_queue)
    
    return {
        "queue_size": len(token_queue),
        "tokens": token_list[:limit],
        "test_data_enabled": enable_test_data
    }

# Set debug mode on startup (but not test data by default)
@app.on_event("startup")
async def startup_event():
    app.debug = True  # Set to False in production
    global enable_test_data
    enable_test_data = False

def generate_test_tokens():
    """Generate random test tokens for development"""
    words = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
             "Hello", "world", "This", "is", "a", "test", "of", "token", "streaming",
             "with", "different", "experts", "handling", "various", "parts", 
             "of", "the", "sentence", ".", "!", "?", ",", "and", "more", " "]
             
    experts = ["simple", "balanced", "complex"]
    
    # Generate 1-3 tokens
    num_tokens = random.randint(1, 3)
    tokens = []
    
    for _ in range(num_tokens):
        # Ensure we have some spaces in the stream
        if random.random() < 0.2:
            text = " "  # Use a space token 20% of the time
        else:
            text = random.choice(words)
            
        expert = random.choice(experts)
        tokens.append(Token(text=text, expert=expert))
        
    return tokens
