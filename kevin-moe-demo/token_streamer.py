#!/usr/bin/env python3
"""
Token Streamer for EEG-Enhanced LLM Demo
----------------------------------------
This module provides functionality to stream tokens and their 
associated experts from the EEG demo to the backend API.
"""

import requests
import threading
import time
from typing import List, Dict, Any, Optional
import queue
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('token_streamer')

class TokenStreamer:
    """
    Streams tokens and their associated experts to the backend API.
    
    This class provides a thread-safe way to queue tokens for streaming
    to the backend API. It runs in a separate thread to avoid blocking
    the main demo thread.
    """
    
    def __init__(
        self, 
        api_url: str = "http://localhost:8000",
        batch_size: int = 1,
        max_queue_size: int = 1000,
        send_interval: float = 0.1,
        auto_start: bool = True
    ):
        """
        Initialize the token streamer.
        
        Args:
            api_url: URL of the backend API
            batch_size: Number of tokens to send in each batch
            max_queue_size: Maximum size of the token queue
            send_interval: Interval between sending batches (seconds)
            auto_start: Whether to automatically start the streamer thread
        """
        self.api_url = api_url
        self.batch_size = batch_size
        self.send_interval = send_interval
        
        # Thread-safe queue for tokens
        self.token_queue = queue.Queue(maxsize=max_queue_size)
        
        # Threading control
        self._running = False
        self._thread = None
        
        # Stats
        self.tokens_sent = 0
        self.batches_sent = 0
        self.errors = 0
        
        if auto_start:
            self.start()
    
    def start(self) -> bool:
        """Start the streamer thread."""
        if self._running:
            logger.warning("Streamer is already running")
            return False
        
        self._running = True
        self._thread = threading.Thread(
            target=self._streaming_worker,
            daemon=True
        )
        self._thread.start()
        logger.info(f"Token streamer started, sending to {self.api_url}")
        return True
    
    def stop(self) -> bool:
        """Stop the streamer thread."""
        if not self._running:
            logger.warning("Streamer is not running")
            return False
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
        logger.info(f"Token streamer stopped. Sent {self.tokens_sent} tokens in {self.batches_sent} batches")
        return True
    
    def is_running(self) -> bool:
        """Check if the streamer is running."""
        return self._running and (self._thread is not None) and self._thread.is_alive()
    
    def add_token(self, token: str, expert: str) -> bool:
        """
        Add a token to the streaming queue.
        
        Args:
            token: The token text
            expert: The expert that generated the token
            
        Returns:
            True if the token was added, False if the queue is full
        """
        try:
            self.token_queue.put_nowait({
                "token": token,
                "expert": expert
            })
            return True
        except queue.Full:
            logger.warning(f"Token queue is full, dropping token: {token}")
            return False
    
    def add_tokens(self, tokens: List[Dict[str, str]]) -> int:
        """
        Add multiple tokens to the streaming queue.
        
        Args:
            tokens: List of token data (dicts with 'token' and 'expert' keys)
            
        Returns:
            Number of tokens successfully added
        """
        count = 0
        for token_data in tokens:
            if self.add_token(token_data["token"], token_data["expert"]):
                count += 1
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streamer statistics."""
        return {
            "tokens_sent": self.tokens_sent,
            "batches_sent": self.batches_sent,
            "errors": self.errors,
            "queue_size": self.token_queue.qsize(),
            "is_running": self.is_running(),
            "api_url": self.api_url,
            "batch_size": self.batch_size
        }
    
    def clear_queue(self) -> int:
        """
        Clear the token queue.
        
        Returns:
            Number of tokens cleared
        """
        count = 0
        while not self.token_queue.empty():
            try:
                self.token_queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count
    
    def _streaming_worker(self) -> None:
        """
        Worker thread that sends tokens to the backend API.
        This method runs in a separate thread.
        """
        logger.info("Streaming worker started")
        
        while self._running:
            try:
                # Collect a batch of tokens from the queue
                batch = []
                for _ in range(self.batch_size):
                    try:
                        token_data = self.token_queue.get(block=True, timeout=0.1)
                        batch.append(token_data)
                    except queue.Empty:
                        break
                
                # If we collected any tokens, send them
                if batch:
                    self._send_batch(batch)
                else:
                    # No tokens in queue, sleep a bit
                    time.sleep(0.1)
            
            except Exception as e:
                self.errors += 1
                logger.error(f"Error in streaming worker: {e}")
                time.sleep(1.0)  # Back off on error
        
        logger.info("Streaming worker stopped")
    
    def _send_batch(self, batch: List[Dict[str, str]]) -> bool:
        """
        Send a batch of tokens to the backend API.
        
        Args:
            batch: List of token data
            
        Returns:
            True if successful, False otherwise
        """
        if not batch:
            return True
        
        try:
            # Format data according to the TokenStreamData model in the backend
            # Each token needs to be a TokenData object with token and expert fields
            payload = {
                "tokens": [
                    {"token": item["token"], "expert": item["expert"]} 
                    for item in batch
                ]
            }
            
            response = requests.post(
                f"{self.api_url}/add-tokens",
                json=payload,
                timeout=5.0
            )
            
            if response.status_code == 200:
                self.tokens_sent += len(batch)
                self.batches_sent += 1
                logger.debug(f"Sent batch of {len(batch)} tokens")
                return True
            else:
                self.errors += 1
                logger.error(f"Error sending batch: {response.status_code} - {response.text}")
                return False
                
        except requests.RequestException as e:
            self.errors += 1
            logger.error(f"Request failed: {e}")
            return False

# Singleton instance for easy import and use
default_streamer = None

def init_streamer(api_url: str = "http://localhost:8000", **kwargs) -> TokenStreamer:
    """
    Initialize the default token streamer.
    
    Args:
        api_url: URL of the backend API
        **kwargs: Additional arguments for TokenStreamer
        
    Returns:
        The initialized streamer instance
    """
    global default_streamer
    if default_streamer is not None:
        default_streamer.stop()
    
    default_streamer = TokenStreamer(api_url=api_url, **kwargs)
    return default_streamer

def add_token(token: str, expert: str) -> bool:
    """
    Add a token to the default streamer.
    
    Args:
        token: The token text
        expert: The expert that generated the token
        
    Returns:
        True if the token was added, False otherwise
    """
    global default_streamer
    if default_streamer is None:
        init_streamer()
    
    return default_streamer.add_token(token, expert)

def stop_streamer() -> None:
    """Stop the default streamer."""
    global default_streamer
    if default_streamer is not None:
        default_streamer.stop()
        default_streamer = None

# For usage in cleanup functions
import atexit
atexit.register(stop_streamer) 