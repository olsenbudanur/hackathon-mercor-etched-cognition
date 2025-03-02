"""
Unit tests for token processing functionality in the backend.

These tests cover the token processing logic, including token validation,
expert color mapping, and token queue management.
"""

import pytest
from collections import deque
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Import the backend modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
from main import EXPERT_COLORS, token_queue, add_tokens, event_stream, TokenData, TokenStreamData

# Test data
test_tokens = [
    TokenData(token="Hello", expert="simple"),
    TokenData(token=" ", expert="balanced"),
    TokenData(token="world", expert="complex")
]


class TestTokenProcessing:
    """Test class for token processing functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear the token queue before each test
        token_queue.clear()

    def test_expert_color_mapping(self):
        """Test the expert to color mapping."""
        # Check that each expert is mapped to the correct color
        assert EXPERT_COLORS["simple"] == 1
        assert EXPERT_COLORS["balanced"] == 2
        assert EXPERT_COLORS["complex"] == 3
        assert EXPERT_COLORS["unknown"] == 2  # Default to balanced/middle color

    def test_token_validation_and_processing(self):
        """Test token validation and processing logic."""
        # Create a TokenStreamData object with test tokens
        token_stream_data = TokenStreamData(tokens=test_tokens)
        
        # Process the tokens
        add_tokens(token_stream_data)
        
        # Check that tokens were processed correctly
        assert len(token_queue) == 3
        
        # Check the first token
        assert token_queue[0]["word"] == "Hello"
        assert token_queue[0]["number"] == EXPERT_COLORS["simple"]
        
        # Check the second token (space)
        assert token_queue[1]["word"] == " "
        assert token_queue[1]["number"] == EXPERT_COLORS["balanced"]
        
        # Check the third token
        assert token_queue[2]["word"] == "world"
        assert token_queue[2]["number"] == EXPERT_COLORS["complex"]

    def test_token_validation_with_empty_token(self):
        """Test token validation with empty token."""
        # Create a TokenStreamData object with an empty token
        token_stream_data = TokenStreamData(tokens=[TokenData(token="", expert="simple")])
        
        # Process the tokens
        add_tokens(token_stream_data)
        
        # Check that the empty token was not added to the queue
        assert len(token_queue) == 0

    def test_token_validation_with_space_token(self):
        """Test token validation with space token."""
        # Create a TokenStreamData object with a space token
        token_stream_data = TokenStreamData(tokens=[TokenData(token=" ", expert="simple")])
        
        # Process the tokens
        add_tokens(token_stream_data)
        
        # Check that the space token was added to the queue
        assert len(token_queue) == 1
        assert token_queue[0]["word"] == " "
        assert token_queue[0]["number"] == EXPERT_COLORS["simple"]

    def test_token_validation_with_long_token(self):
        """Test token validation with a long token."""
        # Create a TokenStreamData object with a long token
        long_token = "a" * 50
        token_stream_data = TokenStreamData(tokens=[TokenData(token=long_token, expert="simple")])
        
        # Process the tokens
        add_tokens(token_stream_data)
        
        # Check that the long token was truncated
        assert len(token_queue) == 1
        assert token_queue[0]["word"] == "a" * 27 + "..."
        assert token_queue[0]["number"] == EXPERT_COLORS["simple"]

    def test_token_validation_with_special_characters(self):
        """Test token validation with special characters."""
        # Create a TokenStreamData object with tokens containing special characters
        special_tokens = [
            TokenData(token="Hello\nWorld", expert="simple"),  # Newline
            TokenData(token="Hello\tWorld", expert="balanced"),  # Tab
            TokenData(token="Hello\\World", expert="complex")  # Backslash
        ]
        token_stream_data = TokenStreamData(tokens=special_tokens)
        
        # Process the tokens
        add_tokens(token_stream_data)
        
        # Check that special characters were handled correctly
        assert len(token_queue) == 3
        assert token_queue[0]["word"] == "Hello World"  # Newline replaced with space
        assert token_queue[1]["word"] == "Hello World"  # Tab replaced with space
        assert token_queue[2]["word"] == "Hello\\\\World"  # Backslash escaped

    def test_token_validation_with_unknown_expert(self):
        """Test token validation with unknown expert."""
        # Create a TokenStreamData object with an unknown expert
        token_stream_data = TokenStreamData(tokens=[TokenData(token="Hello", expert="unknown_expert")])
        
        # Process the tokens
        add_tokens(token_stream_data)
        
        # Check that the token was added with the default expert color
        assert len(token_queue) == 1
        assert token_queue[0]["word"] == "Hello"
        assert token_queue[0]["number"] == EXPERT_COLORS["unknown"]  # Default to unknown color

    def test_token_queue_maxlen(self):
        """Test token queue maximum length."""
        # Add more tokens than the maximum queue length
        max_len = token_queue.maxlen
        
        # Create a large number of tokens
        many_tokens = [TokenData(token=f"Token{i}", expert="simple") for i in range(max_len + 10)]
        token_stream_data = TokenStreamData(tokens=many_tokens)
        
        # Process the tokens
        add_tokens(token_stream_data)
        
        # Check that the queue length is limited to maxlen
        assert len(token_queue) == max_len


class TestEventStream:
    """Test class for event stream functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear the token queue before each test
        token_queue.clear()

    @pytest.mark.asyncio
    async def test_event_stream_with_tokens(self):
        """Test the event_stream function with tokens in the queue."""
        # Add tokens to the queue
        token_queue.extend([
            {"word": "Hello", "number": 1},
            {"word": "world", "number": 3}
        ])
        
        # Create a mock for asyncio.sleep to avoid actual delays
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Get the event stream generator
            generator = event_stream()
            
            # Get the first event
            event = await generator.__anext__()
            
            # Check that the event has the expected format
            assert event.startswith("data: ")
            assert "Hello" in event
            assert "1" in event
            
            # Check that sleep was called
            mock_sleep.assert_called_once()
            
            # Reset the mock
            mock_sleep.reset_mock()
            
            # Get the second event
            event = await generator.__anext__()
            
            # Check that the event has the expected format
            assert event.startswith("data: ")
            assert "world" in event
            assert "3" in event
            
            # Check that sleep was called
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_stream_with_empty_queue(self):
        """Test the event_stream function with an empty queue."""
        # Create a mock for asyncio.sleep to avoid actual delays
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Get the event stream generator
            generator = event_stream()
            
            # Set enable_test_data to False to avoid generating random tokens
            import main
            main.enable_test_data = False
            
            # Try to get an event (should sleep and not yield anything)
            # We'll use a timeout to avoid hanging
            try:
                await asyncio.wait_for(generator.__anext__(), timeout=0.1)
                assert False, "Should have timed out"
            except asyncio.TimeoutError:
                pass
            
            # Check that sleep was called
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_stream_with_test_data(self):
        """Test the event_stream function with test data enabled."""
        # Create a mock for asyncio.sleep to avoid actual delays
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Create a mock for generate_random_token
            with patch('main.generate_random_token') as mock_generate_random_token:
                # Set up the mock to return a test token
                mock_generate_random_token.return_value = {"word": "TestToken", "number": 2}
                
                # Get the event stream generator
                generator = event_stream()
                
                # Set enable_test_data to True to generate random tokens
                import main
                main.enable_test_data = True
                
                # Get the first event
                event = await generator.__anext__()
                
                # Check that the event has the expected format
                assert event.startswith("data: ")
                assert "TestToken" in event
                assert "2" in event
                
                # Check that generate_random_token was called
                mock_generate_random_token.assert_called_once()
                
                # Check that sleep was called
                mock_sleep.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
