"""
Unit tests for the FastAPI backend application.

These tests cover the API endpoints and functionality of the backend server.
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the FastAPI app
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
from main import app, token_queue, EXPERT_COLORS

# Create a test client
client = TestClient(app)

# Test data
test_tokens = [
    {"token": "Hello", "expert": "simple"},
    {"token": " ", "expert": "balanced"},
    {"token": "world", "expert": "complex"}
]


class TestBackendEndpoints:
    """Test class for backend API endpoints."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear the token queue before each test
        token_queue.clear()

    def test_add_tokens_endpoint(self):
        """Test the /add-tokens endpoint."""
        # Prepare test data
        data = {"tokens": test_tokens}
        
        # Send POST request to add tokens
        response = client.post("/add-tokens", json=data)
        
        # Check response
        assert response.status_code == 200
        assert response.json() == {"status": "success", "tokens_added": 3}
        
        # Check that tokens were added to the queue
        assert len(token_queue) == 3
        assert token_queue[0]["word"] == "Hello"
        assert token_queue[0]["number"] == EXPERT_COLORS["simple"]
        assert token_queue[1]["word"] == " "
        assert token_queue[1]["number"] == EXPERT_COLORS["balanced"]
        assert token_queue[2]["word"] == "world"
        assert token_queue[2]["number"] == EXPERT_COLORS["complex"]

    def test_add_tokens_with_empty_tokens(self):
        """Test the /add-tokens endpoint with empty tokens."""
        # Prepare test data with empty tokens
        data = {"tokens": []}
        
        # Send POST request to add tokens
        response = client.post("/add-tokens", json=data)
        
        # Check response
        assert response.status_code == 200
        assert response.json() == {"status": "success", "tokens_added": 0}
        
        # Check that no tokens were added to the queue
        assert len(token_queue) == 0

    def test_add_tokens_with_invalid_tokens(self):
        """Test the /add-tokens endpoint with invalid tokens."""
        # Prepare test data with invalid tokens (missing required fields)
        data = {"tokens": [{"token": "Hello"}]}  # Missing 'expert' field
        
        # Send POST request to add tokens
        response = client.post("/add-tokens", json=data)
        
        # Check response (should be a validation error)
        assert response.status_code == 422  # Unprocessable Entity

    def test_clear_tokens_endpoint(self):
        """Test the /clear-tokens endpoint."""
        # Add some tokens to the queue
        token_queue.extend([
            {"word": "Hello", "number": 1},
            {"word": "world", "number": 3}
        ])
        
        # Send POST request to clear tokens
        response = client.post("/clear-tokens")
        
        # Check response
        assert response.status_code == 200
        assert response.json() == {"status": "success", "message": "Token queue cleared"}
        
        # Check that the queue is empty
        assert len(token_queue) == 0

    def test_toggle_test_data_endpoint(self):
        """Test the /toggle-test-data endpoint."""
        # Send POST request to enable test data
        response = client.post("/toggle-test-data", json={"enable": True})
        
        # Check response
        assert response.status_code == 200
        assert response.json() == {"status": "success", "test_data_enabled": True}
        
        # Send POST request to disable test data
        response = client.post("/toggle-test-data", json={"enable": False})
        
        # Check response
        assert response.status_code == 200
        assert response.json() == {"status": "success", "test_data_enabled": False}

    def test_get_status_endpoint(self):
        """Test the /status endpoint."""
        # Add some tokens to the queue
        token_queue.extend([
            {"word": "Hello", "number": 1},
            {"word": "world", "number": 3}
        ])
        
        # Send GET request to get status
        response = client.get("/status")
        
        # Check response
        assert response.status_code == 200
        assert response.json()["queue_size"] == 2
        assert "test_data_enabled" in response.json()

    def test_debug_tokens_endpoint(self):
        """Test the /debug-tokens endpoint."""
        # Add some tokens to the queue
        token_queue.extend([
            {"word": "Hello", "number": 1},
            {"word": "world", "number": 3}
        ])
        
        # Send GET request to get debug tokens
        response = client.get("/debug-tokens")
        
        # Check response
        assert response.status_code == 200
        assert response.json()["queue_size"] == 2
        assert len(response.json()["tokens"]) == 2
        assert "test_data_enabled" in response.json()
        
        # Test with limit parameter
        response = client.get("/debug-tokens?limit=1")
        assert response.status_code == 200
        assert len(response.json()["tokens"]) == 1

    @pytest.mark.asyncio
    async def test_stream_endpoint(self):
        """Test the /stream endpoint."""
        # This test is more complex because it involves server-sent events
        # We'll use a mock to test the event_stream function
        
        with patch('main.event_stream') as mock_event_stream:
            # Set up the mock to return a sequence of events
            mock_event_stream.return_value = [
                f"data: {json.dumps({'word': 'Hello', 'number': 1})}\n\n",
                f"data: {json.dumps({'word': 'world', 'number': 3})}\n\n"
            ]
            
            # Send GET request to stream endpoint
            response = client.get("/stream")
            
            # Check response
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            
            # Check that event_stream was called
            mock_event_stream.assert_called_once()


class TestRandomTokenGeneration:
    """Test class for random token generation functionality."""

    def test_generate_random_token(self):
        """Test the generate_random_token function."""
        from main import generate_random_token
        
        # Generate a random token
        token = generate_random_token()
        
        # Check that the token has the expected structure
        assert "word" in token
        assert "number" in token
        assert isinstance(token["word"], str)
        assert isinstance(token["number"], int)
        assert 1 <= token["number"] <= 3  # Expert color should be 1, 2, or 3

    def test_generate_test_tokens(self):
        """Test the generate_test_tokens function."""
        from main import generate_test_tokens
        
        # Generate test tokens
        tokens = generate_test_tokens()
        
        # Check that tokens were generated
        assert len(tokens) >= 1
        
        # Check that each token has the expected structure
        for token in tokens:
            assert hasattr(token, "token")
            assert hasattr(token, "expert")
            assert isinstance(token.token, str)
            assert isinstance(token.expert, str)
            assert token.expert in ["simple", "balanced", "complex"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
