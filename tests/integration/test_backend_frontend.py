"""
Integration tests for the interaction between the backend and frontend components.

These tests verify that the backend and frontend components work together correctly,
focusing on token streaming, event handling, and data flow between the components.
"""

import pytest
import requests
import json
import time
import asyncio
import threading
from unittest.mock import patch, MagicMock
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))

# Import the backend app
from main import app, token_queue, add_tokens, TokenData, TokenStreamData

# Import the test client
from fastapi.testclient import TestClient

# Create a test client
client = TestClient(app)

# Test data
test_tokens = [
    {"token": "Hello", "expert": "simple"},
    {"token": " ", "expert": "balanced"},
    {"token": "world", "expert": "complex"}
]


class TestBackendFrontendIntegration:
    """Test class for backend-frontend integration."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear the token queue before each test
        token_queue.clear()

    def test_token_streaming_flow(self):
        """Test the complete token streaming flow from backend to frontend."""
        # 1. Add tokens to the backend
        response = client.post("/add-tokens", json={"tokens": test_tokens})
        assert response.status_code == 200
        assert response.json()["tokens_added"] == 3

        # 2. Start a streaming connection
        # We'll use a separate thread to simulate the frontend's EventSource connection
        received_tokens = []
        stop_event = threading.Event()

        def stream_tokens():
            """Simulate the frontend's EventSource connection."""
            with client.stream("GET", "/stream") as response:
                # Check that the response has the correct headers
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"

                # Read events from the stream
                for line in response.iter_lines():
                    if stop_event.is_set():
                        break

                    if line.startswith(b"data: "):
                        # Parse the event data
                        data = json.loads(line[6:].decode("utf-8"))
                        received_tokens.append(data)

                        # If we've received all the tokens, stop
                        if len(received_tokens) >= 3:
                            stop_event.set()
                            break

        # Start the streaming thread
        stream_thread = threading.Thread(target=stream_tokens)
        stream_thread.start()

        # Wait for the thread to receive all tokens or timeout
        stop_event.wait(timeout=5)
        stream_thread.join(timeout=1)

        # Check that we received all the tokens
        assert len(received_tokens) == 3

        # Check that the tokens were received in the correct order
        assert received_tokens[0]["word"] == "Hello"
        assert received_tokens[0]["number"] == 1  # simple expert
        assert received_tokens[1]["word"] == " "
        assert received_tokens[1]["number"] == 2  # balanced expert
        assert received_tokens[2]["word"] == "world"
        assert received_tokens[2]["number"] == 3  # complex expert

    def test_clear_tokens_integration(self):
        """Test clearing tokens and its effect on the stream."""
        # 1. Add tokens to the backend
        response = client.post("/add-tokens", json={"tokens": test_tokens})
        assert response.status_code == 200
        assert response.json()["tokens_added"] == 3

        # 2. Clear the tokens
        response = client.post("/clear-tokens")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # 3. Check that the token queue is empty
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["queue_size"] == 0

        # 4. Start a streaming connection
        # We'll use a separate thread to simulate the frontend's EventSource connection
        received_tokens = []
        stop_event = threading.Event()

        def stream_tokens():
            """Simulate the frontend's EventSource connection."""
            with client.stream("GET", "/stream") as response:
                # Check that the response has the correct headers
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"

                # Read events from the stream for a short time
                start_time = time.time()
                for line in response.iter_lines():
                    if stop_event.is_set() or time.time() - start_time > 1:
                        break

                    if line.startswith(b"data: "):
                        # Parse the event data
                        data = json.loads(line[6:].decode("utf-8"))
                        received_tokens.append(data)

        # Start the streaming thread
        stream_thread = threading.Thread(target=stream_tokens)
        stream_thread.start()

        # Wait for a short time
        time.sleep(1)
        stop_event.set()
        stream_thread.join(timeout=1)

        # Check that we didn't receive any tokens
        assert len(received_tokens) == 0

    def test_test_data_integration(self):
        """Test enabling test data and its effect on the stream."""
        # 1. Enable test data
        response = client.post("/toggle-test-data", json={"enable": True})
        assert response.status_code == 200
        assert response.json()["test_data_enabled"] == True

        # 2. Start a streaming connection
        # We'll use a separate thread to simulate the frontend's EventSource connection
        received_tokens = []
        stop_event = threading.Event()

        def stream_tokens():
            """Simulate the frontend's EventSource connection."""
            with client.stream("GET", "/stream") as response:
                # Check that the response has the correct headers
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"

                # Read events from the stream
                for line in response.iter_lines():
                    if stop_event.is_set():
                        break

                    if line.startswith(b"data: "):
                        # Parse the event data
                        data = json.loads(line[6:].decode("utf-8"))
                        received_tokens.append(data)

                        # If we've received some tokens, stop
                        if len(received_tokens) >= 5:
                            stop_event.set()
                            break

        # Start the streaming thread
        stream_thread = threading.Thread(target=stream_tokens)
        stream_thread.start()

        # Wait for the thread to receive some tokens or timeout
        stop_event.wait(timeout=5)
        stream_thread.join(timeout=1)

        # Check that we received some tokens
        assert len(received_tokens) > 0

        # 3. Disable test data
        response = client.post("/toggle-test-data", json={"enable": False})
        assert response.status_code == 200
        assert response.json()["test_data_enabled"] == False

    def test_debug_tokens_integration(self):
        """Test the debug tokens endpoint with tokens in the queue."""
        # 1. Add tokens to the backend
        response = client.post("/add-tokens", json={"tokens": test_tokens})
        assert response.status_code == 200
        assert response.json()["tokens_added"] == 3

        # 2. Get debug tokens
        response = client.get("/debug-tokens")
        assert response.status_code == 200
        assert response.json()["queue_size"] == 3
        assert len(response.json()["tokens"]) == 3

        # Check that the tokens are correct
        tokens = response.json()["tokens"]
        assert tokens[0]["word"] == "Hello"
        assert tokens[0]["number"] == 1  # simple expert
        assert tokens[1]["word"] == " "
        assert tokens[1]["number"] == 2  # balanced expert
        assert tokens[2]["word"] == "world"
        assert tokens[2]["number"] == 3  # complex expert

    def test_frontend_rendering_simulation(self):
        """Simulate the frontend rendering of tokens from the backend."""
        # This test simulates how the frontend would render tokens
        # received from the backend.

        # 1. Add tokens to the backend
        response = client.post("/add-tokens", json={"tokens": test_tokens})
        assert response.status_code == 200
        assert response.json()["tokens_added"] == 3

        # 2. Get the tokens from the debug endpoint
        response = client.get("/debug-tokens")
        assert response.status_code == 200
        tokens = response.json()["tokens"]

        # 3. Simulate the frontend rendering
        # In a real frontend, this would be done with React components
        rendered_html = ""
        for token in tokens:
            # Map the expert number to a CSS class
            expert_class = f"expert-{token['number']}"
            # Create a span element with the token text and expert class
            rendered_html += f'<span class="{expert_class}">{token["word"]}</span>'

        # Check that the rendered HTML contains all the tokens with the correct classes
        assert '<span class="expert-1">Hello</span>' in rendered_html
        assert '<span class="expert-2"> </span>' in rendered_html
        assert '<span class="expert-3">world</span>' in rendered_html

        # The complete rendered text should be "Hello world"
        rendered_text = "".join(token["word"] for token in tokens)
        assert rendered_text == "Hello world"


class TestBackendFrontendErrorHandling:
    """Test class for backend-frontend error handling."""

    def test_invalid_token_format(self):
        """Test handling of invalid token format."""
        # Try to add tokens with invalid format
        invalid_tokens = [
            {"token": "Hello"},  # Missing expert field
            {"expert": "simple"}  # Missing token field
        ]
        response = client.post("/add-tokens", json={"tokens": invalid_tokens})
        assert response.status_code == 422  # Unprocessable Entity

    def test_invalid_expert_value(self):
        """Test handling of invalid expert value."""
        # Try to add tokens with invalid expert value
        invalid_tokens = [
            {"token": "Hello", "expert": "invalid_expert"}
        ]
        response = client.post("/add-tokens", json={"tokens": invalid_tokens})
        assert response.status_code == 200  # Should still accept it but use default expert

        # Check that the token was added with the default expert
        response = client.get("/debug-tokens")
        assert response.status_code == 200
        tokens = response.json()["tokens"]
        assert tokens[0]["word"] == "Hello"
        assert tokens[0]["number"] == 2  # Default to balanced/middle color

    def test_empty_token_queue(self):
        """Test streaming with an empty token queue."""
        # Clear the token queue
        response = client.post("/clear-tokens")
        assert response.status_code == 200

        # Start a streaming connection
        # We'll use a separate thread to simulate the frontend's EventSource connection
        received_tokens = []
        stop_event = threading.Event()

        def stream_tokens():
            """Simulate the frontend's EventSource connection."""
            with client.stream("GET", "/stream") as response:
                # Check that the response has the correct headers
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"

                # Read events from the stream for a short time
                start_time = time.time()
                for line in response.iter_lines():
                    if stop_event.is_set() or time.time() - start_time > 1:
                        break

                    if line.startswith(b"data: "):
                        # Parse the event data
                        data = json.loads(line[6:].decode("utf-8"))
                        received_tokens.append(data)

        # Start the streaming thread
        stream_thread = threading.Thread(target=stream_tokens)
        stream_thread.start()

        # Wait for a short time
        time.sleep(1)
        stop_event.set()
        stream_thread.join(timeout=1)

        # Check that we didn't receive any tokens
        assert len(received_tokens) == 0

    def test_concurrent_connections(self):
        """Test multiple concurrent streaming connections."""
        # Add tokens to the backend
        response = client.post("/add-tokens", json={"tokens": test_tokens})
        assert response.status_code == 200
        assert response.json()["tokens_added"] == 3

        # Start multiple streaming connections
        # We'll use separate threads to simulate multiple frontend clients
        received_tokens_1 = []
        received_tokens_2 = []
        stop_event_1 = threading.Event()
        stop_event_2 = threading.Event()

        def stream_tokens_1():
            """Simulate the first frontend client."""
            with client.stream("GET", "/stream") as response:
                for line in response.iter_lines():
                    if stop_event_1.is_set():
                        break

                    if line.startswith(b"data: "):
                        data = json.loads(line[6:].decode("utf-8"))
                        received_tokens_1.append(data)

                        if len(received_tokens_1) >= 3:
                            stop_event_1.set()
                            break

        def stream_tokens_2():
            """Simulate the second frontend client."""
            with client.stream("GET", "/stream") as response:
                for line in response.iter_lines():
                    if stop_event_2.is_set():
                        break

                    if line.startswith(b"data: "):
                        data = json.loads(line[6:].decode("utf-8"))
                        received_tokens_2.append(data)

                        if len(received_tokens_2) >= 3:
                            stop_event_2.set()
                            break

        # Start the streaming threads
        stream_thread_1 = threading.Thread(target=stream_tokens_1)
        stream_thread_2 = threading.Thread(target=stream_tokens_2)
        stream_thread_1.start()
        stream_thread_2.start()

        # Wait for the threads to receive all tokens or timeout
        stop_event_1.wait(timeout=5)
        stop_event_2.wait(timeout=5)
        stream_thread_1.join(timeout=1)
        stream_thread_2.join(timeout=1)

        # Check that both clients received all the tokens
        assert len(received_tokens_1) == 3
        assert len(received_tokens_2) == 3

        # Check that the tokens were received in the correct order by both clients
        for tokens in [received_tokens_1, received_tokens_2]:
            assert tokens[0]["word"] == "Hello"
            assert tokens[0]["number"] == 1  # simple expert
            assert tokens[1]["word"] == " "
            assert tokens[1]["number"] == 2  # balanced expert
            assert tokens[2]["word"] == "world"
            assert tokens[2]["number"] == 3  # complex expert


if __name__ == "__main__":
    pytest.main(["-v", __file__])
