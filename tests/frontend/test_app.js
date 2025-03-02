/**
 * Unit tests for the React frontend application.
 * 
 * These tests cover the rendering and functionality of the React components.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from '../../frontend/src/App';
import { StreamingApp } from '../../frontend/src/App';
import { getColor } from '../../frontend/src/utils/colorMapping';

// Mock the EventSource API
class MockEventSource {
  constructor(url) {
    this.url = url;
    this.onmessage = null;
    this.onerror = null;
    this.onopen = null;
    this.readyState = 0; // 0 = CONNECTING, 1 = OPEN, 2 = CLOSED
    
    // Simulate connection
    setTimeout(() => {
      this.readyState = 1;
      if (this.onopen) this.onopen();
    }, 100);
  }
  
  // Method to simulate receiving a message
  simulateMessage(data) {
    if (this.onmessage) {
      const event = { data: JSON.stringify(data) };
      this.onmessage(event);
    }
  }
  
  // Method to simulate an error
  simulateError() {
    if (this.onerror) {
      this.onerror(new Error('EventSource error'));
    }
  }
  
  // Method to close the connection
  close() {
    this.readyState = 2;
  }
}

// Mock the fetch API
global.fetch = jest.fn();

// Mock the EventSource API
global.EventSource = MockEventSource;

describe('App Component', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    
    // Mock successful fetch response
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'success' })
    });
  });
  
  test('renders without crashing', () => {
    render(<App />);
    expect(screen.getByText(/EEG-Enhanced Language Model/i)).toBeInTheDocument();
  });
  
  test('renders the StreamingApp component', () => {
    render(<App />);
    expect(screen.getByTestId('streaming-app')).toBeInTheDocument();
  });
  
  test('renders the header with the correct title', () => {
    render(<App />);
    expect(screen.getByRole('heading', { name: /EEG-Enhanced Language Model/i })).toBeInTheDocument();
  });
  
  test('renders the footer with the correct text', () => {
    render(<App />);
    expect(screen.getByText(/© 2023 EEG-Enhanced Language Model/i)).toBeInTheDocument();
  });
});

describe('StreamingApp Component', () => {
  let mockEventSource;
  
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    
    // Mock successful fetch response
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'success' })
    });
    
    // Render the component
    render(<StreamingApp />);
    
    // Get the mock EventSource instance
    mockEventSource = new MockEventSource('http://localhost:8000/stream');
  });
  
  test('renders the output container', () => {
    expect(screen.getByTestId('output-container')).toBeInTheDocument();
  });
  
  test('renders the control buttons', () => {
    expect(screen.getByText(/Clear/i)).toBeInTheDocument();
    expect(screen.getByText(/Test Data/i)).toBeInTheDocument();
  });
  
  test('clicking Clear button calls clearTokens', async () => {
    // Click the Clear button
    fireEvent.click(screen.getByText(/Clear/i));
    
    // Check that fetch was called with the correct URL and method
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/clear-tokens',
      { method: 'POST' }
    );
    
    // Wait for the success message
    await waitFor(() => {
      expect(screen.getByText(/Tokens cleared/i)).toBeInTheDocument();
    });
  });
  
  test('clicking Test Data button calls toggleTestData', async () => {
    // Click the Test Data button
    fireEvent.click(screen.getByText(/Test Data/i));
    
    // Check that fetch was called with the correct URL, method, and body
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/toggle-test-data',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable: true })
      }
    );
    
    // Wait for the success message
    await waitFor(() => {
      expect(screen.getByText(/Test data enabled/i)).toBeInTheDocument();
    });
    
    // Click the Test Data button again to disable
    fireEvent.click(screen.getByText(/Test Data/i));
    
    // Check that fetch was called with the correct URL, method, and body
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/toggle-test-data',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable: false })
      }
    );
    
    // Wait for the success message
    await waitFor(() => {
      expect(screen.getByText(/Test data disabled/i)).toBeInTheDocument();
    });
  });
  
  test('displays tokens received from the EventSource', async () => {
    // Simulate receiving a message
    mockEventSource.simulateMessage({ word: 'Hello', number: 1 });
    
    // Check that the token was added to the output container
    await waitFor(() => {
      const tokenElement = screen.getByText('Hello');
      expect(tokenElement).toBeInTheDocument();
      expect(tokenElement).toHaveClass('expert-1');
    });
    
    // Simulate receiving another message
    mockEventSource.simulateMessage({ word: ' world', number: 3 });
    
    // Check that the token was added to the output container
    await waitFor(() => {
      const tokenElement = screen.getByText(' world');
      expect(tokenElement).toBeInTheDocument();
      expect(tokenElement).toHaveClass('expert-3');
    });
  });
  
  test('handles EventSource errors', async () => {
    // Simulate an error
    mockEventSource.simulateError();
    
    // Check that the error message was displayed
    await waitFor(() => {
      expect(screen.getByText(/Error connecting to the server/i)).toBeInTheDocument();
    });
  });
  
  test('cleans up EventSource on unmount', () => {
    // Create a spy on the EventSource.close method
    const closeSpy = jest.spyOn(MockEventSource.prototype, 'close');
    
    // Unmount the component
    const { unmount } = render(<StreamingApp />);
    unmount();
    
    // Check that close was called
    expect(closeSpy).toHaveBeenCalled();
  });
});

describe('Token Display', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    
    // Mock successful fetch response
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'success' })
    });
    
    // Render the component
    render(<StreamingApp />);
  });
  
  test('displays tokens with the correct color class', async () => {
    // Get the mock EventSource instance
    const mockEventSource = new MockEventSource('http://localhost:8000/stream');
    
    // Simulate receiving messages with different expert numbers
    mockEventSource.simulateMessage({ word: 'Simple', number: 1 });
    mockEventSource.simulateMessage({ word: ' balanced', number: 2 });
    mockEventSource.simulateMessage({ word: ' complex', number: 3 });
    
    // Check that the tokens were added with the correct color classes
    await waitFor(() => {
      const simpleToken = screen.getByText('Simple');
      expect(simpleToken).toHaveClass('expert-1');
      
      const balancedToken = screen.getByText(' balanced');
      expect(balancedToken).toHaveClass('expert-2');
      
      const complexToken = screen.getByText(' complex');
      expect(complexToken).toHaveClass('expert-3');
    });
  });
  
  test('handles tokens with special characters', async () => {
    // Get the mock EventSource instance
    const mockEventSource = new MockEventSource('http://localhost:8000/stream');
    
    // Simulate receiving a message with special characters
    mockEventSource.simulateMessage({ word: 'Hello\nWorld', number: 1 });
    
    // Check that the token was added and displayed correctly
    await waitFor(() => {
      const tokenElement = screen.getByText('Hello\nWorld');
      expect(tokenElement).toBeInTheDocument();
    });
  });
  
  test('handles empty tokens', async () => {
    // Get the mock EventSource instance
    const mockEventSource = new MockEventSource('http://localhost:8000/stream');
    
    // Simulate receiving an empty message
    mockEventSource.simulateMessage({ word: '', number: 1 });
    
    // Check that the token was not added (or was added as an empty string)
    // This is implementation-dependent, so we just check that the component didn't crash
    await waitFor(() => {
      expect(screen.getByTestId('output-container')).toBeInTheDocument();
    });
  });
  
  test('handles a large number of tokens', async () => {
    // Get the mock EventSource instance
    const mockEventSource = new MockEventSource('http://localhost:8000/stream');
    
    // Simulate receiving many messages
    for (let i = 0; i < 100; i++) {
      mockEventSource.simulateMessage({ word: `Token${i}`, number: (i % 3) + 1 });
    }
    
    // Check that the component didn't crash
    await waitFor(() => {
      expect(screen.getByTestId('output-container')).toBeInTheDocument();
    });
    
    // Check that at least some of the tokens were added
    await waitFor(() => {
      expect(screen.getByText('Token0')).toBeInTheDocument();
      expect(screen.getByText('Token99')).toBeInTheDocument();
    });
  });
});

describe('Error Handling', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
  });
  
  test('handles fetch errors when clearing tokens', async () => {
    // Mock a failed fetch response
    global.fetch.mockRejectedValue(new Error('Network error'));
    
    // Render the component
    render(<StreamingApp />);
    
    // Click the Clear button
    fireEvent.click(screen.getByText(/Clear/i));
    
    // Check that the error message was displayed
    await waitFor(() => {
      expect(screen.getByText(/Error clearing tokens/i)).toBeInTheDocument();
    });
  });
  
  test('handles fetch errors when toggling test data', async () => {
    // Mock a failed fetch response
    global.fetch.mockRejectedValue(new Error('Network error'));
    
    // Render the component
    render(<StreamingApp />);
    
    // Click the Test Data button
    fireEvent.click(screen.getByText(/Test Data/i));
    
    // Check that the error message was displayed
    await waitFor(() => {
      expect(screen.getByText(/Error toggling test data/i)).toBeInTheDocument();
    });
  });
  
  test('handles non-OK fetch responses', async () => {
    // Mock a non-OK fetch response
    global.fetch.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error'
    });
    
    // Render the component
    render(<StreamingApp />);
    
    // Click the Clear button
    fireEvent.click(screen.getByText(/Clear/i));
    
    // Check that the error message was displayed
    await waitFor(() => {
      expect(screen.getByText(/Error: 500 Internal Server Error/i)).toBeInTheDocument();
    });
  });
});

describe('Accessibility', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    
    // Mock successful fetch response
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'success' })
    });
  });
  
  test('buttons have accessible names', () => {
    render(<StreamingApp />);
    
    // Check that the buttons have accessible names
    expect(screen.getByRole('button', { name: /Clear/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Test Data/i })).toBeInTheDocument();
  });
  
  test('output container has appropriate ARIA attributes', () => {
    render(<StreamingApp />);
    
    // Check that the output container has appropriate ARIA attributes
    const outputContainer = screen.getByTestId('output-container');
    expect(outputContainer).toHaveAttribute('aria-live', 'polite');
    expect(outputContainer).toHaveAttribute('aria-atomic', 'false');
  });
});
