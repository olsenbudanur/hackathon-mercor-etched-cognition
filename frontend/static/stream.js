// Function to create token element with appropriate styling
function createTokenElement(token, expert) {
    const tokenElement = document.createElement('span');
    
    // Apply different colors based on expert
    let colorClass = 'token-balanced'; // Default class
    
    if (expert === 'simple') {
        colorClass = 'token-simple';
    } else if (expert === 'balanced') {
        colorClass = 'token-balanced';
    } else if (expert === 'complex') {
        colorClass = 'token-complex';
    }
    
    tokenElement.classList.add(colorClass);
    
    // Handle spaces specially to make them visible
    if (token === ' ') {
        tokenElement.innerHTML = '&nbsp;';
        tokenElement.classList.add('token-space');
    } else {
        tokenElement.textContent = token;
    }
    
    return tokenElement;
}

// Function to process incoming tokens
function processToken(token) {
    const tokenContainer = document.getElementById('token-container');
    
    // Create token element
    const tokenElement = createTokenElement(token.text, token.expert);
    
    // Add token to container
    tokenContainer.appendChild(tokenElement);
    
    // Scroll to bottom if container is scrollable
    tokenContainer.scrollTop = tokenContainer.scrollHeight;
}

// Fetch tokens from server
async function fetchTokens() {
    try {
        const response = await fetch('/tokens');
        if (!response.ok) {
            console.error('Failed to fetch tokens:', response.status);
            return;
        }
        
        const tokens = await response.json();
        
        // Process each token
        tokens.forEach(token => {
            processToken(token);
        });
        
    } catch (error) {
        console.error('Error fetching tokens:', error);
    }
    
    // Schedule next fetch
    setTimeout(fetchTokens, 500);
}

// Initialize when document is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Token stream initialized');
    fetchTokens();
}); 