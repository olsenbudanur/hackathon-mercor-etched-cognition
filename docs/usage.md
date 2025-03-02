# Usage Guide

This guide provides detailed instructions for using the EEG-Enhanced Language Model with Mixture-of-Experts project. It covers how to start the backend and frontend components, run the EEG processing script, and use the Kevin-MOE demo.

## Table of Contents

- [Starting the Backend Server](#starting-the-backend-server)
- [Starting the Frontend Application](#starting-the-frontend-application)
- [Running the EEG Processing Script](#running-the-eeg-processing-script)
- [Using the Kevin-MOE Demo](#using-the-kevin-moe-demo)
- [Advanced Usage](#advanced-usage)
- [API Endpoints](#api-endpoints)

## Starting the Backend Server

The backend server is built with FastAPI and handles token streaming and expert routing.

1. **Navigate to the backend directory**:
   ```bash
   cd hackathon-mercor-etched-cognition/backend
   ```

2. **Start the server**:
   ```bash
   uvicorn main:app --reload
   ```

   This will start the server on `http://localhost:8000`. The `--reload` flag enables auto-reloading when code changes are detected, which is useful during development.

3. **Verify the server is running**:
   - Open a web browser and navigate to `http://localhost:8000/docs`
   - You should see the FastAPI Swagger UI with all available endpoints

## Starting the Frontend Application

The frontend application is built with React and provides a visualization interface for the EEG-enhanced language model.

1. **Navigate to the frontend directory**:
   ```bash
   cd hackathon-mercor-etched-cognition/frontend
   ```

2. **Install dependencies** (if not already done):
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

   This will start the frontend application on `http://localhost:3000`.

4. **Interact with the application**:
   - Open a web browser and navigate to `http://localhost:3000`
   - You should see the visualization interface for the EEG-enhanced language model
   - The interface will display tokens streamed from the backend with different colors based on the expert that generated them

## Running the EEG Processing Script

The EEG processing script processes EEG signals to extract attention metrics.

### Using Real EEG Data

1. **Connect your EEG device** (if using real EEG data):
   - Turn on your Muse headset
   - Pair it with your computer via Bluetooth
   - Start streaming EEG data using muselsl:
     ```bash
     muselsl stream --name YOUR_DEVICE_NAME
     ```

2. **Navigate to the EEG directory**:
   ```bash
   cd hackathon-mercor-etched-cognition/eeg
   ```

3. **Run the EEG processing script**:
   ```bash
   python eeg.py
   ```

   This will start processing EEG data and visualizing attention levels in real-time.

### Using Simulated EEG Data

If you don't have an EEG device, you can use simulated data:

1. **Navigate to the EEG directory**:
   ```bash
   cd hackathon-mercor-etched-cognition/eeg
   ```

2. **Run the EEG processing script with the simulation flag**:
   ```bash
   python eeg.py --simulate
   ```

   This will generate simulated EEG data and process it as if it were coming from a real device.

## Using the Kevin-MOE Demo

The Kevin-MOE demo implements the EEG-Enhanced Language Model with Mixture-of-Experts.

1. **Navigate to the Kevin-MOE demo directory**:
   ```bash
   cd hackathon-mercor-etched-cognition/kevin-moe-demo
   ```

2. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo script**:
   ```bash
   python demo.py
   ```

   This will start the demo with simulated EEG data.

4. **Run the demo with real EEG data** (if available):
   ```bash
   python demo.py --real-eeg
   ```

   This will use real EEG data from your connected device.

5. **Interact with the demo**:
   - Type prompts when prompted
   - Observe how the language model adapts its responses based on your attention levels
   - The demo will display which expert is being used for each token

## Advanced Usage

### Running the Full System

To run the full system with all components:

1. **Start the backend server** (in one terminal):
   ```bash
   cd hackathon-mercor-etched-cognition/backend
   uvicorn main:app --reload
   ```

2. **Start the EEG processing script** (in another terminal):
   ```bash
   cd hackathon-mercor-etched-cognition/eeg
   python eeg.py
   ```

3. **Start the frontend application** (in a third terminal):
   ```bash
   cd hackathon-mercor-etched-cognition/frontend
   npm start
   ```

4. **Start the Kevin-MOE demo** (in a fourth terminal):
   ```bash
   cd hackathon-mercor-etched-cognition/kevin-moe-demo
   python demo.py
   ```

### Customizing the System

#### Adjusting EEG Processing Parameters

You can adjust the EEG processing parameters in `eeg/eeg.py`:

- `FS`: EEG sampling rate (default: 256 Hz)
- `WINDOW_SIZE`: Window size for frequency analysis (default: 2 seconds)
- `STEP_SIZE`: Step size for sliding window (default: 1 second)
- `BANDS`: Frequency bands for analysis (theta, alpha, beta, gamma, sixty)
- `THRESHOLD_GAMMA`: Threshold for gamma band power to determine focus level

#### Customizing the Mixture-of-Experts Controller

You can customize the Mixture-of-Experts controller in `kevin-moe-demo/moe_control.py`:

- Adjust the expert configurations for different attention levels
- Modify the token biasing strategies
- Change the temperature and top-k sampling parameters

## API Endpoints

The backend server provides several API endpoints:

### `/add-tokens` (POST)

Add tokens to the streaming queue.

**Request Body**:
```json
{
  "tokens": [
    {
      "token": "Hello",
      "expert": "simple"
    },
    {
      "token": " ",
      "expert": "balanced"
    },
    {
      "token": "world",
      "expert": "complex"
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "tokens_added": 3
}
```

### `/stream` (GET)

Stream tokens to the frontend using server-sent events.

**Response**: Server-sent events with token data.

### `/clear-tokens` (POST)

Clear all tokens from the token queue.

**Response**:
```json
{
  "status": "success",
  "message": "Token queue cleared"
}
```

### `/toggle-test-data` (POST)

Enable or disable random test data generation.

**Request Body**:
```json
{
  "enable": true
}
```

**Response**:
```json
{
  "status": "success",
  "test_data_enabled": true
}
```

### `/status` (GET)

Get the current status of the server.

**Response**:
```json
{
  "queue_size": 10,
  "test_data_enabled": false
}
```

### `/debug-tokens` (GET)

Get a dump of recently processed tokens for debugging.

**Query Parameters**:
- `limit`: Maximum number of tokens to return (default: 50)

**Response**:
```json
{
  "queue_size": 15,
  "tokens": [
    {
      "word": "Hello",
      "number": 1
    },
    {
      "word": " ",
      "number": 2
    },
    {
      "word": "world",
      "number": 3
    }
  ],
  "test_data_enabled": false
}
```

For more detailed API documentation, see the [API Documentation](api.md).
