# Installation Guide

This guide provides detailed instructions for setting up the EEG-Enhanced Language Model with Mixture-of-Experts project on different operating systems.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Linux](#linux)
  - [macOS](#macos)
  - [Windows](#windows)
- [Setting Up EEG Hardware](#setting-up-eeg-hardware)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before installing the project, ensure you have the following prerequisites:

### Software Requirements

- **Python**: Version 3.8 or higher
- **Node.js**: Version 14.0 or higher
- **npm**: Version 6.0 or higher
- **Git**: For cloning the repository

### Hardware Requirements

- **EEG Device** (optional): The project works with Muse EEG headsets, but can also run with simulated EEG data
- **Computer**: Any modern computer with at least 4GB of RAM and 2GB of free disk space

### Python Packages

The project requires several Python packages, which will be installed automatically during the setup process:

- FastAPI
- Uvicorn
- PyTorch
- Transformers
- NumPy
- Matplotlib
- SciPy
- PyLSL (for EEG data streaming)

## Installation

### Linux

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/hackathon-mercor-etched-cognition.git
   cd hackathon-mercor-etched-cognition
   ```

2. **Set up a Python virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Install the package in development mode** (optional):
   ```bash
   pip install -e .
   ```

### macOS

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/hackathon-mercor-etched-cognition.git
   cd hackathon-mercor-etched-cognition
   ```

2. **Set up a Python virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Install the package in development mode** (optional):
   ```bash
   pip install -e .
   ```

### Windows

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/hackathon-mercor-etched-cognition.git
   cd hackathon-mercor-etched-cognition
   ```

2. **Set up a Python virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Install the package in development mode** (optional):
   ```bash
   pip install -e .
   ```

## Setting Up EEG Hardware

### Muse Headset Setup

If you have a Muse EEG headset, follow these steps to set it up:

1. **Install the Muse SDK**:
   ```bash
   pip install muselsl
   ```

2. **Connect your Muse headset**:
   - Turn on your Muse headset
   - Pair it with your computer via Bluetooth
   - Run the following command to list available Muse devices:
     ```bash
     muselsl list
     ```

3. **Stream EEG data**:
   ```bash
   muselsl stream --name YOUR_DEVICE_NAME
   ```

4. **Verify the connection**:
   - Run the EEG visualization script to verify that data is being received:
     ```bash
     cd eeg
     python eeg.py
     ```

### Using Simulated EEG Data

If you don't have an EEG headset, you can use simulated data:

1. **Enable test data generation**:
   - Start the backend server (see [Usage Guide](usage.md))
   - Send a POST request to enable test data:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"enable": true}' http://localhost:8000/toggle-test-data
     ```

## Troubleshooting

### Common Issues

#### Python Dependencies Installation Fails

**Problem**: Error when installing Python dependencies.

**Solution**:
- Ensure you have the latest pip version:
  ```bash
  pip install --upgrade pip
  ```
- Try installing dependencies one by one to identify the problematic package
- For PyTorch installation issues, visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for specific installation instructions for your system

#### EEG Device Not Detected

**Problem**: The system cannot detect your EEG device.

**Solution**:
- Ensure your device is charged and turned on
- Check Bluetooth connectivity
- Restart your device and computer
- Try reinstalling the Muse SDK:
  ```bash
  pip uninstall muselsl
  pip install muselsl
  ```

#### Frontend Build Fails

**Problem**: Error when building or running the frontend.

**Solution**:
- Ensure you have the correct Node.js version:
  ```bash
  node --version
  ```
- Clear npm cache:
  ```bash
  npm cache clean --force
  ```
- Delete node_modules and reinstall:
  ```bash
  cd frontend
  rm -rf node_modules
  npm install
  ```

#### Backend Server Won't Start

**Problem**: Error when starting the FastAPI backend server.

**Solution**:
- Check if the port is already in use:
  ```bash
  # Linux/macOS
  lsof -i :8000
  
  # Windows
  netstat -ano | findstr :8000
  ```
- Kill the process using the port if necessary
- Ensure all dependencies are installed correctly

### Getting Help

If you encounter issues not covered in this troubleshooting guide, please:

1. Check the [GitHub Issues](https://github.com/yourusername/hackathon-mercor-etched-cognition/issues) to see if someone has reported a similar problem
2. Create a new issue with detailed information about your problem, including:
   - Operating system and version
   - Python and Node.js versions
   - Error messages
   - Steps to reproduce the issue
