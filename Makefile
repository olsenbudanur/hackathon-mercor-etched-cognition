.PHONY: setup backend frontend test clean docs venv

# Variables
VENV_NAME := venv
PYTHON := python3
PIP := pip3
PYTEST := pytest
UVICORN := uvicorn
NPM := npm

# Setup virtual environment and install dependencies
setup: venv
	@echo "Installing Python dependencies..."
	$(VENV_NAME)/bin/$(PIP) install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && $(NPM) install

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Virtual environment created at $(VENV_NAME)/"

# Run backend server
backend:
	@echo "Starting backend server..."
	cd backend && $(VENV_NAME)/bin/$(UVICORN) main:app --reload --host 0.0.0.0 --port 8000

# Run frontend application
frontend:
	@echo "Starting frontend application..."
	cd frontend && $(NPM) start

# Run tests
test: test-backend test-eeg test-frontend test-integration

# Run backend tests
test-backend:
	@echo "Running backend tests..."
	$(VENV_NAME)/bin/$(PYTEST) tests/backend/

# Run EEG tests
test-eeg:
	@echo "Running EEG tests..."
	$(VENV_NAME)/bin/$(PYTEST) tests/eeg/

# Run frontend tests
test-frontend:
	@echo "Running frontend tests..."
	cd frontend && $(NPM) test

# Run integration tests
test-integration:
	@echo "Running integration tests..."
	$(VENV_NAME)/bin/$(PYTEST) tests/integration/

# Run demo
demo:
	@echo "Running Kevin-MOE demo..."
	$(VENV_NAME)/bin/$(PYTHON) kevin-moe-demo/demo.py

# Build documentation
docs:
	@echo "Building documentation..."
	@echo "Documentation is in markdown format in the docs/ directory"

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf frontend/node_modules
	rm -rf $(VENV_NAME)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install package in development mode
develop: venv
	@echo "Installing package in development mode..."
	$(VENV_NAME)/bin/$(PIP) install -e .

# Package the project
package: clean
	@echo "Packaging the project..."
	$(VENV_NAME)/bin/$(PYTHON) setup.py sdist bdist_wheel

# Help command
help:
	@echo "Available commands:"
	@echo "  setup         - Set up virtual environment and install dependencies"
	@echo "  backend       - Run the backend server"
	@echo "  frontend      - Run the frontend application"
	@echo "  test          - Run all tests"
	@echo "  test-backend  - Run backend tests"
	@echo "  test-eeg      - Run EEG tests"
	@echo "  test-frontend - Run frontend tests"
	@echo "  test-integration - Run integration tests"
	@echo "  demo          - Run the Kevin-MOE demo"
	@echo "  docs          - Build documentation"
	@echo "  clean         - Clean up temporary files"
	@echo "  develop       - Install package in development mode"
	@echo "  package       - Package the project"
	@echo "  help          - Show this help message"
