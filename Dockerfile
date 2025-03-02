# Multi-stage build for EEG-Enhanced Language Model with Mixture-of-Experts

# Stage 1: Build the frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend

# Copy frontend package files and install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy frontend source code and build
COPY frontend/ ./
RUN npm run build

# Stage 2: Python base for backend and EEG processing
FROM python:3.10-slim AS backend

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/
COPY eeg/ ./eeg/
COPY kevin-moe-demo/ ./kevin-moe-demo/

# Copy frontend build from the frontend-build stage
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Copy documentation
COPY docs/ ./docs/

# Copy setup files
COPY setup.py .
COPY README.md .
COPY LICENSE .

# Install the package
RUN pip install -e .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
