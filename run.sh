#!/bin/bash

# KHUNEHO? Neural Analysis System - Run Script
# This script sets up the environment and runs the system

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "KHUNEHO? Neural Analysis System"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

# Virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q torch transformers duckduckgo-search python-dotenv

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Using default configuration."
fi

# Run
echo "Starting KHUNEHO?..."
python3 main.py

deactivate
