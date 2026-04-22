#!/bin/bash

# KHUNEHO? Neural Analysis System - Run Script
# This script uses the nirmitbatabaran environment and runs the system

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "KHUNEHO? Neural Analysis System"
echo "Using nirmitbatabaran environment"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

# Check nirmitbatabaran environment
if [ ! -d "nirmitbatabaran" ]; then
    echo "Error: nirmitbatabaran environment not found"
    exit 1
fi

echo "Activating nirmitbatabaran environment..."
source nirmitbatabaran/bin/activate

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "Warning: models directory not found. Some models may need to be downloaded."
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Using default configuration."
fi

# Run
echo "Starting KHUNEHO?..."
python3 main.py

deactivate
