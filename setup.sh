#!/bin/bash

# Multi-Neuron AI Chat System Setup Script
# This script sets up the complete environment for the AI chat system

set -e

echo "🚀 Setting up Multi-Neuron AI Chat System..."

# Create project structure
echo "📁 Creating project structure..."
mkdir -p models app static

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download models (this will take time)
echo "🧠 Downloading models (this may take 10-20 minutes)..."

# Create models directory if it doesn't exist
mkdir -p models

# Download Qwen2-7B-Instruct (GGUF Q4_K_M)
echo "Downloading Qwen2-7B-Instruct..."
if [ ! -f "models/qwen2-7b-instruct-q4_k_m.gguf" ]; then
    curl -L -o models/qwen2-7b-instruct-q4_k_m.gguf \
        "https://huggingface.co/TheBloke/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf"
fi

# Download Mistral-7B-Instruct-v0.3 (GGUF Q4_K_M)
echo "Downloading Mistral-7B-Instruct-v0.3..."
if [ ! -f "models/mistral-7b-instruct-v0.3-q4_k_m.gguf" ]; then
    curl -L -o models/mistral-7b-instruct-v0.3-q4_k_m.gguf \
        "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3-q4_k_m.gguf"
fi

# Download DeepSeek-LLM-7B-Chat (GGUF Q4_K_M)
echo "Downloading DeepSeek-LLM-7B-Chat..."
if [ ! -f "models/deepseek-llm-7b-chat-q4_k_m.gguf" ]; then
    curl -L -o models/deepseek-llm-7b-chat-q4_k_m.gguf \
        "https://huggingface.co/TheBloke/deepseek-llm-7b-chat-GGUF/resolve/main/deepseek-llm-7b-chat-q4_k_m.gguf"
fi

echo "✅ Setup complete!"
echo "🎯 To start the system:"
echo "   source venv/bin/activate"
echo "   python app/main.py"
echo ""
echo "🌐 The system will be available at: http://localhost:8000"
