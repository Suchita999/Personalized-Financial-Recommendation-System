#!/bin/bash

# Financial Recommendation System - Startup Script
# This script starts all components of the system

echo "Starting Financial Recommendation System..."
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
if ! pip list | grep -q "streamlit"; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Set up environment
echo "🔧 Setting up environment..."
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Environment variables loaded"
else
    echo "No .env file found, using defaults"
fi

# Test RAG system
echo "Testing RAG system..."
python test_rag_integration.py
if [ $? -eq 0 ]; then
    echo "RAG system ready"
else
    echo "RAG system test failed, but continuing..."
fi

echo ""
echo "Choose how to run the system:"
echo ""
echo "1. Full Dashboard (Streamlit app with all features)"
echo "2. Chatbot Only (RAG-enhanced financial advisor)"
echo "3. Landing Page (Simple entry point)"
echo "4. Dashboard Only (Advanced analytics)"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Starting Full Dashboard..."
        streamlit run src/streamlit/app.py --server.port 8501
        ;;
    2)
        echo "Starting Chatbot Only..."
        streamlit run src/streamlit/streamlit_chatbot.py --server.port 8502
        ;;
    3)
        echo "Starting Landing Page..."
        streamlit run src/streamlit/landing_page.py --server.port 8503
        ;;
    4)
        echo "Starting Dashboard Only..."
        streamlit run src/streamlit/dashboard.py --server.port 8504
        ;;
    *)
        echo "Invalid choice. Starting Full Dashboard..."
        streamlit run src/streamlit/app.py --server.port 8501
        ;;
esac
