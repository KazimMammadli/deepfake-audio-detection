#!/bin/bash

echo "========================================"
echo "Deepfake Audio Detection - Streamlit App"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10+ from https://www.python.org/"
    exit 1
fi

echo "[1/3] Checking dependencies..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
else
    echo "Dependencies OK"
fi

echo ""
echo "[2/3] Checking model files..."
if [ ! -f "models/resnet50_v2.keras" ]; then
    echo "WARNING: Model files not found in models/ directory"
    echo "Please ensure you have the trained models"
    echo ""
fi

echo "[3/3] Starting Streamlit app..."
echo ""
echo "App will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""
streamlit run app.py
