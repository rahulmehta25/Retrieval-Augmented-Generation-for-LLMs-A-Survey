#!/bin/bash

# Script to run the Streamlit RAG interface

echo "🚀 Starting RAG Streamlit Interface..."
echo "=================================="

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Virtual environment found"
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "📦 Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Check if Ollama is running (for local LLM)
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    
    # Check if gemma:2b model is available
    if ollama list | grep -q "gemma:2b"; then
        echo "✅ gemma:2b model is available"
    else
        echo "⚠️  gemma:2b model not found. Pulling it now..."
        ollama pull gemma:2b
    fi
else
    echo "⚠️  Ollama not found. The system will use fallback options."
fi

# Run Streamlit
echo ""
echo "🌐 Starting Streamlit server..."
echo "=================================="
echo "Open your browser and go to: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#667eea" \
    --theme.backgroundColor "#FFFFFF" \
    --theme.secondaryBackgroundColor "#F0F2F6" \
    --theme.textColor "#262730"