#!/bin/bash
"""
Launch script for the ML Model Search Streamlit Application
"""

echo "🚀 Starting ML Model Search Application..."
echo "📋 Make sure Elasticsearch is running and accessible"
echo "🔗 The app will be available at: http://localhost:8501"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
