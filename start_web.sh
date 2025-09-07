#!/bin/bash

# AI Shield Web Interface Startup Script

echo "🚀 Starting AI Shield Web Interface..."
echo "======================================"

# Activate virtual environment
if [ -d "ai_shield_env" ]; then
    echo "📦 Activating virtual environment..."
    source ai_shield_env/bin/activate
else
    echo "❌ Virtual environment not found! Please run setup.sh first."
    exit 1
fi

# Install Flask if not already installed
echo "🔧 Checking Flask installation..."
pip show flask > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "📥 Installing Flask..."
    pip install flask>=2.3.0 werkzeug>=2.3.0
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads results static/css static/js templates

# Check if sample data exists
if [ ! -f "models/sample_model.pkl" ] || [ ! -f "data/sample_dataset.csv" ]; then
    echo "📊 Generating sample data..."
    python create_samples.py
fi

# Export Flask app
export FLASK_APP=app.py
export FLASK_ENV=development

# Start the web server
echo ""
echo "🌐 Starting AI Shield Web Interface..."
echo "======================================"
echo "Starting AI Shield Web Interface..."
echo "Dashboard: http://localhost:5001"
echo "Upload: http://localhost:5001/upload"
echo "API: http://localhost:5001/api"
echo "================================"

# Start the Flask app on port 5001
export FLASK_RUN_PORT=5001
python app.py
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================"

python app.py