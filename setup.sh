#!/bin/bash

# AI Shield Setup Script
# This script sets up the AI Shield environment and dependencies

echo "=================================================="
echo "           AI Shield Setup Script"
echo "=================================================="

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv ai_shield_env

# Activate virtual environment
echo "Activating virtual environment..."
source ai_shield_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p logs
mkdir -p data/processed
mkdir -p models/trained
mkdir -p results

# Generate sample data and models
echo ""
echo "Generating sample data and models..."
python create_samples.py

# Check AWS CLI installation
echo ""
echo "Checking AWS CLI..."
aws --version

if [ $? -ne 0 ]; then
    echo "Warning: AWS CLI not found. Please install AWS CLI for S3 integration."
    echo "Visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
else
    echo "AWS CLI found. Please ensure your credentials are configured."
    echo "Run 'aws configure' if you haven't set up your credentials yet."
fi

echo ""
echo "=================================================="
echo "           Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source ai_shield_env/bin/activate"
echo "2. Configure AWS credentials: aws configure"
echo "3. Create S3 bucket for your data"
echo "4. Run a test: python main.py --help"
echo "5. Try the demo notebook: jupyter notebook notebooks/ai_shield_demo.ipynb"
echo ""
echo "For detailed AWS setup instructions, see: docs/aws_setup_guide.md"
echo ""