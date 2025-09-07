# AI Shield - Adversarial Machine Learning Security Analysis

## Overview
AI Shield is a cybersecurity project that performs adversarial machine learning security analysis on user-provided AI models. The system can execute adversarial attacks and generate explainable AI (XAI) insights to identify model vulnerabilities.

## Features
- **Adversarial Attacks**: Implements FGSM and PGD attacks using IBM's Adversarial Robustness Toolbox (ART)
- **XAI Analysis**: Provides explanations using SHAP and LIME for understanding model vulnerabilities
- **AWS Integration**: Designed for deployment on AWS SageMaker with S3 storage
- **Modular Architecture**: Clean separation of concerns for easy maintenance and extension

## Project Structure
```
AI-Shield/
├── src/                    # Source code
│   ├── adversarial/       # Adversarial attack modules
│   ├── xai/              # XAI explanation modules
│   ├── aws/              # AWS integration utilities
│   └── utils/            # Common utilities
├── data/                  # Data storage
├── models/               # Model storage
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks for experiments
├── requirements.txt      # Python dependencies
└── main.py              # Main execution script
```

## Installation

### Prerequisites
- Python 3.11+
- AWS CLI configured
- VS Code with AWS Toolkit extension

### Setup
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure AWS credentials (see AWS Setup section)

## AWS Setup with VS Code

### 1. Install AWS Toolkit for VS Code
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "AWS Toolkit"
4. Install the AWS Toolkit extension

### 2. Configure AWS Credentials
1. Open Command Palette (Ctrl+Shift+P)
2. Type "AWS: Create Credentials Profile"
3. Follow the prompts to enter:
   - Access Key ID
   - Secret Access Key
   - Default region (e.g., us-east-1)

### 3. Create S3 Bucket
1. Open Command Palette (Ctrl+Shift+P)
2. Type "AWS: Create S3 Bucket"
3. Enter bucket name (e.g., "ai-shield-data-bucket")
4. Select your region

### 4. Set up SageMaker Notebook Instance
1. Open AWS Toolkit panel in VS Code
2. Navigate to SageMaker section
3. Right-click and select "Create Notebook Instance"
4. Configure:
   - Instance name: "ai-shield-notebook"
   - Instance type: ml.t3.medium (for testing)
   - IAM role: Create new role with S3 access

## Usage

### Basic Usage
```python
from main import AIShieldEngine

# Initialize the engine
engine = AIShieldEngine(
    bucket_name="your-s3-bucket",
    aws_region="us-east-1"
)

# Run analysis
results = engine.run_security_analysis(
    model_key="models/your_model.pkl",
    data_key="data/test_data.csv"
)

# View results
print(results)
```

### Running from Command Line
```bash
python main.py --bucket your-s3-bucket --model models/your_model.pkl --data data/test_data.csv
```

## Configuration
Edit `config/config.yaml` to customize:
- Attack parameters (epsilon values, iterations)
- XAI explanation settings
- AWS configuration
- Output formats

## Output Format
The analysis generates a structured JSON output containing:
- Attack success rates
- Adversarial examples
- SHAP feature importance
- LIME explanations
- Vulnerability summary

## Dependencies
See `requirements.txt` for complete list of dependencies including:
- PyTorch/TensorFlow for model handling
- IBM ART for adversarial attacks
- SHAP and LIME for explanations
- Boto3 for AWS integration
- SageMaker SDK

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Submit a pull request

## License
This project is licensed under the MIT License.

## Support
For issues and questions, please create an issue in the repository.