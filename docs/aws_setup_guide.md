# AWS Setup Guide for AI Shield

This guide provides step-by-step instructions for setting up AWS integration with AI Shield using VS Code.

## Prerequisites

- VS Code installed
- Python 3.11+ installed
- AWS account with appropriate permissions

## Step 1: Install AWS Toolkit for VS Code

1. Open VS Code
2. Click on the Extensions icon (Ctrl+Shift+X / Cmd+Shift+X)
3. Search for "AWS Toolkit"
4. Install the "AWS Toolkit" extension by Amazon Web Services
5. Restart VS Code after installation

## Step 2: Configure AWS Credentials

### Option A: Using AWS CLI (Recommended)

1. Install AWS CLI if not already installed:
   ```bash
   # macOS
   brew install awscli
   
   # Windows
   pip install awscli
   
   # Linux
   sudo apt-get install awscli
   ```x

2. Configure AWS credentials:
   ```bash
   aws configure
   ```
   
   Enter your:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region (e.g., us-east-1)
   - Default output format (json)

### Option B: Using VS Code AWS Toolkit

1. Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Type "AWS: Create Credentials Profile"
3. Follow the prompts to enter your AWS credentials
4. Select your default region

## Step 3: Create S3 Bucket for AI Shield

### Using AWS Toolkit in VS Code:

1. Open the AWS Toolkit panel (View > Open View > AWS Toolkit)
2. Expand the "S3" section
3. Right-click and select "Create Bucket"
4. Enter bucket name: `ai-shield-data-bucket-[your-unique-id]`
5. Select your region
6. Click "Create"

### Using AWS CLI:

```bash
# Replace 'your-unique-id' with something unique
aws s3 mb s3://ai-shield-data-bucket-your-unique-id --region us-east-1
```

## Step 4: Set up SageMaker Notebook Instance

### Using AWS Toolkit in VS Code:

1. In the AWS Toolkit panel, expand "SageMaker"
2. Right-click on "Notebook Instances" and select "Create Notebook Instance"
3. Configure the instance:
   - **Instance name**: `ai-shield-notebook`
   - **Instance type**: `ml.t3.medium` (for development) or `ml.m5.large` (for production)
   - **Platform identifier**: `notebook-al2-v2` (Amazon Linux 2)
   - **IAM role**: Create new role or select existing role with S3 access

### Using AWS CLI:

```bash
# Create IAM role for SageMaker (if not exists)
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

# Attach policies to the role
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create notebook instance
aws sagemaker create-notebook-instance \
    --notebook-instance-name ai-shield-notebook \
    --instance-type ml.t3.medium \
    --role-arn arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerExecutionRole
```

## Step 5: Upload Sample Data and Models

### Create sample data and model files:

1. Run the following Python script to create sample files:

```python
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

# Save dataset
df.to_csv('data/sample_dataset.csv', index=False)

# Train and save a sample model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open('models/sample_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Sample data and model created!")
```

### Upload to S3:

```bash
# Upload sample data
aws s3 cp data/sample_dataset.csv s3://ai-shield-data-bucket-your-unique-id/data/

# Upload sample model
aws s3 cp models/sample_model.pkl s3://ai-shield-data-bucket-your-unique-id/models/
```

## Step 6: Configure AI Shield for AWS

1. Update the configuration file `config/config.yaml`:

```yaml
aws:
  region: "us-east-1"
  bucket_name: "ai-shield-data-bucket-your-unique-id"  # Replace with your bucket name
  profile: "default"
```

## Step 7: Test the Setup

### Run a test analysis:

```bash
python main.py \
    --bucket ai-shield-data-bucket-your-unique-id \
    --model models/sample_model.pkl \
    --data data/sample_dataset.csv \
    --output test_results.json
```

### Expected output:
```
AI SHIELD SECURITY ANALYSIS COMPLETE
====================================
Overall Vulnerability Score: 0.25
Execution Time: 45.23 seconds
Results saved to: test_results.json

Attack Success Rates:
  FGSM: 0.22
  PGD: 0.28

Recommendations:
  - Low vulnerability - monitor for new attack vectors
```

## Step 8: Deploy to SageMaker

### Option A: Using SageMaker Notebook

1. Access your SageMaker notebook instance through the AWS console
2. Upload the AI Shield project files
3. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
4. Run the demo notebook: `notebooks/ai_shield_demo.ipynb`

### Option B: Using SageMaker Processing Job

1. Create a processing script:

```python
# processing_script.py
import sys
import subprocess

# Install requirements
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Run AI Shield analysis
from main import AIShieldEngine

engine = AIShieldEngine(
    bucket_name="ai-shield-data-bucket-your-unique-id",
    aws_region="us-east-1"
)

results = engine.run_security_analysis(
    model_key="models/sample_model.pkl",
    data_key="data/sample_dataset.csv"
)

print("Analysis complete!")
```

2. Submit processing job:

```python
import boto3
import sagemaker

session = sagemaker.Session()
role = "arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerExecutionRole"

processor = sagemaker.processing.ScriptProcessor(
    command=['python3'],
    image_uri='python:3.11',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large'
)

processor.run(
    code='processing_script.py',
    inputs=[
        sagemaker.processing.ProcessingInput(
            source='s3://ai-shield-data-bucket-your-unique-id/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name='results',
            source='/opt/ml/processing/output'
        )
    ]
)
```

## Troubleshooting

### Common Issues:

1. **"NoCredentialsError"**
   - Ensure AWS credentials are properly configured
   - Check `aws configure list` to verify credentials

2. **"Bucket does not exist"**
   - Verify bucket name in configuration
   - Ensure bucket exists in the correct region

3. **"Access Denied"**
   - Check IAM permissions for S3 and SageMaker
   - Ensure your user/role has necessary policies attached

4. **"Module not found"**
   - Install requirements: `pip install -r requirements.txt`
   - Ensure you're in the correct Python environment

### Useful Commands:

```bash
# Check AWS configuration
aws configure list

# List S3 buckets
aws s3 ls

# List SageMaker notebook instances
aws sagemaker list-notebook-instances

# Check SageMaker instance status
aws sagemaker describe-notebook-instance --notebook-instance-name ai-shield-notebook
```

## Security Best Practices

1. **Use IAM roles with minimal necessary permissions**
2. **Enable S3 bucket encryption**
3. **Use VPC endpoints for SageMaker if possible**
4. **Regularly rotate AWS access keys**
5. **Monitor CloudTrail logs for unusual activity**

## Next Steps

Once setup is complete:

1. **Customize attack parameters** in `config/config.yaml`
2. **Add your own models and datasets** to S3
3. **Explore the demo notebook** for detailed examples
4. **Set up automated security scanning** using SageMaker scheduled jobs
5. **Integrate with your CI/CD pipeline** for continuous security testing

For additional support, refer to the main README.md file or create an issue in the project repository.