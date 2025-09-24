# AWS SageMaker Customer Churn Prediction

A comprehensive machine learning project that demonstrates how to train and deploy a customer churn prediction model using Amazon SageMaker.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#p#### 2. Role#### 3. S3 A#### 4. Endp#### 5. Training Job Failures
```
Algorithm error: Training job failed
```
**Solution**: Check training script, data format, hyperparameters, and S3 permissionsDeployment Timeout
```
Failed to deploy model
```
**Solution**: Check CloudWatch logs, verify inference script, ensure IAM permissions

#### 5. Training Job Failuresrrors
```
NoSuchBucket: The specified bucket does not exist
```
**Solution**: Verify bucket name and region, update S3 ARNs in IAM policy

#### 4. Endpoint Deployment Timeoutsion Errors
```
UnauthorizedOperation: You are not authorized to perform this operation
```
**Solution**: Ensure your SageMaker role has proper permissions (see IAM Policy Setup above)

#### 3. S3 Access Errorssites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Model Details](#model-details)
- [API Reference](#api-reference)
- [Cost Considerations](#cost-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🔍 Overview

This project implements an end-to-end machine learning pipeline for predicting customer churn using:

- **Amazon SageMaker** for training and deployment
- **Random Forest Classifier** for the ML model
- **SKLearn framework** for SageMaker integration
- **Real-time inference** via SageMaker endpoints

## 📁 Project Structure

```
customer-churn-aws-ml/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── sagemaker-iam-policy.json             # IAM policy for SageMaker permissions
├── ml_experiment_mlflow.ipynb             # MLflow experiment notebook
├── ml_experiment.ipynb                    # Local ML experiment notebook
├── data/
│   ├── customer_churn.csv                 # Original dataset
│   └── customer_churn_processed.csv       # Processed dataset
├── model/                                 # Model artifacts (gitignored)
│   ├── best_model_xgboost.joblib
│   └── preprocessor.joblib
└── sagemaker/
    ├── sagemaker_training_deployment.ipynb # Main SageMaker notebook
    ├── training.py                        # Training script for SageMaker
    └── inference.py                       # Inference script for deployment
```

## 🔧 Prerequisites

### AWS Requirements
- AWS Account with SageMaker access
- Properly configured IAM role with SageMaker permissions
- S3 bucket for storing data and model artifacts

### Required IAM Permissions

Your SageMaker execution role needs the following permissions:

#### Option 1: AWS Managed Policies (Recommended for Development)
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (or specific bucket permissions)
- `IAMReadOnlyAccess`

#### Option 2: Custom IAM Policy (Recommended for Production)
Use the provided `sagemaker-iam-policy.json` file for fine-grained permissions:

```bash
# Create IAM policy using AWS CLI
aws iam create-policy \
    --policy-name SageMakerCustomerChurnPolicy \
    --policy-document file://sagemaker-iam-policy.json

# Attach policy to your SageMaker execution role
aws iam attach-role-policy \
    --role-name YourSageMakerExecutionRole \
    --policy-arn arn:aws:iam::YOUR-ACCOUNT-ID:policy/SageMakerCustomerChurnPolicy
```

**Note**: Update the S3 bucket ARNs and MLflow tracking server ARN in the policy file to match your resources.

### Local Environment
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- AWS CLI configured (optional but recommended)

## ⚙️ Setup Instructions

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd aws-ml
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure AWS Credentials
```bash
aws configure
```

### 4. Prepare Your Data
- Upload your processed customer churn dataset to S3
- Ensure the CSV file is named `customer_churn_processed.csv`
- The target variable should be named `Churn`

### 5. Update Configuration
In the notebook, update these variables:
- `bucket`: Your S3 bucket name
- `role`: Your SageMaker execution role ARN

## 🚀 Usage

### Running the Complete Pipeline

1. **Open the main notebook:**
   ```bash
   jupyter notebook sagemaker_training_deployment.ipynb
   ```

2. **Execute cells in order:**
   - Setup and imports
   - Model training
   - Model deployment
   - Testing predictions
   - Cleanup

### Quick Start Commands

```python
# In your notebook or Python script
import sagemaker

# Get SageMaker session and role
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Train model (see notebook for full example)
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large'
)

# Deploy model
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)
```

## 📄 File Descriptions

### Core Files

#### `sagemaker-iam-policy.json`
Custom IAM policy document that provides:
- S3 bucket access for data and model storage
- SageMaker permissions for training and deployment
- MLflow tracking server access for experiment management
- Minimal permissions following security best practices

#### `sagemaker/sagemaker_training_deployment.ipynb`
The main Jupyter notebook containing:
- Complete end-to-end ML pipeline
- Step-by-step instructions
- Explanatory markdown cells
- Error handling examples

#### `sagemaker/training.py`
SageMaker training script that:
- Loads data from S3
- Trains Random Forest model
- Saves model artifacts
- Supports hyperparameter tuning

#### `sagemaker/inference.py`
SageMaker inference script that:
- Loads trained model
- Processes input data (JSON/CSV)
- Returns predictions with probabilities
- Handles multiple content types

#### `requirements.txt`
Python dependencies including:
- Core ML libraries (scikit-learn, pandas, numpy)
- AWS libraries (sagemaker, boto3)
- Jupyter ecosystem packages

## 🤖 Model Details

### Algorithm
- **Model Type**: Random Forest Classifier
- **Framework**: Scikit-learn
- **Default Parameters**:
  - `n_estimators`: 100
  - `max_depth`: 10
  - `random_state`: 42

### Features
- Accepts multiple input formats (JSON, CSV)
- Returns both predictions and probabilities
- Supports real-time inference
- Scalable deployment options

### Performance Considerations
- Training time: ~5-10 minutes on ml.m5.large
- Inference latency: <100ms typical response time
- Throughput: Depends on instance type

## 📡 API Reference

### Endpoint Input Formats

#### JSON Format
```json
{
    "feature1": 25.5,
    "feature2": 100,
    "feature3": 1
}
```

#### CSV Format
```
25.5,100,1
```

### Response Format
```json
{
    "predictions": [0],
    "probabilities": [[0.8, 0.2]]
}
```

### Python Client Example
```python
import json

# JSON prediction
result = predictor.predict(
    {"feature1": 25.5, "feature2": 100},
    initial_args={'ContentType': 'application/json'}
)

# CSV prediction
result = predictor.predict(
    "25.5,100,1",
    initial_args={'ContentType': 'text/csv'}
)
```

## 💰 Cost Considerations

### Training Costs
- **Instance**: ml.m5.large (~$0.115/hour)
- **Typical Duration**: 5-10 minutes
- **Estimated Cost**: <$0.02 per training job

### Inference Costs
- **Instance**: ml.t2.medium (~$0.056/hour)
- **Minimum Billing**: 1 hour when endpoint is created
- **Ongoing**: Charged for uptime regardless of usage

### Cost Optimization Tips
- Delete endpoints when not in use
- Use smaller instances for development
- Consider batch transform for bulk predictions

## 🔧 Troubleshooting

### Common Issues

#### 1. IAM Policy Setup
**Issue**: Setting up custom IAM permissions
```bash
# Step 1: Update the policy file with your specific resources
# Edit sagemaker-iam-policy.json:
# - Replace "SageMaker" with your actual S3 bucket name
# - Update the MLflow tracking server ARN with your account ID and region

# Step 2: Create the policy
aws iam create-policy \
    --policy-name SageMakerCustomerChurnPolicy \
    --policy-document file://sagemaker-iam-policy.json

# Step 3: Attach to your SageMaker execution role
aws iam attach-role-policy \
    --role-name YourSageMakerExecutionRoleName \
    --policy-arn arn:aws:iam::YOUR-ACCOUNT-ID:policy/SageMakerCustomerChurnPolicy
```

#### 2. Role Permission Errors
```
UnauthorizedOperation: You are not authorized to perform this operation
```
**Solution**: Ensure your SageMaker role has proper permissions

#### 2. S3 Access Errors
```
NoSuchBucket: The specified bucket does not exist
```
**Solution**: Verify bucket name and region

#### 3. Endpoint Deployment Timeout
```
Failed to deploy model
```
**Solution**: Check CloudWatch logs, verify inference script

#### 4. Training Job Failures
```
Algorithm error: Training job failed
```
**Solution**: Check training script, data format, and hyperparameters

### Debug Commands
```python
# Check training job status
print(sklearn_estimator.latest_training_job.job_name)

# View logs
sklearn_estimator.logs()

# Check endpoint status
print(predictor.endpoint_name)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for changes
- Test with different data formats

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check AWS SageMaker documentation
- Review CloudWatch logs for debugging

## 🔗 Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [AWS ML University](https://aws.amazon.com/machine-learning/mlu/)

---

**Note**: Remember to clean up AWS resources when done to avoid unnecessary charges!
