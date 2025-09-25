# Customer Churn Prediction with AWS SageMaker & MLflow

A comprehensive machine learning project demonstrating end-to-end customer churn prediction using Amazon SageMaker, MLflow experiment tracking, and production-ready deployment strategies.

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Cost Considerations](#cost-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## �� Overview

This project implements a complete machine learning pipeline for predicting customer churn, featuring:

- **Local Experimentation**: Jupyter notebooks for data exploration and model development
- **MLflow Integration**: Comprehensive experiment tracking and model registry
- **AWS SageMaker**: Scalable training and deployment on cloud infrastructure
- **Production Ready**: IAM policies, monitoring, and best practices implementation

### Key Technologies

- **Machine Learning**: Scikit-learn, XGBoost, Random Forest
- **Experiment Tracking**: MLflow with local and cloud backends
- **Cloud Platform**: Amazon SageMaker for training and inference
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization**: Matplotlib, Seaborn for insights

## ✨ Features

### 🧪 Experiment Management
- MLflow integration for experiment tracking
- Automated model comparison and selection
- Performance visualization and reporting
- Model registry with versioning and staging

### �� AWS SageMaker Integration
- Scalable model training on cloud infrastructure
- Real-time inference endpoints
- Batch prediction capabilities
- Automated hyperparameter tuning

### 🔒 Production Ready
- Custom IAM policies for security
- Comprehensive error handling
- Monitoring and logging setup
- Cost optimization strategies

### 📊 Comprehensive Analysis
- Exploratory data analysis with visualizations
- Feature importance analysis
- Model performance comparison
- Business impact assessment

## 📁 Project Structure

```
customer-churn-aws-ml/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package setup configuration
├── Makefile                           # Build and deployment commands
├── .gitignore                         # Git ignore patterns
├── sagemaker-iam-policy.json          # AWS IAM policy for SageMaker
│
├── data/                              # Dataset storage
│   ├── customer_churn.csv             # Raw customer data
│   └── customer_churn_processed.csv   # Preprocessed training data
│
├── model/                             # Saved model artifacts (gitignored)
│   ├── best_model_xgboost.joblib      # Best performing model
│   └── preprocessor.joblib            # Data preprocessing pipeline
│
├── sagemaker/                         # AWS SageMaker components
│   ├── sagemaker_e2e.ipynb           # End-to-end SageMaker workflow
│   ├── training.py                    # SageMaker training script
│   └── inference.py                  # SageMaker inference script
│
├── ml_experiment.ipynb                # Local ML experimentation
└── ml_experiment_mlflow.ipynb         # MLflow experiment tracking
```

## 🔧 Prerequisites

### AWS Requirements
- **AWS Account** with SageMaker access
- **IAM Role** with appropriate permissions
- **S3 Bucket** for data and model storage
- **AWS CLI** configured (optional but recommended)

### Local Environment
- **Python 3.8+**
- **Jupyter Notebook** or **JupyterLab**
- **Git** for version control

### Required Python Packages
See `requirements.txt` for complete list:
```bash
sagemaker>=2.0.0
mlflow>=2.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.0.0
seaborn>=0.11.0
xgboost>=1.0.0
boto3>=1.0.0
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/ashishpal2702/customer-churn-aws-ml.git
cd customer-churn-aws-ml
pip install -r requirements.txt
```

### 2. Local Experimentation
```bash
# Start with local ML experiments
jupyter notebook ml_experiment.ipynb

# Try MLflow experiment tracking
jupyter notebook ml_experiment_mlflow.ipynb
```

### 3. AWS SageMaker Deployment
```bash
# Configure AWS credentials
aws configure

# Run SageMaker workflow
jupyter notebook sagemaker/sagemaker_e2e.ipynb
```

## ⚙️ Detailed Setup

### AWS IAM Configuration

#### Option 1: Quick Setup (Development)
Attach these managed policies to your SageMaker execution role:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

#### Option 2: Secure Setup (Production)
Use the provided custom IAM policy:

```bash
# Update the policy file with your resources
# Edit sagemaker-iam-policy.json:
# - Replace "SageMaker" with your actual S3 bucket name
# - Update account ID and region in MLflow ARNs

# Create custom policy
aws iam create-policy \
    --policy-name SageMakerChurnPredictionPolicy \
    --policy-document file://sagemaker-iam-policy.json

# Attach to your SageMaker execution role
aws iam attach-role-policy \
    --role-name YourSageMakerExecutionRole \
    --policy-arn arn:aws:iam::YOUR-ACCOUNT-ID:policy/SageMakerChurnPredictionPolicy
```

### MLflow Setup

#### Local Tracking
```bash
# MLflow UI will start automatically in notebooks
# Or start manually:
mlflow ui --backend-store-uri ./mlruns
```

#### Remote Tracking (Optional)
```bash
# Set remote tracking server
export MLFLOW_TRACKING_URI=https://your-mlflow-server.com
```

## 📖 Usage Guide

### 1. Data Exploration
Start with `ml_experiment.ipynb` to:
- Load and explore the customer churn dataset
- Perform exploratory data analysis
- Understand feature relationships and distributions

### 2. MLflow Experiment Tracking
Use `ml_experiment_mlflow.ipynb` to:
- Train multiple models with automated tracking
- Compare model performance across experiments
- Register best models in MLflow registry
- Generate comprehensive experiment reports

### 3. SageMaker Deployment
Execute `sagemaker/sagemaker_e2e.ipynb` to:
- Upload data to S3
- Train models on SageMaker infrastructure
- Deploy models to real-time endpoints
- Test inference with sample data

### 4. Model Inference

#### Local Inference
```python
import joblib
import pandas as pd

# Load saved model and preprocessor
model = joblib.load('model/best_model_xgboost.joblib')
preprocessor = joblib.load('model/preprocessor.joblib')

# Make predictions
sample_data = pd.DataFrame({...})  # Your data
processed_data = preprocessor.transform(sample_data)
predictions = model.predict(processed_data)
```

#### SageMaker Endpoint
```python
import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime')

# Prepare data
payload = json.dumps({
    "instances": [{
        "feature1": value1,
        "feature2": value2,
        # ... more features
    }]
})

# Get prediction
response = runtime.invoke_endpoint(
    EndpointName='customer-churn-endpoint',
    ContentType='application/json',
    Body=payload
)

result = json.loads(response['Body'].read())
```

## 📊 Model Performance

### Current Best Model: XGBoost Classifier

| Metric    | Score |
|-----------|-------|
| ROC-AUC   | 0.887 |
| Accuracy  | 0.834 |
| Precision | 0.801 |
| Recall    | 0.743 |
| F1-Score  | 0.771 |

### Feature Importance
Top 5 most important features:
1. **Total Spend** (0.234) - Customer's total spending amount
2. **Usage Frequency** (0.198) - How often customer uses the service
3. **Support Calls** (0.156) - Number of customer support interactions
4. **Tenure** (0.134) - Length of customer relationship
5. **Payment Delay** (0.127) - Average payment delay in days

## 📡 API Reference

### SageMaker Endpoint

**Endpoint URL**: `https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint-name}/invocations`

#### Request Format
```json
{
  "instances": [{
    "Age": 35,
    "Tenure": 12,
    "Usage_Frequency": 25.5,
    "Support_Calls": 2,
    "Payment_Delay": 1,
    "Total_Spend": 1250.75,
    "Last_Interaction": 7,
    "Gender_Male": 1,
    "Subscription_Type_Premium": 1,
    "Contract_Length_Monthly": 0
  }]
}
```

#### Response Format
```json
{
  "predictions": [0],
  "probabilities": [[0.78, 0.22]],
  "confidence": 0.78
}
```

### MLflow Model Registry

**Model URI**: `models:/{model_name}/{version|stage}`

```python
# Load model from registry
import mlflow

model = mlflow.sklearn.load_model("models:/customer-churn-predictor/Production")
```

## 💰 Cost Considerations

### Training Costs (Approximate)
- **Instance Type**: ml.m5.large ($0.115/hour)
- **Training Duration**: 10-15 minutes
- **Cost per Training**: ~$0.03-0.05

### Inference Costs
- **Real-time Endpoint**: ml.t2.medium ($0.056/hour)
- **Batch Transform**: ml.m5.large ($0.115/hour, pay per use)
- **Serverless Inference**: Pay per request (recommended for low traffic)

### Cost Optimization Tips
1. **Delete endpoints** when not actively using
2. **Use batch prediction** for bulk inference
3. **Consider serverless inference** for sporadic usage
4. **Monitor CloudWatch metrics** for rightsizing

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. IAM Permission Errors
**Error**: `UnauthorizedOperation: You are not authorized to perform this operation`

**Solution**:
- Verify SageMaker execution role has correct permissions
- Check IAM policy document for missing permissions
- Ensure S3 bucket ARNs match your actual resources

#### 2. S3 Access Issues
**Error**: `NoSuchBucket: The specified bucket does not exist`

**Solution**:
- Verify bucket name and region
- Check S3 bucket permissions
- Update bucket ARNs in IAM policy

#### 3. Model Training Failures
**Error**: `Algorithm error: Training job failed`

**Solution**:
- Check CloudWatch logs for detailed error messages
- Verify training data format and location
- Ensure training script has no syntax errors

#### 4. Endpoint Deployment Issues
**Error**: `Failed to deploy model`

**Solution**:
- Check inference script for errors
- Verify model artifacts are properly saved
- Review endpoint configuration parameters

#### 5. MLflow Registry Issues
**Error**: `INVALID_PARAMETER_VALUE: Registered model must satisfy pattern`

**Solution**:
- Use only alphanumeric characters and hyphens in model names
- Avoid underscores and special characters
- Follow pattern: `^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,56}$`

### Debug Commands

```python
# Check training job logs
estimator.logs()

# List active endpoints
import boto3
sm_client = boto3.client('sagemaker')
sm_client.list_endpoints()

# MLflow experiments
import mlflow
mlflow.search_experiments()
```

### Getting Help

1. **Check CloudWatch Logs** for detailed error messages
2. **Review SageMaker Console** for job status and logs
3. **Validate IAM Permissions** using AWS IAM Policy Simulator
4. **Test locally first** before deploying to SageMaker

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup
```bash
git clone https://github.com/ashishpal2702/customer-churn-aws-ml.git
cd customer-churn-aws-ml
pip install -r requirements.txt
pip install -e .
```

### Contribution Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass and code follows style guidelines
5. Update documentation as needed
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write tests for new functionality

## 📞 Support & Resources

### Documentation
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Community
- Open issues on GitHub for bugs and feature requests
- Join [AWS SageMaker Community](https://forums.aws.amazon.com/forum.jspa?forumID=285)
- Follow [MLflow Community](https://github.com/mlflow/mlflow/discussions)

### Training Resources
- [AWS Machine Learning University](https://aws.amazon.com/machine-learning/mlu/)
- [SageMaker Examples Repository](https://github.com/aws/amazon-sagemaker-examples)
- [MLflow Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AWS SageMaker team for excellent documentation and examples
- MLflow community for the fantastic experiment tracking platform
- Scikit-learn contributors for the robust ML library
- Open source community for continuous innovation

---

**⚠️ Important**: Remember to clean up AWS resources when done to avoid unnecessary charges!

**📊 Project Status**: Active development - contributions welcome!

**🔄 Last Updated**: September 2025
