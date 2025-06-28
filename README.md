# CS611 Assignment 2: Machine Learning Pipeline for Loan Default Prediction

## Project Overview

This project implements an end-to-end Machine Learning pipeline for predicting loan defaults at a financial institution. The pipeline includes data preprocessing, model training, inference, monitoring, and visualization capabilities, all orchestrated through Apache Airflow.

## Architecture

### Data Pipeline (Bronze-Silver-Gold Architecture)
- **Bronze Layer**: Raw data ingestion and basic validation
- **Silver Layer**: Data cleaning, feature engineering, and business logic
- **Gold Layer**: Aggregated features and ML-ready datasets

### ML Pipeline Components
- **Model Training**: LightGBM, XGBoost, and CatBoost models
- **Model Selection**: Automated best model selection based on performance metrics
- **Model Registry**: MLflow-based model versioning and tracking
- **Inference Pipeline**: Batch prediction generation
- **Monitoring**: Performance tracking and drift detection
- **Visualization**: Comprehensive metrics dashboard

## Technology Stack

### Core Technologies
- **Apache Airflow 2.8.1**: Workflow orchestration and scheduling
- **Apache Spark 3.4.1**: Large-scale data processing
- **PostgreSQL**: Metadata storage and metrics database
- **MLflow**: Model registry and experiment tracking

### Machine Learning Libraries
- **LightGBM 4.0.0**: Gradient boosting framework
- **XGBoost**: Extreme gradient boosting
- **CatBoost 1.2.0**: Categorical boosting
- **Scikit-learn**: Traditional ML algorithms and metrics

### Monitoring and Visualization
- **Grafana**: Metrics visualization dashboard
- **Matplotlib/Seaborn**: Custom metric plots
- **Plotly**: Interactive visualizations

## Project Structure

```
cs611_MLE_A2/
├── dags/                          # Airflow DAG definitions
│   ├── dag.py                     # Main ML pipeline DAG
│   ├── dag_functions.py           # DAG task functions
│   └── data_pipeline.py           # Data processing pipeline
├── scripts/
│   ├── datamart/                  # Data processing scripts
│   └── utils/                     # ML utilities and training scripts
│       ├── LightGBM_training_run.py
│       ├── XGBoost_training_run.py
│       ├── CatBoost_training_run.py
│       ├── model_inference_utils.py
│       ├── metrics_visualization.py
│       └── query_model_performance.py
├── data/                          # Input data files
├── datamart/                      # Processed data storage
├── grafana/                       # Grafana dashboard configurations
├── logs/                          # Application logs
├── docker-compose.yaml            # Multi-container setup
├── Dockerfile                     # Container image definition
├── requirements.txt               # Python dependencies
└── prometheus.yml                 # Prometheus configuration
```

## Data Sources

The pipeline processes the following data sources:
- **lms_loan_daily.csv**: Loan management system daily data
- **features_attributes.csv**: User attribute features
- **features_financials.csv**: Financial behavior features
- **feature_clickstream.csv**: User clickstream behavior data

## License

This project is developed for CS611 Machine Learning Engineering course at Singapore Management University.
