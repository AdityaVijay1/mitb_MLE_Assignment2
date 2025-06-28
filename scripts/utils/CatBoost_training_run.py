import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import catboost as cb
import mlflow
import mlflow.sklearn
import json
from datetime import datetime, timedelta
import os
import sys
import time
from typing import List

# Import from the same directory since we're now in utils/
from model_operations import load_data_for_training

# Import from the same directory since we're now in utils/
from model_inference_utils import calculate_comprehensive_metrics, save_model_metrics_to_postgres

def get_date_range_for_training(end_date: datetime, num_months: int) -> List[str]:
    """
    Calculates the list of monthly partition strings for data loading.
    """
    months = []
    for i in range(num_months):
        partition_date = end_date - timedelta(days=30*i)  # Approximate month
        months.append(partition_date.strftime('%Y_%m_%d'))
    return sorted(months)

def load_and_prepare_data():
    """
    Load and prepare the data for training.
    """
    start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting data loading...")
    
    # Initialize Spark
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Spark...")
    spark = SparkSession.builder \
        .appName("ModelTraining") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # Configuration
    SNAPSHOT_DATE = datetime(2024, 5, 1)  # Use May 2024 as end date for initial training
    TRAINING_MONTHS = 12  # Use 12 months of data
    FEATURE_STORE_PATH = "/opt/airflow/scripts/datamart/gold/feature_store"
    LABEL_STORE_PATH = "/opt/airflow/scripts/datamart/gold/label_store"
    
    # Get training months
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating training months...")
    training_months = get_date_range_for_training(SNAPSHOT_DATE, TRAINING_MONTHS)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data for {len(training_months)} months, from {training_months[0]} to {training_months[-1]}")
    
    # Load data using the chunked approach
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Calling load_data_for_training...")
    load_start = time.time()
    
    try:
        full_df = load_data_for_training(spark, FEATURE_STORE_PATH, LABEL_STORE_PATH, training_months)
        load_time = time.time() - load_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data loading completed in {load_time:.2f} seconds")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully loaded {full_df.shape[0]} records.")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR during data loading: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] This might be due to:")
        print(f"  - Missing parquet files")
        print(f"  - Memory issues with .toPandas()")
        print(f"  - Spark configuration problems")
        raise
    
    # Prepare features and target
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Preparing features and target...")
    # Drop columns that cause issues with CatBoost (date columns and other non-numeric)
    features_to_drop = [
        'id',  # UNIQUE_ID_COLUMN
        'grade',  # TARGET_COLUMN
        'snapshot_date', 
        'earliest_cr_date',
        'snapshot_month',
        'earliest_cr_month',
        'months_since_earliest_cr_line'
    ]
    
    # Drop the problematic columns
    full_df_cleaned = full_df.drop(columns=features_to_drop, errors='ignore')
    
    # Get remaining feature columns
    feature_columns = [col for col in full_df_cleaned.columns if col != 'grade_encoded']
    X = full_df_cleaned[feature_columns]
    y = full_df['grade']
    
    # Create grade mapping
    grade_mapping = {grade: idx for idx, grade in enumerate(sorted(y.unique()))}
    y_encoded = y.map(grade_mapping)
    
    total_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Data preparation completed in {total_time:.2f} seconds")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Data shape: {X.shape}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Grade distribution: {y.value_counts().to_dict()}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Grade mapping: {grade_mapping}")
    
    return X, y_encoded, grade_mapping

def main():
    """
    Main training function using fixed, fast hyperparameters (no tuning).
    """
    total_start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting ULTRA-FAST CatBoost training...")

    # Load data
    X, y, grade_mapping = load_and_prepare_data()

    # Split data
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training set size: {X_train.shape[0]}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Test set size: {X_test.shape[0]}")

    # Define a single, fixed set of fast hyperparameters. NO SEARCHING.
    best_params = {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU'
    }
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Using fixed parameters: {best_params}")

    # Train final model on full training data
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training final model...")
    final_start = time.time()
    final_model = cb.CatBoostClassifier(**best_params)
    
    # Use early stopping for final training
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    final_model.fit(
        X_train_final, y_train_final,
        eval_set=(X_val_final, y_val_final),
        early_stopping_rounds=15,
        verbose=False
    )
    final_time = time.time() - final_start
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Final model training completed in {final_time:.2f} seconds")
    
    # Evaluate on test set
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluating model...")
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)
    
    # Calculate comprehensive metrics
    comprehensive_metrics = calculate_comprehensive_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba
    )
    
    macro_f1 = comprehensive_metrics['macro_f1']
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Final Test Performance:")
    print(f"Accuracy: {comprehensive_metrics['accuracy']:.4f}")
    print(f"Precision: {comprehensive_metrics['precision']:.4f}")
    print(f"Recall: {comprehensive_metrics['recall']:.4f}")
    print(f"AUC: {comprehensive_metrics['auc']:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Log to MLflow
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Logging to MLflow...")
    try:
        # Use the service name 'mlflow' instead of 'localhost' for Docker networking
        mlflow.set_tracking_uri("http://mlflow:5000")

        # Set the experiment for this run. Will be created if it doesn't exist.
        mlflow.set_experiment("loandefault_mle2")

        # Set a descriptive run name for easy identification
        run_name = f"catboost_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            # Log parameters from our fixed dictionary
            mlflow.log_params(best_params)
            mlflow.log_param("num_classes", len(grade_mapping))
            mlflow.log_param("training_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("feature_count", X.shape[1])
            mlflow.log_param("model_type", "CatBoost")
            
            # Log all comprehensive metrics
            for metric_name, metric_value in comprehensive_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log the model
            mlflow.sklearn.log_model(final_model, "model")
            
            # Log feature names for inference
            feature_names = list(X.columns)
            mlflow.log_param("feature_names", feature_names)
            
            # Log grade mapping for inference
            mlflow.log_param("grade_mapping", grade_mapping)
            
            # Get the run ID for later use
            run_id = mlflow.active_run().info.run_id
            print(f"[{datetime.now().strftime('%H:%M:%S')}] MLflow run ID: {run_id}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Macro F1 Score: {macro_f1:.4f}")
            
            # Save run ID and metrics to a file for Airflow to pick up
            run_info = {
                'run_id': run_id,
                'macro_f1': macro_f1,
                'model_type': 'CatBoost',
                'timestamp': datetime.now().isoformat()
            }
            
            with open('/tmp/catboost_run_info.json', 'w') as f:
                json.dump(run_info, f)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Run info saved to /tmp/catboost_run_info.json")
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR during MLflow logging: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] This might be due to:")
        print(f"  - MLflow server not running")
        print(f"  - Network connectivity issues")
        print(f"  - MLflow configuration problems")
        raise
    
    total_time = time.time() - total_start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Total CatBoost training time: {total_time:.2f} seconds")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] CatBoost training completed successfully!")
    
    return f"CatBoost training completed. Run ID: {run_id}, Macro F1: {macro_f1:.4f}"

if __name__ == "__main__":
    main() 