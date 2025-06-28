from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.branch import BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.state import State
from datetime import datetime, timedelta
import os
import json
import sys
import pandas as pd
from pyspark.sql import SparkSession
from typing import List
from airflow.exceptions import AirflowSkipException
from dateutil.relativedelta import relativedelta
from airflow.utils.trigger_rule import TriggerRule

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts/utils'))
from dag_functions import (
    decide_pipeline_path,
    check_retraining_trigger,
    check_static_data_loaded,
    select_best_model_initial,
    register_model_initial,
    evaluate_production_model,
    prepare_training_data_monthly,
    check_data_availability,
    run_model_inference,
    train_lightgbm_monthly,
    train_xgboost_monthly,
    train_catboost_monthly,
    select_best_model_monthly,
    register_model_monthly,
    train_lightgbm_initial,
    train_xgboost_initial,
    train_catboost_initial,
    process_all_bronze_tables,
    process_all_silver_tables,
    process_all_gold_tables
)

class SafeExternalTaskSensor(ExternalTaskSensor):
    def poke(self, context):
        prev_execution_date = context['execution_date'] - relativedelta(months=1)
        dag = context['dag']
        dag_start_date = getattr(dag, 'start_date', None) or (dag.default_args.get('start_date') if hasattr(dag, 'default_args') else None)
        if dag_start_date is None:
            # If we can't determine the start date, always succeed
            return True
        if prev_execution_date < dag_start_date:
            # Mark as success explicitly
            context['ti'].set_state(State.SUCCESS)
            return True
        return super().poke(context)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': True,  # Critical: Each run waits for previous month's run to complete
    'start_date': datetime(2023, 1, 1),
    'end_date': datetime(2024, 12, 31),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'streamlined_data_ml_pipeline',
    default_args=default_args,
    description='Streamlined end-to-end ML lifecycle pipeline for credit scoring',
    schedule_interval='0 6 1 * *',  # Monthly on the 1st at 6 AM
    catchup=True,
    tags=['data-preprocessing', 'credit-scoring', 'ml-lifecycle', 'streamlined']
) as dag:
    """
    Streamlined ML Lifecycle Pipeline
    
    This is a simplified version that processes all data in sequential blocks:
    1. Bronze Layer: Process all bronze tables in one task
    2. Silver Layer: Process all silver tables in one task  
    3. Gold Layer: Process all gold tables in one task
    4. ML Pipeline: Standard ML lifecycle tasks
    """

    # === Start of Pipeline ===
    wait_for_previous_run = SafeExternalTaskSensor(
        task_id='wait_for_previous_run_mleA2',
        external_dag_id='streamlined_data_ml_pipeline',
        external_task_id='end_pipeline_mleA2',
        execution_delta=relativedelta(months=1),
        allowed_states=['success'],
        mode='poke',
        timeout=60 * 60 * 3,
        poke_interval=5,
    )

    start = DummyOperator(task_id='start_pipeline_mleA2')
    start_preprocessing = DummyOperator(task_id='start_preprocessing_mleA2')

    # === Data Source Dependencies ===
    # Check if all required data files exist - separate sensors for each file
    check_lms_file = FileSensor(
        task_id='check_lms_file_mleA2',
        filepath='/opt/airflow/data/lms_loan_daily.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )
    
    check_attributes_file = FileSensor(
        task_id='check_attributes_file_mleA2',
        filepath='/opt/airflow/data/features_attributes.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )
    
    check_financials_file = FileSensor(
        task_id='check_financials_file_mleA2',
        filepath='/opt/airflow/data/features_financials.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )
    
    check_clickstream_file = FileSensor(
        task_id='check_clickstream_file_mleA2',
        filepath='/opt/airflow/data/feature_clickstream.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )

    # === Streamlined Data Processing ===
    
    # Bronze Layer - Process all bronze tables in one task
    bronze_processing = PythonOperator(
        task_id='process_all_bronze_tables_mleA2',
        python_callable=process_all_bronze_tables,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/']
    )

    # Silver Layer - Process all silver tables in one task
    silver_processing = PythonOperator(
        task_id='process_all_silver_tables_mleA2',
        python_callable=process_all_silver_tables,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/']
    )

    # Gold Layer - Process all gold tables in one task
    gold_processing = PythonOperator(
        task_id='process_all_gold_tables_mleA2',
        python_callable=process_all_gold_tables,
        op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/silver/', '/opt/airflow/scripts/datamart/gold/', 30, 6]
    )

    preprocessing_complete = DummyOperator(task_id='preprocessing_complete_mleA2')

    # === ML Pipeline ===
    
    # Gate 1: Decide on the main pipeline path
    decide_pipeline_path_task = BranchPythonOperator(
        task_id='decide_pipeline_path_mleA2',
        python_callable=decide_pipeline_path,
    )

    # Path 1: Skip (for historical runs before 2023)
    skip_run = DummyOperator(task_id='skip_run_mleA2')

    # Path 2: One-Time Initial Training Flow
    run_initial_training_flow = DummyOperator(task_id='run_initial_training_flow_mleA2')
    train_lightgbm_initial = PythonOperator(
        task_id='train_lightgbm_initial_mleA2',
        python_callable=train_lightgbm_initial,
    )
    train_xgboost_initial = PythonOperator(
        task_id='train_xgboost_initial_mleA2',
        python_callable=train_xgboost_initial,
    )
    train_catboost_initial = PythonOperator(
        task_id='train_catboost_initial_mleA2',
        python_callable=train_catboost_initial,
    )
    select_best_model_initial_task = PythonOperator(
        task_id='select_best_model_initial_mleA2',
        python_callable=select_best_model_initial,
    )
    register_model_initial_task = PythonOperator(
        task_id='register_model_initial_mleA2',
        python_callable=register_model_initial,
    )

    # Path 3: Standard Monthly Lifecycle Flow
    run_monthly_lifecycle_flow = DummyOperator(task_id='run_monthly_lifecycle_flow_mleA2')
    evaluate_production_model_task = PythonOperator(
        task_id='evaluate_production_model_mleA2',
        python_callable=evaluate_production_model,
    )
    check_retraining_trigger_task = BranchPythonOperator(
        task_id='check_retraining_trigger_mleA2',
        python_callable=check_retraining_trigger,
    )
    skip_retraining = DummyOperator(task_id='skip_retraining_mleA2')
    trigger_retraining = DummyOperator(task_id='trigger_retraining_mleA2')
    train_lightgbm_monthly_task = PythonOperator(
        task_id='train_lightgbm_monthly_mleA2',
        python_callable=train_lightgbm_monthly,
    )
    train_xgboost_monthly_task = PythonOperator(
        task_id='train_xgboost_monthly_mleA2',
        python_callable=train_xgboost_monthly,
    )
    train_catboost_monthly_task = PythonOperator(
        task_id='train_catboost_monthly_mleA2',
        python_callable=train_catboost_monthly,
    )
    select_best_model_monthly_task = PythonOperator(
        task_id='select_best_model_monthly_mleA2',
        python_callable=select_best_model_monthly,
    )
    register_model_monthly_task = PythonOperator(
        task_id='register_model_monthly_mleA2',
        python_callable=register_model_monthly,
    )
    run_model_inference_task = PythonOperator(
        task_id='run_model_inference_mleA2',
        python_callable=run_model_inference,
        trigger_rule='one_success',
    )

    end = DummyOperator(task_id='end_pipeline_mleA2', trigger_rule='one_success')

    # === Define Dependencies ===
    
    # Data Pipeline Dependencies (Sequential)
    wait_for_previous_run >> start >> start_preprocessing >> [check_lms_file, check_attributes_file, check_financials_file, check_clickstream_file]
    [check_lms_file, check_attributes_file, check_financials_file, check_clickstream_file] >> bronze_processing >> silver_processing >> gold_processing >> preprocessing_complete

    # ML Pipeline Dependencies
    preprocessing_complete >> decide_pipeline_path_task

    # Path 1: Skip
    decide_pipeline_path_task >> skip_run >> end

    # Path 2: Initial Training
    decide_pipeline_path_task >> run_initial_training_flow
    run_initial_training_flow >> train_lightgbm_initial >> train_xgboost_initial >> train_catboost_initial >> select_best_model_initial_task >> register_model_initial_task >> end

    # Path 3: Monthly Lifecycle
    decide_pipeline_path_task >> run_monthly_lifecycle_flow
    run_monthly_lifecycle_flow >> evaluate_production_model_task >> check_retraining_trigger_task

    # Branching after evaluation
    check_retraining_trigger_task >> skip_retraining >> run_model_inference_task
    check_retraining_trigger_task >> trigger_retraining

    # Retraining sub-path
    trigger_retraining >> train_lightgbm_monthly_task >> train_xgboost_monthly_task >> train_catboost_monthly_task >> select_best_model_monthly_task >> register_model_monthly_task
    register_model_monthly_task >> run_model_inference_task

    # Inference is the final step for the monthly path
    run_model_inference_task >> end
