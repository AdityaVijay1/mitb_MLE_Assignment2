from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts/utils'))
from dag_functions import (
    process_all_bronze_tables,
    process_all_silver_tables,
    process_all_gold_tables
)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': True,
    'start_date': datetime(2023, 1, 1),
    'end_date': datetime(2024, 12, 31),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'streamlined_data_pipeline',
    default_args=default_args,
    description='Streamlined data preprocessing pipeline for credit scoring',
    schedule_interval='0 6 1 * *',
    catchup=True,
    tags=['data-preprocessing', 'credit-scoring', 'streamlined']
) as dag:
    """
    Streamlined Data Preprocessing Pipeline
    
    This is a simplified version that processes all data in sequential blocks:
    1. Bronze Layer: Process all bronze tables in one task
    2. Silver Layer: Process all silver tables in one task  
    3. Gold Layer: Process all gold tables in one task
    """

    start = DummyOperator(task_id='start_pipeline_mleA2')

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
    end = DummyOperator(task_id='end_pipeline_mleA2')

    # Define sequential dependencies
    start >> [check_lms_file, check_attributes_file, check_financials_file, check_clickstream_file] >> bronze_processing >> silver_processing >> gold_processing >> preprocessing_complete >> end 