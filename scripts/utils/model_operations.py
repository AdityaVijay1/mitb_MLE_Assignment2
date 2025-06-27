import os
import pandas as pd
from pyspark.sql import SparkSession
from typing import List

def load_data_for_training(spark: SparkSession, feature_store_path: str, label_store_path: str, months: List[str]) -> pd.DataFrame:
    """
    Loads feature and label data for a given list of month partitions and joins them.
    
    Args:
        spark: SparkSession instance
        feature_store_path: Path to the feature store directory
        label_store_path: Path to the label store directory
        months: List of month strings in format 'YYYY_MM_DD'
    
    Returns:
        pandas.DataFrame: Joined feature and label data
    """
    feature_dfs = []
    label_dfs = []
    
    for month in months:
        # Load feature data
        feature_path = os.path.join(feature_store_path, f'gold_feature_store_{month}.parquet')
        if os.path.exists(feature_path):
            feature_df = spark.read.parquet(feature_path)
            feature_dfs.append(feature_df)
            print(f"Loaded features from: {feature_path}")
        else:
            print(f"Warning: Feature file not found: {feature_path}")
        
        # Load label data
        label_path = os.path.join(label_store_path, f'gold_label_store_{month}.parquet')
        if os.path.exists(label_path):
            label_df = spark.read.parquet(label_path)
            label_dfs.append(label_df)
            print(f"Loaded labels from: {label_path}")
        else:
            print(f"Warning: Label file not found: {label_path}")
    
    if not feature_dfs or not label_dfs:
        raise ValueError("No feature or label data found for the specified months")
    
    # Union all feature dataframes
    features_df = feature_dfs[0]
    for df in feature_dfs[1:]:
        features_df = features_df.unionByName(df)
    
    # Union all label dataframes
    labels_df = label_dfs[0]
    for df in label_dfs[1:]:
        labels_df = labels_df.unionByName(df)
    
    # Join features and labels on loan_id
    final_df = features_df.join(labels_df, "loan_id", "inner")
    
    print(f"Final dataset shape: {final_df.count()} rows, {len(final_df.columns)} columns")
    
    return final_df.toPandas() 