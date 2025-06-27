import os
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType

def process_gold_table(snapshot_date_str, silver_dir, gold_dir, spark, dpd, mob):
    """
    Wrapper for gold table processing. Reads silver outputs, uses lms as anchor, joins static and monthly features, writes to gold feature and label store.
    Args:
        snapshot_date_str: str, e.g. '2023-01-01'
        silver_dir: str, path to silver directory
        gold_dir: str, path to gold directory
        spark: SparkSession
        dpd: int, days past due threshold
        mob: int, month on book
    """
    def standardize_columns(df):
        return df.toDF(*[c.lower() for c in df.columns])

    # LMS anchor
    lms_path = os.path.join(silver_dir, 'lms', f'silver_lms_mthly_{snapshot_date_str.replace("-", "_")}.parquet')
    df_lms = spark.read.parquet(lms_path)
    df_lms = standardize_columns(df_lms)
    
    # Check available mob values before filtering
    mob_counts = df_lms.groupBy('mob').count().orderBy('mob')
    print(f"Available MOB values in {snapshot_date_str}:")
    mob_counts.show()
    
    # Filter by mob
    df_lms_filtered = df_lms.filter(F.col("mob") == mob)
    filtered_count = df_lms_filtered.count()
    print(f"Records with mob={mob}: {filtered_count}")
    
    if filtered_count == 0:
        print(f"Warning: No records found with mob={mob} for date {snapshot_date_str}")
        print("This is expected for early months when loans haven't reached the required months on book yet.")
        print("Gold tables will be created with 0 entries.")

    # Label creation
    df_label = df_lms_filtered.withColumn("label", F.when(F.col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType())) \
        .withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType())) \
        .select("loan_id", "customer_id", "label", "label_def", "snapshot_date")
    label_path = os.path.join(gold_dir, 'label_store', f'gold_label_store_{snapshot_date_str.replace("-", "_")}.parquet')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    
    df_label.write.mode("overwrite").parquet(label_path)
    print(f"Saved {label_path}. No of entries: {df_label.count()}.")

    # Feature store
    # Join monthly attributes
    attr_path = os.path.join(silver_dir, 'attributes', f'silver_attributes_mthly_{snapshot_date_str.replace("-", "_")}.parquet')
    df_attr = spark.read.parquet(attr_path)
    df_attr = standardize_columns(df_attr).drop("snapshot_date")
    # Join monthly financials
    fin_path = os.path.join(silver_dir, 'financials', f'silver_financials_mthly_{snapshot_date_str.replace("-", "_")}.parquet')
    df_fin = spark.read.parquet(fin_path)
    df_fin = standardize_columns(df_fin).drop("snapshot_date")
    # Join clickstream
    clickstream_path = os.path.join(silver_dir, 'clickstream', f'silver_clickstream_mthly_{snapshot_date_str.replace("-", "_")}.parquet')
    df_click = spark.read.parquet(clickstream_path)
    df_click = standardize_columns(df_click)
    # Join all

    df_lms = df_lms_filtered.select('loan_id', 'customer_id')
    df = df_lms \
        .join(df_attr, on="customer_id", how="left") \
        .join(df_fin, on="customer_id", how="left") \
        .join(df_click.drop("snapshot_date"), on="customer_id", how="left")
    
    # Drop unnecessary columns from feature store
    df = df.drop("credit_mix", "payment_behaviour", "customer_id", "occupation", "label", "label_def", "loan_start_date", "installment_num", "first_missed_date")

    feature_path = os.path.join(gold_dir, 'feature_store', f'gold_feature_store_{snapshot_date_str.replace("-", "_")}.parquet')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    
    df.write.mode("overwrite").parquet(feature_path)
    print(f"Saved {feature_path}. No of entries: {df.count()}.")
    return df 