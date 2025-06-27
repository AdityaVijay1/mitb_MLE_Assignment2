import os
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder.appName('RowCount').getOrCreate()

# Check silver tables
print("=== SILVER TABLES ROW COUNTS ===")
silver_tables = [
    ('lms', 'scripts/datamart/silver/lms/silver_lms_mthly_2023_01_01.parquet'),
    ('clickstream', 'scripts/datamart/silver/clickstream/silver_clickstream_mthly_2023_01_01.parquet'),
    ('attributes', 'scripts/datamart/silver/attributes/silver_attributes_mthly_2023_01_01.parquet'),
    ('financials', 'scripts/datamart/silver/financials/silver_financials_mthly_2023_01_01.parquet')
]

for table_name, path in silver_tables:
    if os.path.exists(path):
        count = spark.read.parquet(path).count()
        print(f"{table_name}: {count} rows")
    else:
        print(f"{table_name}: File not found")

print("\n=== GOLD TABLES ROW COUNTS ===")
gold_tables = [
    ('label_store', 'scripts/datamart/gold/label_store/gold_label_store_2023_01_01.parquet'),
    ('feature_store', 'scripts/datamart/gold/feature_store/gold_feature_store_2023_01_01.parquet')
]

for table_name, path in gold_tables:
    if os.path.exists(path):
        count = spark.read.parquet(path).count()
        print(f"{table_name}: {count} rows")
    else:
        print(f"{table_name}: File not found")

# Check LMS data filtering
print("\n=== LMS DATA ANALYSIS ===")
lms_path = 'scripts/datamart/silver/lms/silver_lms_mthly_2023_01_01.parquet'
if os.path.exists(lms_path):
    lms_df = spark.read.parquet(lms_path)
    print(f"Total LMS rows: {lms_df.count()}")
    
    # Check mob values
    mob_counts = lms_df.groupBy('mob').count().orderBy('mob')
    print("MOB distribution:")
    mob_counts.show()
    
    # Check if any rows have mob=1 (default filter)
    mob_1_count = lms_df.filter(lms_df.mob == 1).count()
    print(f"Rows with mob=1: {mob_1_count}")

spark.stop() 