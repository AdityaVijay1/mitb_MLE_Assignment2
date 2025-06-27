import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse
import re

from collections import Counter

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, MapType, BooleanType

############################
# Attributes
############################
def process_df_attributes(df):
    """
    Function to process attributes table
    """
    numeric_regex = r'([-+]?\d*\.?\d+)'
    
    # Extract numeric part from string in 'Age' column
    df = df.withColumn("age", F.regexp_extract(col("age"), numeric_regex, 1))

    # Define column data types
    columns = {
        'customer_id': StringType(),
        'name': StringType(),
        'age': IntegerType(),
        'ssn': StringType(),
        'occupation': StringType(),
        'snapshot_date': DateType()
    }

    # Cast columns to the proper data type
    for column, new_type in columns.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Enforce valid age constraints
    # The oldest person in the world is a little less than 120 years old, so make everything above that invalid
    # Minimum is 0 because some banks allow opening joint accounts for children
    df = df.withColumn(
        "age",
        F.when((col("age") >= 0) & (col("age") <= 120), col("age"))  # keep valid
        .otherwise(None)  # redact invalid
    ) 

    # Enforce valid SSN
    df = df.withColumn(
        "ssn",
        F.regexp_extract(col("ssn"), r'^(\d{3}-\d{2}-\d{4})$', 1)
    )
    df = df.withColumn(
        "ssn",
        F.when(col("ssn") == "", None).otherwise(col("ssn"))
    )

    # Null empty occupation
    df = df.withColumn(
        "occupation",
        F.when(col("occupation") == "_______", None).otherwise(col("occupation"))
    )
    return df

############################
# Clickstream
############################
def process_df_clickstream(df):
    """
    Function to process clickstream table
    """
    # Define column data types
    columns = {
        **{f'fe_{i}': IntegerType() for i in range(1, 21)},
        'customer_id': StringType(),
        'snapshot_date': DateType()
    }

    # Cast columns to the proper data type
    for column, new_type in columns.items():
        df = df.withColumn(column, col(column).cast(new_type))
    return df

############################
# Financials
############################
def split_loan_type(loan_type):
    """
    Utility function to split loan type into frequency table
    """
    if not isinstance(loan_type, str):
        return {}
    
    loans_list = loan_type.replace(' and ', ',').split(',')

    cleaned = [item.strip().replace(' ', '_').lower() for item in loans_list if item.strip() != '']

    return dict(Counter(cleaned))

def process_df_financials(df, silver_db, snapshot_date_str):
    """
    Function to process financials table
    """
    numeric_regex = r'([-+]?\d*\.?\d+)'
    
    columns = {
        'annual_income': FloatType(),
        'monthly_inhand_salary': FloatType(),
        'num_bank_accounts': IntegerType(),
        'num_credit_card': IntegerType(),
        'interest_rate': IntegerType(),
        'num_of_loan': IntegerType(),
        'delay_from_due_date': IntegerType(),
        'num_of_delayed_payment': IntegerType(),
        'changed_credit_limit': FloatType(),
        'num_credit_inquiries': FloatType(),
        'outstanding_debt': FloatType(),
        'credit_utilization_ratio': FloatType(),
        'total_emi_per_month': FloatType(),
        'amount_invested_monthly': FloatType(),
        'monthly_balance': FloatType()
    }

    # Cast columns to the proper data type
    for col_name, dtype in columns.items():
        df = df.withColumn(col_name, F.regexp_extract(col(col_name), numeric_regex, 1))
        df = df.withColumn(col_name, col(col_name).cast(dtype))

    # Split credit history age
    df = df.withColumn("credit_history_age_year",
                        F.regexp_extract(col('credit_history_age'), r'(\d+)\s+Year', 1))
    df = df.withColumn("credit_history_age_year", col("credit_history_age_year").cast(IntegerType()))
    df = df.withColumn("credit_history_age_month",
                        F.regexp_extract(col('credit_history_age'), r'(\d+)\s+Month', 1))
    df = df.withColumn("credit_history_age_month", col("credit_history_age_month").cast(IntegerType()))

    # Remove negative values from columns that should not have it
    for column_name in ['num_of_loan', 'delay_from_due_date', 'num_of_delayed_payment']:
        df = df.withColumn(
            column_name,
            F.when(col(column_name) >= 0, col(column_name))  # keep valid
            .otherwise(None)  # redact invalid
        ) 
    
    # Clip outliers to 90th percentile
    for column_name in ['num_bank_accounts', 'num_credit_card', 'interest_rate', 'num_of_loan', 'num_of_delayed_payment']:
        percentile_value = df.approxQuantile(column_name, [0.97], 0.01)[0]
        df = df.withColumn(
            column_name,
            F.when(col(column_name) > percentile_value, percentile_value)
            .otherwise(col(column_name))
        )

    # Split payment behaviour
    payment_behaviour_regex = r'(Low|High)_spent_(Small|Medium|Large)_value'
    df = df.withColumn(
        'payment_behaviour_spent',
        F.regexp_extract(col('payment_behaviour'), payment_behaviour_regex, 1)
    )
    df = df.withColumn(
        'payment_behaviour_spent',
        F.when(col('payment_behaviour_spent') != '', col('payment_behaviour_spent'))
        .otherwise(None)
    )
    df = df.withColumn(
        'payment_behaviour_value',
        F.regexp_extract(col('payment_behaviour'), payment_behaviour_regex, 2)
    )
    df = df.withColumn(
        'payment_behaviour_value',
        F.when(col('payment_behaviour_value') != '', col('payment_behaviour_value'))
        .otherwise(None)
    )

    # Null empty credit_mix
    df = df.withColumn(
        "credit_mix",
        F.when(col("credit_mix") == "_", None).otherwise(col("credit_mix"))
    )
    
    ######################################
    # Split loan type into its own table
    ######################################
    df_loan_type = df.select('customer_id', 'snapshot_date', 'type_of_loan')

    # Register helper function as a udf
    split_loan_type_udf = F.udf(split_loan_type, MapType(StringType(), IntegerType()))

    # Apply UDF to column
    df_loan_type = df_loan_type.withColumn("loan_type_counts", split_loan_type_udf(col("Type_of_Loan")))
    all_keys = (
        df_loan_type.select("loan_type_counts")
        .rdd.flatMap(lambda row: row["loan_type_counts"].keys() if row["loan_type_counts"] else [])
        .distinct()
        .collect()
    )

    # Create individual columns for each loan type
    for key in all_keys:
        df_loan_type = df_loan_type.withColumn(
            key,
            F.coalesce(col("loan_type_counts").getItem(key), F.lit(0))
        )

    # Drop intermedate columns
    df_loan_type = df_loan_type.drop("loan_type_counts")
    
    # Save new table
    partition_name = 'silver_loan_type_mthly_' + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_db, 'loan_type', partition_name)
    df_loan_type.write.mode("overwrite").parquet(filepath)

    return df.drop('payment_behaviour', 'type_of_loan')

############################
# LMS
############################
def process_df_lms(df):
    """
    Function to process LMS table
    """
    column_type_map = {
        "loan_id": StringType(),
        "customer_id": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    # Cast columns to proper data type
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Add "month on book" column
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # Add "days past due" column
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
    
    return df

############################
# Pipeline
############################
def process_silver_table(table_name, bronze_db, silver_db, snapshot_date_str):
    """
    Function to process silver table
    """
    # Get the absolute path to the current script to send to Spark workers
    script_path = os.path.abspath(__file__)

    spark = (pyspark.sql.SparkSession.builder
        .appName(f'silver_table_{table_name}')
        .master('local[*]')
        # Distribute the current script to all worker nodes so they can see the UDFs
        .config("spark.submit.pyFiles", script_path)
        .getOrCreate()
    )

    try:
        # connect to bronze table
        if table_name == "attributes":
            # Static data - read from static file
            filepath = os.path.join(bronze_db, table_name, 'bronze_attr_static.csv')
            if not os.path.exists(filepath):
                print(f"Static attributes file not found: {filepath}")
                return None
            df = spark.read.csv(filepath, header=True, inferSchema=True)
        elif table_name == 'clickstream':
            # Monthly data - read from monthly partition
            partition_name = 'bronze_clks_mthly_' + snapshot_date_str.replace('-','_') + '.csv'
            filepath = os.path.join(bronze_db, table_name, partition_name)
            df = spark.read.csv(filepath, header=True, inferSchema=True)
        elif table_name == "financials":
            # Static data - read from static file
            filepath = os.path.join(bronze_db, table_name, 'bronze_fin_static.csv')
            if not os.path.exists(filepath):
                print(f"Static financials file not found: {filepath}")
                return None
            df = spark.read.csv(filepath, header=True, inferSchema=True)
        elif table_name == "lms":
            # Monthly data - read from monthly partition
            partition_name = 'bronze_loan_daily_' + snapshot_date_str.replace('-','_') + '.csv'
            filepath = os.path.join(bronze_db, table_name, partition_name)
            df = spark.read.csv(filepath, header=True, inferSchema=True)
        else:
            print("Table does not exist!")
            return None

        # Change all column names to be lowercase
        df = df.toDF(*[c.lower() for c in df.columns])

        if table_name == "attributes":
            df = process_df_attributes(df)
        elif table_name == 'clickstream':
            df = process_df_clickstream(df)
        elif table_name == "financials":
            df = process_df_financials(df, silver_db, snapshot_date_str)   
        elif table_name == "lms":
            df = process_df_lms(df)
        else:
            raise ValueError("Table does not exist!")

        # Save silver table
        if table_name in ["attributes", "financials"]:
            # Static data - save as static file (only once)
            partition_name = 'silver_' + table_name + '_static.parquet'
            filepath = os.path.join(silver_db, table_name, partition_name)
            # Check if static file already exists
            if os.path.exists(filepath):
                print(f"Static silver {table_name} file already exists: {filepath}")
                return None
        else:
            # Monthly data - save as monthly partition
            partition_name = 'silver_' + table_name + '_mthly_' + snapshot_date_str.replace('-','_') + '.parquet'
            filepath = os.path.join(silver_db, table_name, partition_name)
        
        df.write.mode("overwrite").parquet(filepath)
    
    finally:
        spark.stop()

def process_silver_loan_table(snapshot_date_str, bronze_lms_directory, silver_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_lms_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_clickstream_table(snapshot_date_str, bronze_clks_directory, silver_clks_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_clks_mthly_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clks_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # clean data: enforce schema / data type
    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Remove negative numbers in fe_1 to fe_20
    feature_cols = [f"fe_{i}" for i in range(1, 21)]
    for feature in feature_cols:
        df = df.withColumn(feature, F.when(F.col(feature) < 0, 0).otherwise(F.col(feature)))
    
    # save silver table - IRL connect to database to write
    partition_name = "silver_clks_mthly_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clks_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_attributes_table(snapshot_date_str, bronze_attr_directory, silver_attr_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_attr_mthly_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attr_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean Name field
    def clean_name(name):
        if name is None:
            return None
        cleaned = re.sub(r"[^A-Za-z .'-]", '', name)  # remove invalid characters
        return cleaned.strip()  # strip whitespace
    
    clean_name_udf = F.udf(clean_name, StringType())
    
    # Apply my function to the Name column
    df = df.withColumn("Name", clean_name_udf(F.col("Name")))

    # Clean Age field
    # Remove non-digit characters from Age, then convert to int and remove negative and large values
    df = df.withColumn("Age", F.regexp_replace(F.col("Age").cast(StringType()), r"\D", ""))
    df = df.withColumn("Age", F.col("Age").cast(IntegerType()))
    df = df.withColumn("Age", F.when((F.col("Age") >= 0) & (F.col("Age") <= 100), F.col("Age")).otherwise(None))

    # Clean SSN
    valid_ssn_pattern = r"^\d{3}-\d{2}-\d{4}$"
    df = df.withColumn("SSN", F.when(F.col("SSN").rlike(valid_ssn_pattern), F.col("SSN")).otherwise(None))

    # CLean Occupation
    df = df.withColumn("Occupation", F.when(F.col("Occupation") == '_______', None).otherwise(F.col("Occupation")))
    df = df.withColumn("Occupation", F.when(F.col("Occupation") == 'Media_Manager', 'Media Manager').otherwise(F.col("Occupation")))

    # save silver table - IRL connect to database to write
    partition_name = "silver_attr_mthly_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attr_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_financials_table(snapshot_date_str, bronze_fin_directory, silver_fin_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_fin_mthly_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_fin_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Decimal currency columns (3dp based on ISO standard)
    cols_decimal3 = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
    ]
    for c in cols_decimal3:
        df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), r"[^\d.]", ""))
        df = df.withColumn(c, F.round(F.col(c).cast("double"), 3))
        df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))

    # Integer columns
    cols_integer = [
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan',
        'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Num_Credit_Inquiries', 'Interest_Rate'
    ]
    for c in cols_integer:
        df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), r"[^\d]", ""))
        df = df.withColumn(c, F.col(c).cast("int"))
        df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))

    # Convert Credit_History_Age to months
    def convert_to_months(s):
        if s is None:
            return None
        m = re.match(r"(\d+) Years and (\d+) Months", s)
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
        return None

    convert_udf = F.udf(convert_to_months, IntegerType())
    df = df.withColumn("Credit_History_Age", convert_udf(F.col("Credit_History_Age")))

    # Float columns
    cols_float = ['Changed_Credit_Limit', 'Credit_Utilization_Ratio']
    for c in cols_float:
        df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), r"[^\d.]", ""))
        df = df.withColumn(c, F.col(c).cast("double"))

    # Remove negative credit utilization ratios
    df = df.withColumn("Credit_Utilization_Ratio", F.when(F.col("Credit_Utilization_Ratio") < 0, None).otherwise(F.col("Credit_Utilization_Ratio")))

    # Cap outliers
    outlier_caps = {
        'Num_Bank_Accounts': 10,
        'Num_Credit_Card': 10,
        'Interest_Rate': 34,
        'Num_of_Loan': 9,
        'Num_of_Delayed_Payment': 47,
        'Num_Credit_Inquiries': 26
    }
    for colname, cap in outlier_caps.items():
        df = df.withColumn(colname, F.when(F.col(colname) > cap, cap).otherwise(F.col(colname)))

    # Encode Credit_Mix
    credit_mix_mapping = ['Bad', 'Standard', 'Good']
    # Remove invalid values first
    df = df.withColumn("Credit_Mix", F.when(F.col("Credit_Mix").isin(credit_mix_mapping), F.col("Credit_Mix")).otherwise(None))
    mapping_expr = F.when(F.col("Credit_Mix") == "Bad", 0)\
                   .when(F.col("Credit_Mix") == "Standard", 1)\
                   .when(F.col("Credit_Mix") == "Good", 2)
    df = df.withColumn("Credit_Mix", mapping_expr.cast("int"))

    # Encode Payment_of_Min_Amount
    df = df.withColumn("Payment_of_Min_Amount", 
        F.when(F.col("Payment_of_Min_Amount") == "Yes", True)
        .when(F.col("Payment_of_Min_Amount") == "No", False)
        .otherwise(None))

    # Encode Payment_Behaviour
    valid_pb_enums = [
        'High_spent_Small_value_payments', 'High_spent_Medium_value_payments',
        'High_spent_Large_value_payments', 'Low_spent_Small_value_payments',
        'Low_spent_Medium_value_payments', 'Low_spent_Large_value_payments'
    ]

    # Remove invalid values first
    df = df.withColumn("Payment_Behaviour",
        F.when(F.col("Payment_Behaviour").isin(valid_pb_enums), F.col("Payment_Behaviour")).otherwise(None))

    mapping_expr_pb = F.when(F.col("Payment_Behaviour") == "Low_spent_Small_value_payments", 0)\
        .when(F.col("Payment_Behaviour") == "Low_spent_Medium_value_payments", 1)\
        .when(F.col("Payment_Behaviour") == "Low_spent_Large_value_payments", 2)\
        .when(F.col("Payment_Behaviour") == "High_spent_Small_value_payments", 3)\
        .when(F.col("Payment_Behaviour") == "High_spent_Medium_value_payments", 4)\
        .when(F.col("Payment_Behaviour") == "High_spent_Large_value_payments", 5)
    df = df.withColumn("Payment_Behaviour", mapping_expr_pb.cast("int"))

    # Feature Engineering
    df = df.withColumn("Num_Fin_Pdts", F.col("Num_Bank_Accounts") + F.col("Num_Credit_Card") + F.col("Num_of_Loan"))   
    df = df.withColumn("Loans_per_Credit_Item", F.col("Num_of_Loan") / (F.col("Num_Bank_Accounts") + F.col("Num_Credit_Card") + F.lit(1)))
    df = df.withColumn("Debt_to_Salary", F.col("Outstanding_Debt") / (F.col("Monthly_Inhand_Salary") + F.lit(1)))
    df = df.withColumn("EMI_to_Salary", F.col("Total_EMI_per_month") / (F.col("Monthly_Inhand_Salary") + F.lit(1)))
    df = df.withColumn("Repayment_Ability", F.col("Monthly_Inhand_Salary") - F.col("Total_EMI_per_month"))
    df = df.withColumn("Loan_Extent", F.col("Delay_from_due_date") * F.col("Num_of_Loan"))
    
    # save silver table - IRL connect to database to write
    partition_name = "silver_fin_mthly_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_fin_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df