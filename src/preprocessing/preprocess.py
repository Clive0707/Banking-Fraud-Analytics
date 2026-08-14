import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import pyarrow as pa
import pyarrow.parquet as pq

# Sanitize PATH and set JAVA_HOME for PySpark local execution on Windows
if os.path.exists(r"C:\Program Files\Java\jdk-22"):
    os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-22"

clean_paths = []
for p in os.environ.get("PATH", "").split(os.pathsep):
    p_clean = p.replace('"', '').strip()
    if p_clean and "msi" not in p_clean.lower():
        clean_paths.append(p_clean)
os.environ["PATH"] = os.pathsep.join(clean_paths)

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_spark_session():
    """
    Initializes a local PySpark session in local[*] mode without Hadoop/HDFS dependencies.
    """
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("BankingFraudAnalytics-PySpark-15M")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

def run_pyspark_preprocessing(raw_data_path, processed_data_dir):
    """
    Main PySpark processing pipeline for 15,000,000 transaction records.
    """
    processed_path = Path(processed_data_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    spark = get_spark_session()
    logger.info("Spark session initialized in local[*] mode.")
    
    # 1. Schema Definition
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("customer_id", IntegerType(), True),
        StructField("transaction_date", StringType(), True),
        StructField("transaction_time", StringType(), True),
        StructField("transaction_type", StringType(), True),
        StructField("account_type", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("balance_before", DoubleType(), True),
        StructField("balance_after", DoubleType(), True),
        StructField("merchant", StringType(), True),
        StructField("location", StringType(), True),
        StructField("payment_method", StringType(), True),
        StructField("device_type", StringType(), True),
        StructField("transaction_status", StringType(), True),
        StructField("is_fraud", IntegerType(), True),
    ])

    logger.info(f"Loading raw banking transactions from {raw_data_path}...")
    df = spark.read.csv(raw_data_path, header=True, schema=schema)

    # 2. Data Cleaning & Duplicate Removal
    logger.info("Checking data counts and removing duplicates...")
    df_clean = df.dropDuplicates(["transaction_id"])

    # 3. Overview Summary Statistics
    logger.info("Calculating summary statistics across 15M transactions...")
    summary_stats = df_clean.select(
        F.count("*").alias("total_transactions"),
        F.sum("amount").alias("total_transaction_value"),
        F.countDistinct("customer_id").alias("total_customers"),
        F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)).alias("fraudulent_transactions"),
        F.avg("amount").alias("average_transaction_amount")
    ).collect()[0].asDict()

    summary_stats["fraud_rate"] = round((summary_stats["fraudulent_transactions"] / summary_stats["total_transactions"]) * 100, 2)
    summary_stats["total_transaction_value"] = round(summary_stats["total_transaction_value"], 2)
    summary_stats["average_transaction_amount"] = round(summary_stats["average_transaction_amount"], 2)

    logger.info(f"Summary Stats: {summary_stats}")

    # Save summary_stats.json
    summary_file = processed_path / "summary_stats.json"
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=2)

    # 4. Aggregations using PySpark SQL
    logger.info("Computing categorical & time-series aggregations...")
    tx_type_agg = df_clean.groupBy("transaction_type").agg(
        F.count("*").alias("count"),
        F.sum("amount").alias("total_amount"),
        F.avg("amount").alias("avg_amount"),
        F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)).alias("fraud_count")
    ).withColumn("fraud_rate", F.round((F.col("fraud_count") / F.col("count")) * 100, 2)) \
     .withColumn("total_amount", F.round("total_amount", 2)) \
     .withColumn("avg_amount", F.round("avg_amount", 2)) \
     .orderBy(F.col("count").desc())

    pm_agg = df_clean.groupBy("payment_method").agg(
        F.count("*").alias("count"),
        F.sum("amount").alias("total_amount"),
        F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)).alias("fraud_count")
    ).withColumn("fraud_rate", F.round((F.col("fraud_count") / F.col("count")) * 100, 2)) \
     .withColumn("total_amount", F.round("total_amount", 2)) \
     .orderBy(F.col("count").desc())

    loc_agg = df_clean.groupBy("location").agg(
        F.count("*").alias("count"),
        F.sum("amount").alias("total_amount"),
        F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)).alias("fraud_count")
    ).withColumn("fraud_rate", F.round((F.col("fraud_count") / F.col("count")) * 100, 2)) \
     .withColumn("total_amount", F.round("total_amount", 2)) \
     .orderBy(F.col("count").desc())

    device_agg = df_clean.groupBy("device_type").agg(
        F.count("*").alias("count"),
        F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)).alias("fraud_count")
    ).withColumn("fraud_rate", F.round((F.col("fraud_count") / F.col("count")) * 100, 2)) \
     .orderBy(F.col("count").desc())

    df_with_time = df_clean.withColumn("month", F.substring(F.col("transaction_date"), 1, 7))
    monthly_agg = df_with_time.groupBy("month").agg(
        F.count("*").alias("total_transactions"),
        F.sum("amount").alias("total_amount"),
        F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)).alias("fraud_count")
    ).withColumn("total_amount", F.round("total_amount", 2)) \
     .orderBy("month")

    daily_agg = df_clean.groupBy("transaction_date").agg(
        F.count("*").alias("total_transactions"),
        F.sum("amount").alias("total_amount"),
        F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)).alias("fraud_count")
    ).withColumn("total_amount", F.round("total_amount", 2)) \
     .orderBy("transaction_date")

    aggregations = {
        "transaction_type": [row.asDict() for row in tx_type_agg.collect()],
        "payment_method": [row.asDict() for row in pm_agg.collect()],
        "location": [row.asDict() for row in loc_agg.collect()],
        "device_type": [row.asDict() for row in device_agg.collect()],
        "monthly_trends": [row.asDict() for row in monthly_agg.collect()],
        "daily_trends": [row.asDict() for row in daily_agg.collect()]
    }

    aggregations_file = processed_path / "fraud_aggregates.json"
    with open(aggregations_file, "w") as f:
        json.dump(aggregations, f, indent=2)

    time_series = {
        "monthly": aggregations["monthly_trends"],
        "daily": aggregations["daily_trends"]
    }
    with open(processed_path / "time_series_aggregates.json", "w") as f:
        json.dump(time_series, f, indent=2)

    # 5. Customer Aggregation from 15M transactions
    logger.info("Aggregating customer behavioral features from 15M transactions...")
    cust_agg = df_clean.groupBy("customer_id").agg(
        F.round(F.sum("amount"), 2).alias("total_transaction_amount"),
        F.round(F.avg("amount"), 2).alias("average_transaction_amount"),
        F.count("*").alias("transaction_count"),
        F.round(F.avg("balance_before"), 2).alias("average_balance_before"),
        F.countDistinct("merchant").alias("unique_merchants"),
        F.sum(F.when(F.col("is_fraud") == 1, 1).otherwise(0)).alias("fraud_count")
    )
    cust_agg_df = cust_agg.toPandas()
    cust_agg_df.to_csv(processed_path / "customer_profiles_15m.csv", index=False)
    cust_agg_df.to_parquet(processed_path / "customer_profiles_15m.parquet", index=False)
    logger.info(f"Customer aggregation completed: {len(cust_agg_df):,} distinct customer profiles.")

    # 6. Stratified Reproducible Sampling for ML training & fast backend pagination
    logger.info("Creating reproducible stratified sample (500,000 rows)...")
    total_count = summary_stats["total_transactions"]
    sample_fraction = min(1.0, 500000.0 / total_count)
    sample_df = df_clean.sampleBy("is_fraud", fractions={0: sample_fraction, 1: sample_fraction}, seed=42)
    sample_pandas = sample_df.toPandas()
    logger.info(f"Stratified sample shape: {sample_pandas.shape}, fraud counts: {sample_pandas['is_fraud'].value_counts().to_dict()}")

    sample_parquet = processed_path / "transactions_sample.parquet"
    sample_pandas.to_parquet(sample_parquet, engine="pyarrow", index=False)

    # 7. Full processed parquet export (Chunked PyArrow streaming to bypass Windows Hadoop FileOutputCommitter)
    logger.info("Saving full processed dataset to Parquet via PyArrow streaming...")
    parquet_target = processed_path / "transactions_processed.parquet"
    if parquet_target.exists():
        if parquet_target.is_dir():
            shutil.rmtree(parquet_target)
        else:
            parquet_target.unlink()

    batch_size = 250000
    batch = []
    writer = None
    row_counter = 0

    for row in df_clean.toLocalIterator():
        batch.append(row.asDict())
        if len(batch) >= batch_size:
            tbl = pa.Table.from_pandas(pd.DataFrame(batch))
            if writer is None:
                writer = pq.ParquetWriter(str(parquet_target), tbl.schema)
            writer.write_table(tbl)
            row_counter += len(batch)
            logger.info(f"Streamed {row_counter:,} rows to Parquet...")
            batch.clear()

    if batch:
        tbl = pa.Table.from_pandas(pd.DataFrame(batch))
        if writer is None:
            writer = pq.ParquetWriter(str(parquet_target), tbl.schema)
        writer.write_table(tbl)
        row_counter += len(batch)

    if writer:
        writer.close()

    logger.info(f"Full Parquet dataset written successfully ({row_counter:,} total rows)!")

    spark.stop()
    logger.info("PySpark 15M Data Preprocessing completed successfully.")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    raw_csv = base_dir / "data" / "raw" / "banking_transactions_15m.csv"
    processed_dir = base_dir / "data" / "processed"
    run_pyspark_preprocessing(str(raw_csv), str(processed_dir))
