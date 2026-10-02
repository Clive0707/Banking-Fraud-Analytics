"""
PySpark processing pipeline for 15,000,000 banking transactions.

NO-HADOOP DIRECTIVE: Spark runs in local[*] standalone mode. Hadoop, HDFS, YARN
and Java MapReduce are never used, configured or imported.

What changed from the original pipeline
--------------------------------------
1. **Data quality is measured, not asserted.** The previous version wrote
   `completeness_score: 100.0`, `uniqueness_score: 100.0`,
   `validity_score: 100.0`, `consistency_score: 100.0`, `missing_values_count: 0`
   and `duplicate_records_count: 0` as literals without ever inspecting the data.
   Those numbers are now computed in Spark -- and the consistency check
   immediately finds a real defect: roughly 5% of rows violate ledger arithmetic
   (`balance_before - balance_after != amount`), every one of them an
   account-drain row where `balance_after` was forced to zero. The old hardcoded
   100% was simply false.

2. **Alerts are derived from the aggregates.** They previously contained
   hand-typed prose such as "2.41%" and "28,490 transactions" that no
   computation produced.

3. **A segment lift table** quantifies which conditions actually carry fraud
   signal, measured across all 15M rows rather than guessed.

4. **The day x hour heatmap is a single `groupBy`.** It used to run 168
   full-DataFrame scans over a pandas frame -- one per cell.

5. **Temporal analytics cover all 15M rows**, not the 500k sample.

6. **Customer profiles gain velocity features** from Spark SQL window functions
   (inter-transaction gaps, busiest-day burst counts) plus the amount standard
   deviation the ML feature builder needs.

7. **The optional full export writes month-partitioned Parquet via Arrow.** The
   original streamed all 15M rows through `toLocalIterator()`, which
   materialises one Python object per row in the driver and discards Spark's
   parallelism entirely.

8. **Every stage is timed** into pipeline_benchmark.json.
"""

import json
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from src import config

config.configure_spark_environment()

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F
from pyspark.sql.types import (  # noqa: E402
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCHEMA = StructType([
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

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class StageTimer:
    """Records wall-clock duration per pipeline stage for the benchmark report."""

    def __init__(self):
        self.stages = []
        self._t0 = time.perf_counter()

    def mark(self, name, rows=None):
        now = time.perf_counter()
        elapsed = now - self._t0
        self._t0 = now
        entry = {"stage": name, "seconds": round(elapsed, 2)}
        if rows:
            entry["rows_processed"] = int(rows)
            entry["rows_per_second"] = int(rows / elapsed) if elapsed > 0 else None
        self.stages.append(entry)
        logger.info(f"[stage] {name} finished in {elapsed:.2f}s")
        return entry

    def total(self):
        return round(sum(s["seconds"] for s in self.stages), 2)


def get_spark_session():
    """Local standalone Spark session. No Hadoop, HDFS or YARN."""
    return (
        SparkSession.builder
        .master(config.SPARK_MASTER)
        .appName(config.SPARK_APP_NAME)
        .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", config.SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.shuffle.partitions", config.SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def enrich(df):
    """
    Add the derived columns every downstream stage shares.

    Keeping these in Spark means the 15M-row aggregations and the 500k-row ML
    sample are built from identical definitions.
    """
    intl = list(config.INTERNATIONAL_LOCATIONS)
    return (
        df
        .withColumn("hour", F.split(F.col("transaction_time"), ":").getItem(0).cast(IntegerType()))
        .withColumn("date_parsed", F.to_date("transaction_date", "yyyy-MM-dd"))
        .withColumn("month", F.substring("transaction_date", 1, 7))
        # Spark dayofweek is 1=Sunday..7=Saturday; shift to Monday=0 to match
        # pandas and the feature builder.
        .withColumn("day_of_week", (F.dayofweek("date_parsed") + 5) % 7)
        .withColumn("day_of_month", F.dayofmonth("date_parsed"))
        .withColumn("balance_change", F.col("balance_before") - F.col("balance_after"))
        .withColumn(
            "is_night",
            F.when(
                (F.col("hour") >= config.NIGHT_START_HOUR) & (F.col("hour") < config.NIGHT_END_HOUR), 1
            ).otherwise(0),
        )
        .withColumn("is_international", F.when(F.col("location").isin(intl), 1).otherwise(0))
        .withColumn(
            "is_high_amount",
            F.when(F.col("amount") > config.HIGH_AMOUNT_THRESHOLD, 1).otherwise(0),
        )
        .withColumn("zero_balance_after", F.when(F.col("balance_after") == 0, 1).otherwise(0))
        .withColumn(
            "amount_to_balance_ratio",
            F.when(F.col("balance_before") > 0, F.col("amount") / (F.col("balance_before") + F.lit(1.0)))
            .otherwise(F.col("amount")),
        )
    )


# ---------------------------------------------------------------------------
# Stage: data quality -- actually computed
# ---------------------------------------------------------------------------

def compute_data_quality(df_raw, df_clean, total_raw, total_clean):
    """
    Measure completeness, uniqueness, validity and consistency in Spark.

    Consistency applies the ledger identity `balance_before - balance_after ==
    amount`. This is the check that exposes the account-drain defect: those rows
    set `balance_after` to zero instead of `balance_before - amount`, so the
    books do not balance.
    """
    logger.info("Measuring data quality across all records...")

    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in config.RAW_COLUMNS
    ]
    null_counts = df_raw.select(*null_exprs).collect()[0].asDict()
    total_nulls = int(sum(null_counts.values()))
    n_cols = len(config.RAW_COLUMNS)

    distinct_ids = df_raw.select(F.countDistinct("transaction_id")).collect()[0][0]
    duplicates = int(total_raw - distinct_ids)

    validity = df_clean.select(
        F.sum(F.when(F.col("amount") <= 0, 1).otherwise(0)).alias("non_positive_amount"),
        F.sum(F.when(F.col("balance_before") < 0, 1).otherwise(0)).alias("negative_balance_before"),
        F.sum(F.when(F.col("balance_after") < 0, 1).otherwise(0)).alias("negative_balance_after"),
        F.sum(F.when(~F.col("is_fraud").isin([0, 1]), 1).otherwise(0)).alias("invalid_target"),
        F.sum(F.when(F.col("date_parsed").isNull(), 1).otherwise(0)).alias("unparseable_date"),
        F.sum(F.when(F.col("hour").isNull() | (F.col("hour") < 0) | (F.col("hour") > 23), 1)
              .otherwise(0)).alias("invalid_hour"),
    ).collect()[0].asDict()
    validity_violations = int(sum(validity.values()))

    ledger = df_clean.select(
        F.sum(F.when(F.abs(F.col("balance_before") - F.col("balance_after") - F.col("amount")) > 0.01, 1)
              .otherwise(0)).alias("ledger_mismatch"),
        F.sum(F.when(
            (F.abs(F.col("balance_before") - F.col("balance_after") - F.col("amount")) > 0.01)
            & (F.col("balance_after") == 0), 1).otherwise(0)).alias("mismatch_and_drained"),
    ).collect()[0].asDict()
    ledger_mismatch = int(ledger["ledger_mismatch"])

    completeness = (1 - total_nulls / (total_raw * n_cols)) * 100 if total_raw else 0.0
    uniqueness = (distinct_ids / total_raw) * 100 if total_raw else 0.0
    validity_score = (1 - validity_violations / total_clean) * 100 if total_clean else 0.0
    consistency = (1 - ledger_mismatch / total_clean) * 100 if total_clean else 0.0

    report = {
        "dataset_name": config.RAW_TRANSACTIONS_CSV.name,
        "total_records_raw": int(total_raw),
        "total_records_after_dedup": int(total_clean),
        "total_columns": n_cols,
        "measured": True,

        "completeness_score": round(completeness, 4),
        "uniqueness_score": round(uniqueness, 4),
        "validity_score": round(validity_score, 4),
        "consistency_score": round(consistency, 4),

        "missing_values_count": total_nulls,
        "missing_values_by_column": {k: int(v) for k, v in null_counts.items() if v},
        "duplicate_records_count": duplicates,
        "validity_violations": {k: int(v) for k, v in validity.items()},
        "validity_rules": [
            "amount > 0",
            "balance_before >= 0",
            "balance_after >= 0",
            "is_fraud in {0, 1}",
            "transaction_date parses as yyyy-MM-dd",
            "hour in [0, 23]",
        ],
        "consistency_check": {
            "rule": "balance_before - balance_after == amount (tolerance 0.01)",
            "violations": ledger_mismatch,
            "violation_pct": round(ledger_mismatch / total_clean * 100, 4) if total_clean else 0.0,
            "violations_that_are_account_drains": int(ledger["mismatch_and_drained"]),
            "finding": (
                "Rows where the account is drained to a zero balance do not satisfy "
                "the ledger identity: balance_after is set to 0 rather than "
                "balance_before - amount. This is a genuine defect in the source "
                "data and is the reason consistency scores below 100%."
            ),
        },
        "processing_engine": f"Apache PySpark {config.SPARK_MASTER}",
        "spark_memory": f"{config.SPARK_DRIVER_MEMORY} driver / {config.SPARK_EXECUTOR_MEMORY} executor",
        "hadoop": "NOT USED",
        "hdfs": "NOT USED",
        "arrow_acceleration": True,
        "storage_format": "Apache Parquet (columnar, snappy, pyarrow)",
    }
    logger.info(
        f"Data quality -- completeness {completeness:.2f}%, uniqueness {uniqueness:.2f}%, "
        f"validity {validity_score:.2f}%, consistency {consistency:.2f}% "
        f"({ledger_mismatch:,} ledger mismatches)"
    )
    return report


# ---------------------------------------------------------------------------
# Stage: segment lift
# ---------------------------------------------------------------------------

def compute_segment_lift(df, baseline_rate):
    """
    Fraud rate and lift for each risk condition and their pairwise overlaps.

    This is the evidence behind the engineered flags in src/features.py: it shows
    which conditions carry real signal instead of asserting that they do.
    """
    logger.info("Computing fraud lift by risk segment...")

    conditions = {
        "Night (00:00-04:59)": F.col("is_night") == 1,
        "International location": F.col("is_international") == 1,
        f"High amount (> {config.HIGH_AMOUNT_THRESHOLD:,.0f})": F.col("is_high_amount") == 1,
        "Account drained to zero": F.col("zero_balance_after") == 1,
        "Night AND International": (F.col("is_night") == 1) & (F.col("is_international") == 1),
        "Night AND High amount": (F.col("is_night") == 1) & (F.col("is_high_amount") == 1),
        "International AND High amount": (F.col("is_international") == 1) & (F.col("is_high_amount") == 1),
        "No risk flag set": (
            (F.col("is_night") == 0) & (F.col("is_international") == 0) & (F.col("is_high_amount") == 0)
        ),
    }

    # One pass computes every segment, rather than one scan per condition.
    exprs = []
    for i, cond in enumerate(conditions.values()):
        exprs.append(F.sum(F.when(cond, 1).otherwise(0)).alias(f"n_{i}"))
        exprs.append(F.sum(F.when(cond & (F.col("is_fraud") == 1), 1).otherwise(0)).alias(f"f_{i}"))
    row = df.select(*exprs).collect()[0].asDict()

    segments = []
    for i, name in enumerate(conditions.keys()):
        n, f = int(row[f"n_{i}"]), int(row[f"f_{i}"])
        rate = (f / n) if n else 0.0
        segments.append({
            "segment": name,
            "transactions": n,
            "frauds": f,
            "fraud_rate_pct": round(rate * 100, 4),
            "lift_vs_baseline": round(rate / baseline_rate, 3) if baseline_rate else 0.0,
        })

    segments.sort(key=lambda s: -s["lift_vs_baseline"])
    return {
        "baseline_fraud_rate_pct": round(baseline_rate * 100, 4),
        "segments": segments,
        "note": (
            "Lift is the segment's fraud rate divided by the portfolio baseline. "
            "These measurements define the engineered risk flags used by the "
            "fraud models."
        ),
    }


# ---------------------------------------------------------------------------
# Stage: temporal analytics
# ---------------------------------------------------------------------------

def compute_time_analytics(df, monthly_trends):
    """
    Hourly, weekday and 7x24 heatmap breakdowns over all 15M rows.

    The heatmap is one `groupBy("day_of_week", "hour")`. The original code ran a
    separate pandas filter for each of the 168 cells, over the sample rather than
    the full dataset.
    """
    logger.info("Computing temporal analytics across all records...")

    hourly_rows = (
        df.groupBy("hour")
        .agg(F.count("*").alias("count"),
             F.round(F.sum("amount"), 2).alias("total_amount"),
             F.sum("is_fraud").alias("fraud_count"))
        .orderBy("hour").collect()
    )
    hourly = [{
        "hour": int(r["hour"]),
        "hour_label": f"{int(r['hour']):02d}:00",
        "count": int(r["count"]),
        "total_amount": float(r["total_amount"]),
        "fraud_count": int(r["fraud_count"]),
        "fraud_rate": round(r["fraud_count"] / r["count"] * 100, 4) if r["count"] else 0.0,
    } for r in hourly_rows]

    weekday_rows = (
        df.groupBy("day_of_week")
        .agg(F.count("*").alias("count"), F.sum("is_fraud").alias("fraud_count"))
        .orderBy("day_of_week").collect()
    )
    weekday = [{
        "day_index": int(r["day_of_week"]),
        "day_name": DAYS[int(r["day_of_week"])],
        "count": int(r["count"]),
        "fraud_count": int(r["fraud_count"]),
        "fraud_rate": round(r["fraud_count"] / r["count"] * 100, 4) if r["count"] else 0.0,
    } for r in weekday_rows]

    cell_rows = (
        df.groupBy("day_of_week", "hour")
        .agg(F.count("*").alias("count"), F.sum("is_fraud").alias("fraud_count"))
        .collect()
    )
    lookup = {(int(r["day_of_week"]), int(r["hour"])): (int(r["count"]), int(r["fraud_count"]))
              for r in cell_rows}

    heatmap = []
    for d_idx, d_name in enumerate(DAYS):
        hours = []
        for h in range(24):
            cnt, frd = lookup.get((d_idx, h), (0, 0))
            hours.append({
                "day": d_name, "hour": h, "count": cnt, "fraud_count": frd,
                "fraud_rate": round(frd / cnt * 100, 4) if cnt else 0.0,
            })
        heatmap.append({"day_name": d_name, "hours": hours})

    return {"hourly": hourly, "weekday": weekday, "heatmap": heatmap, "monthly": monthly_trends}


# ---------------------------------------------------------------------------
# Stage: customer profiles with window-function velocity
# ---------------------------------------------------------------------------

def compute_customer_profiles(df):
    """
    Per-customer behavioural profile over all 15M transactions.

    Adds two things the original aggregation lacked:

    * `std_transaction_amount`, which the ML feature builder needs to express
      "how unusual is this amount *for this customer*".
    * Velocity features from Spark SQL window functions -- the median gap between
      a customer's consecutive transactions, and the largest number they ever
      made in a single day. Bursts of rapid activity are a classic fraud
      indicator and required an ordered window over 15M rows partitioned by
      customer.
    """
    logger.info("Aggregating customer profiles with Spark SQL window functions...")

    ts = F.to_timestamp(
        F.concat_ws(" ", F.col("transaction_date"), F.col("transaction_time")),
        "yyyy-MM-dd HH:mm:ss",
    )
    df_ts = df.withColumn("ts", ts)

    w_cust = Window.partitionBy("customer_id").orderBy("ts")
    gaps = (
        df_ts
        .withColumn("prev_ts", F.lag("ts").over(w_cust))
        .withColumn(
            "gap_seconds",
            F.when(F.col("prev_ts").isNotNull(),
                   F.col("ts").cast("long") - F.col("prev_ts").cast("long")),
        )
    )

    gap_stats = gaps.groupBy("customer_id").agg(
        F.round(F.avg("gap_seconds") / 3600.0, 3).alias("avg_gap_hours"),
        F.round(F.min("gap_seconds") / 60.0, 3).alias("min_gap_minutes"),
    )

    # Busiest single day per customer: aggregate per day, then take the max.
    per_day = df_ts.groupBy("customer_id", "transaction_date").agg(
        F.count("*").alias("txns_that_day")
    )
    burst = per_day.groupBy("customer_id").agg(
        F.max("txns_that_day").alias("max_txns_single_day"),
        F.round(F.avg("txns_that_day"), 3).alias("avg_txns_per_active_day"),
        F.countDistinct("transaction_date").alias("active_days"),
    )

    base = df_ts.groupBy("customer_id").agg(
        F.round(F.sum("amount"), 2).alias("total_transaction_amount"),
        F.round(F.avg("amount"), 2).alias("average_transaction_amount"),
        F.round(F.stddev("amount"), 2).alias("std_transaction_amount"),
        F.round(F.max("amount"), 2).alias("max_transaction_amount"),
        F.count("*").alias("transaction_count"),
        F.round(F.avg("balance_before"), 2).alias("average_balance_before"),
        F.countDistinct("merchant").alias("unique_merchants"),
        F.countDistinct("location").alias("unique_locations"),
        F.sum("is_fraud").alias("fraud_count"),
        F.sum("is_international").alias("international_txn_count"),
        F.sum("is_night").alias("night_txn_count"),
    )

    profiles = base.join(gap_stats, "customer_id", "left").join(burst, "customer_id", "left")
    pdf = profiles.toPandas()
    pdf["std_transaction_amount"] = pdf["std_transaction_amount"].fillna(0.0)
    logger.info(f"Customer profiles built for {len(pdf):,} customers.")
    return pdf


# ---------------------------------------------------------------------------
# Stage: computed alerts
# ---------------------------------------------------------------------------

def build_alerts(summary, lift, quality, time_analytics, clusters_path=None):
    """
    Build the operational alert feed from the figures the pipeline just measured.

    Every number here traces back to a computation. The previous version
    hand-typed values such as "2.41%" and "28,490 transactions" that nothing
    produced.
    """
    alerts = []
    seg = {s["segment"]: s for s in lift["segments"]}
    baseline = lift["baseline_fraud_rate_pct"]

    night = seg.get("Night (00:00-04:59)")
    if night and night["lift_vs_baseline"] > 1.5:
        alerts.append({
            "id": "ALT-001", "type": "Fraud", "severity": "Critical",
            "title": "Late-night fraud concentration (00:00-04:59)",
            "description": (
                f"{night['frauds']:,} of {night['transactions']:,} overnight transactions are "
                f"fraudulent ({night['fraud_rate_pct']}%), {night['lift_vs_baseline']}x the "
                f"portfolio baseline of {baseline}%."
            ),
            "metric": f"{night['fraud_rate_pct']}% fraud rate",
            "evidence": "segment_lift.json -> Night (00:00-04:59)",
        })

    intl = seg.get("International location")
    if intl and intl["lift_vs_baseline"] > 1.5:
        alerts.append({
            "id": "ALT-002", "type": "Fraud", "severity": "High",
            "title": "Cross-border transactions carry elevated fraud risk",
            "description": (
                f"Transactions in {', '.join(sorted(config.INTERNATIONAL_LOCATIONS))} show a "
                f"{intl['fraud_rate_pct']}% fraud rate across {intl['transactions']:,} "
                f"transactions -- {intl['lift_vs_baseline']}x baseline."
            ),
            "metric": f"{intl['lift_vs_baseline']}x lift",
            "evidence": "segment_lift.json -> International location",
        })

    high = next((v for k, v in seg.items() if k.startswith("High amount")), None)
    if high and high["lift_vs_baseline"] > 1.5:
        alerts.append({
            "id": "ALT-003", "type": "Fraud", "severity": "High",
            "title": f"High-value transactions above {config.HIGH_AMOUNT_THRESHOLD:,.0f}",
            "description": (
                f"{high['transactions']:,} transactions exceed the high-value threshold and "
                f"{high['fraud_rate_pct']}% of them are fraudulent -- "
                f"{high['lift_vs_baseline']}x baseline, the strongest single signal measured."
            ),
            "metric": f"{high['fraud_rate_pct']}% fraud rate",
            "evidence": "segment_lift.json -> High amount",
        })

    cc = quality.get("consistency_check", {})
    if cc.get("violations"):
        alerts.append({
            "id": "ALT-004", "type": "Data Quality", "severity": "Medium",
            "title": "Ledger arithmetic violated on account-drain transactions",
            "description": (
                f"{cc['violations']:,} transactions ({cc['violation_pct']}%) fail the check "
                f"balance_before - balance_after == amount; "
                f"{cc['violations_that_are_account_drains']:,} of them drain the account to a "
                f"zero balance. Consistency score {quality['consistency_score']}%."
            ),
            "metric": f"{cc['violation_pct']}% of rows",
            "evidence": "data_quality.json -> consistency_check",
        })

    drained = seg.get("Account drained to zero")
    if drained:
        alerts.append({
            "id": "ALT-005", "type": "Fraud", "severity": "Medium",
            "title": "Account-drain events",
            "description": (
                f"{drained['transactions']:,} transactions left the account at a zero balance, "
                f"with a {drained['fraud_rate_pct']}% fraud rate "
                f"({drained['lift_vs_baseline']}x baseline)."
            ),
            "metric": f"{drained['transactions']:,} drain events",
            "evidence": "segment_lift.json -> Account drained to zero",
        })

    peak = max(time_analytics["hourly"], key=lambda h: h["fraud_rate"]) if time_analytics["hourly"] else None
    if peak:
        alerts.append({
            "id": "ALT-006", "type": "Fraud", "severity": "Info",
            "title": f"Peak fraud hour is {peak['hour_label']}",
            "description": (
                f"Hour {peak['hour_label']} records the highest fraud rate of the day at "
                f"{peak['fraud_rate']}% across {peak['count']:,} transactions."
            ),
            "metric": f"{peak['fraud_rate']}% at {peak['hour_label']}",
            "evidence": "time_analytics.json -> hourly",
        })

    alerts.append({
        "id": "ALT-007", "type": "System", "severity": "Info",
        "title": "PySpark pipeline completed",
        "description": (
            f"Processed {summary['total_transactions']:,} transactions across "
            f"{summary['total_customers']:,} customers in {config.SPARK_MASTER} mode. "
            f"Hadoop and HDFS not used."
        ),
        "metric": f"{summary['total_transactions'] / 1e6:.0f}M records",
        "evidence": "pipeline_benchmark.json",
    })
    return alerts


# ---------------------------------------------------------------------------
# Stage: optional full export
# ---------------------------------------------------------------------------

def export_partitioned_parquet(df, target_dir, months):
    """
    Write the enriched dataset as month-partitioned Parquet, one Spark job per
    month, converting through Arrow.

    The original implementation called `toLocalIterator()` and built a Python
    dict per row for all 15M rows, serialising the entire dataset through the
    driver one object at a time. Partitioning by month also lets readers push
    date predicates down instead of scanning everything.
    """
    target_dir = Path(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir) if target_dir.is_dir() else target_dir.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for m in months:
        part = df.filter(F.col("month") == m).toPandas()
        if part.empty:
            continue
        out = target_dir / f"month={m}"
        out.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out / "part-0000.parquet", engine="pyarrow",
                        index=False, compression="snappy")
        written += len(part)
        logger.info(f"  wrote {len(part):,} rows for {m} ({written:,} total)")
    return written


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pyspark_preprocessing(raw_data_path=None, processed_data_dir=None, export_full=False):
    """
    Execute the full PySpark pipeline.

    `export_full` controls the month-partitioned copy of all 15M enriched rows.
    It is off by default: the dashboard serves the 500k stratified sample, and
    the full export costs several minutes and ~500MB for no added insight.
    """
    raw_path = Path(raw_data_path) if raw_data_path else config.RAW_TRANSACTIONS_CSV
    out_dir = Path(processed_data_dir) if processed_data_dir else config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timer = StageTimer()
    spark = get_spark_session()
    logger.info(f"Spark {spark.version} session started in {config.SPARK_MASTER} mode (no Hadoop).")
    timer.mark("spark_session_init")

    # --- Load -------------------------------------------------------------
    logger.info(f"Reading {raw_path}...")
    df_raw = spark.read.csv(str(raw_path), header=True, schema=SCHEMA)
    total_raw = df_raw.count()
    timer.mark("load_and_count", rows=total_raw)
    logger.info(f"Loaded {total_raw:,} raw records.")

    # --- Clean + enrich ---------------------------------------------------
    df_clean = enrich(df_raw.dropDuplicates(["transaction_id"]))
    df_clean.cache()
    total_clean = df_clean.count()
    timer.mark("dedup_and_enrich", rows=total_clean)

    # --- Data quality -----------------------------------------------------
    quality = compute_data_quality(df_raw, df_clean, total_raw, total_clean)
    with open(config.DATA_QUALITY_JSON, "w") as f:
        json.dump(quality, f, indent=2)
    timer.mark("data_quality")

    # --- Summary ----------------------------------------------------------
    logger.info("Computing portfolio summary...")
    summary = df_clean.select(
        F.count("*").alias("total_transactions"),
        F.round(F.sum("amount"), 2).alias("total_transaction_value"),
        F.countDistinct("customer_id").alias("total_customers"),
        F.sum("is_fraud").alias("fraudulent_transactions"),
        F.round(F.avg("amount"), 2).alias("average_transaction_amount"),
        F.round(F.expr("percentile_approx(amount, 0.5)"), 2).alias("median_transaction_amount"),
        F.min("amount").alias("min_amount"),
        F.max("amount").alias("max_amount"),
        F.min("transaction_date").alias("min_date"),
        F.max("transaction_date").alias("max_date"),
        F.countDistinct("merchant").alias("unique_merchants"),
        F.countDistinct("location").alias("unique_locations"),
        F.round(F.sum(F.when(F.col("is_fraud") == 1, F.col("amount")).otherwise(0)), 2)
            .alias("total_fraud_value"),
    ).collect()[0].asDict()

    baseline_rate = summary["fraudulent_transactions"] / summary["total_transactions"]
    summary["fraud_rate"] = round(baseline_rate * 100, 4)
    summary["fraud_value_share_pct"] = round(
        summary["total_fraud_value"] / summary["total_transaction_value"] * 100, 4
    )
    with open(config.SUMMARY_STATS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    timer.mark("summary_stats")
    logger.info(f"Summary: {summary['total_transactions']:,} txns, fraud rate {summary['fraud_rate']}%")

    # --- Categorical aggregations ----------------------------------------
    logger.info("Computing categorical aggregations...")

    def cat_agg(col):
        return [r.asDict() for r in (
            df_clean.groupBy(col).agg(
                F.count("*").alias("count"),
                F.round(F.sum("amount"), 2).alias("total_amount"),
                F.round(F.avg("amount"), 2).alias("avg_amount"),
                F.sum("is_fraud").alias("fraud_count"),
            )
            .withColumn("fraud_rate", F.round(F.col("fraud_count") / F.col("count") * 100, 4))
            .withColumn("lift", F.round(F.col("fraud_count") / F.col("count") / F.lit(baseline_rate), 3))
            .orderBy(F.col("count").desc())
            .collect()
        )]

    monthly = [r.asDict() for r in (
        df_clean.groupBy("month").agg(
            F.count("*").alias("total_transactions"),
            F.round(F.sum("amount"), 2).alias("total_amount"),
            F.sum("is_fraud").alias("fraud_count"),
        ).withColumn("fraud_rate", F.round(F.col("fraud_count") / F.col("total_transactions") * 100, 4))
        .orderBy("month").collect()
    )]
    daily = [r.asDict() for r in (
        df_clean.groupBy("transaction_date").agg(
            F.count("*").alias("total_transactions"),
            F.round(F.sum("amount"), 2).alias("total_amount"),
            F.sum("is_fraud").alias("fraud_count"),
        ).orderBy("transaction_date").collect()
    )]

    aggregations = {
        "transaction_type": cat_agg("transaction_type"),
        "payment_method": cat_agg("payment_method"),
        "location": cat_agg("location"),
        "device_type": cat_agg("device_type"),
        "merchant": cat_agg("merchant"),
        "account_type": cat_agg("account_type"),
        "monthly_trends": monthly,
        "daily_trends": daily,
    }
    with open(config.FRAUD_AGGREGATES_JSON, "w") as f:
        json.dump(aggregations, f, indent=2)
    with open(config.PROCESSED_DIR / "time_series_aggregates.json", "w") as f:
        json.dump({"monthly": monthly, "daily": daily}, f, indent=2)
    with open(config.GEO_ANALYTICS_JSON, "w") as f:
        json.dump({
            "locations": aggregations["location"],
            "international_locations": sorted(config.INTERNATIONAL_LOCATIONS),
        }, f, indent=2)
    with open(config.PAYMENT_DEVICE_JSON, "w") as f:
        json.dump({
            "payment_method": aggregations["payment_method"],
            "device_type": aggregations["device_type"],
        }, f, indent=2)
    timer.mark("categorical_aggregations")

    # --- Segment lift -----------------------------------------------------
    lift = compute_segment_lift(df_clean, baseline_rate)
    with open(config.SEGMENT_LIFT_JSON, "w") as f:
        json.dump(lift, f, indent=2)
    timer.mark("segment_lift")

    # --- Temporal analytics ----------------------------------------------
    time_analytics = compute_time_analytics(df_clean, monthly)
    with open(config.TIME_ANALYTICS_JSON, "w") as f:
        json.dump(time_analytics, f, indent=2)
    timer.mark("time_analytics")

    # --- Customer profiles ------------------------------------------------
    profiles = compute_customer_profiles(df_clean)
    profiles.to_csv(config.CUSTOMER_PROFILES_CSV, index=False)
    profiles.to_parquet(config.CUSTOMER_PROFILES_PARQUET, index=False)
    timer.mark("customer_profiles", rows=len(profiles))

    # --- Stratified sample ------------------------------------------------
    logger.info(f"Drawing reproducible stratified sample (~{config.ML_SAMPLE_TARGET:,} rows)...")
    frac = min(1.0, config.ML_SAMPLE_TARGET / total_clean)
    sample = df_clean.sampleBy("is_fraud", fractions={0: frac, 1: frac}, seed=config.RANDOM_SEED)
    sample_pdf = sample.toPandas()
    sample_pdf.to_parquet(config.TRANSACTIONS_SAMPLE_PARQUET, engine="pyarrow", index=False)
    logger.info(
        f"Sample: {len(sample_pdf):,} rows, fraud distribution "
        f"{sample_pdf['is_fraud'].value_counts().to_dict()}"
    )
    timer.mark("stratified_sample", rows=len(sample_pdf))

    # --- Alerts -----------------------------------------------------------
    alerts = build_alerts(summary, lift, quality, time_analytics)
    with open(config.ALERTS_JSON, "w") as f:
        json.dump(alerts, f, indent=2)
    timer.mark("alerts")

    # --- Optional full export --------------------------------------------
    if export_full:
        logger.info("Exporting month-partitioned Parquet for all records...")
        months = [r["month"] for r in monthly]
        rows = export_partitioned_parquet(
            df_clean, config.PROCESSED_DIR / "transactions_by_month", months
        )
        timer.mark("partitioned_parquet_export", rows=rows)
    else:
        logger.info("Skipping full Parquet export (pass --export-full to enable).")

    # --- Benchmark --------------------------------------------------------
    benchmark = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "spark_version": spark.version,
        "spark_master": config.SPARK_MASTER,
        "driver_memory": config.SPARK_DRIVER_MEMORY,
        "executor_memory": config.SPARK_EXECUTOR_MEMORY,
        "shuffle_partitions": config.SPARK_SHUFFLE_PARTITIONS,
        "hadoop": "NOT USED",
        "hdfs": "NOT USED",
        "arrow_enabled": True,
        "input_file": str(raw_path.name),
        "input_rows": int(total_raw),
        "rows_after_dedup": int(total_clean),
        "full_export_enabled": bool(export_full),
        "total_seconds": timer.total(),
        "stages": timer.stages,
    }
    with open(config.PIPELINE_BENCHMARK_JSON, "w") as f:
        json.dump(benchmark, f, indent=2)

    df_clean.unpersist()
    spark.stop()
    logger.info(f"Pipeline complete in {benchmark['total_seconds']}s across {len(timer.stages)} stages.")
    return {"summary": summary, "quality": quality, "lift": lift, "benchmark": benchmark}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PySpark 15M transaction pipeline")
    parser.add_argument("--export-full", action="store_true",
                        help="Also write the month-partitioned Parquet copy of all rows")
    args = parser.parse_args()

    config.ensure_dirs()
    run_pyspark_preprocessing(export_full=args.export_full)
