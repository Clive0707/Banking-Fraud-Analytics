"""
Central configuration for the Banking Fraud Analytics platform.

Everything that used to be hardcoded in several places -- filesystem paths, the
Windows JAVA_HOME location, Spark tuning, the domain thresholds that define a
"risky" transaction, and the cost model used for threshold optimisation -- lives
here so the pipeline, the ML trainer and the Flask backend all agree.

NO-HADOOP DIRECTIVE: Spark runs in local[*] standalone mode. Hadoop, HDFS, YARN
and Java MapReduce are never used, configured or imported.
"""

import glob
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FRONTEND_DIR = BASE_DIR / "frontend"

RAW_TRANSACTIONS_CSV = RAW_DIR / "banking_transactions_15m.csv"
RAW_CUSTOMERS_CSV = RAW_DIR / "customer_profiles.csv"

# Processed artefacts
TRANSACTIONS_SAMPLE_PARQUET = PROCESSED_DIR / "transactions_sample.parquet"
TRANSACTIONS_PROCESSED_PARQUET = PROCESSED_DIR / "transactions_processed.parquet"
CUSTOMER_PROFILES_PARQUET = PROCESSED_DIR / "customer_profiles_15m.parquet"
CUSTOMER_PROFILES_CSV = PROCESSED_DIR / "customer_profiles_15m.csv"

SUMMARY_STATS_JSON = PROCESSED_DIR / "summary_stats.json"
FRAUD_AGGREGATES_JSON = PROCESSED_DIR / "fraud_aggregates.json"
DATA_QUALITY_JSON = PROCESSED_DIR / "data_quality.json"
TIME_ANALYTICS_JSON = PROCESSED_DIR / "time_analytics.json"
GEO_ANALYTICS_JSON = PROCESSED_DIR / "geo_analytics.json"
PAYMENT_DEVICE_JSON = PROCESSED_DIR / "payment_device_analytics.json"
ALERTS_JSON = PROCESSED_DIR / "analytical_alerts.json"
SEGMENT_LIFT_JSON = PROCESSED_DIR / "segment_lift.json"
LEAKAGE_REPORT_JSON = PROCESSED_DIR / "leakage_report.json"
PIPELINE_BENCHMARK_JSON = PROCESSED_DIR / "pipeline_benchmark.json"
CUSTOMER_CLUSTERS_JSON = PROCESSED_DIR / "customer_clusters.json"
ANOMALIES_JSON = PROCESSED_DIR / "anomalies_summary.json"

MODEL_COMPARISON_JSON = MODELS_DIR / "model_comparison.json"
THRESHOLD_ANALYSIS_JSON = MODELS_DIR / "threshold_analysis.json"
FEATURE_IMPORTANCE_JSON = MODELS_DIR / "feature_importance.json"


def ensure_dirs():
    """Create every output directory the pipeline writes into."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Spark / JVM environment
# ---------------------------------------------------------------------------

SPARK_APP_NAME = "BankingFraudAnalytics-PySpark-15M"
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "8g")
SPARK_EXECUTOR_MEMORY = os.environ.get("SPARK_EXECUTOR_MEMORY", "8g")
SPARK_SHUFFLE_PARTITIONS = os.environ.get("SPARK_SHUFFLE_PARTITIONS", "64")

# Candidate JDK install roots, newest first. Replaces the previously hardcoded
# "C:\Program Files\Java\jdk-22" so the project runs on any machine.
_JDK_GLOBS = [
    r"C:\Program Files\Java\jdk-*",
    r"C:\Program Files\Eclipse Adoptium\jdk-*",
    r"C:\Program Files\Microsoft\jdk-*",
    r"C:\Program Files\Zulu\zulu-*",
    "/usr/lib/jvm/java-*",
    "/Library/Java/JavaVirtualMachines/*/Contents/Home",
]


def _is_valid_jdk(path):
    """
    True only if `path` is a directory containing a java launcher.

    Checking mere existence is not enough: this machine had JAVA_HOME pointing at
    a downloaded `.msi` installer, which exists as a file and would otherwise be
    accepted, leaving Spark unable to start. The original code worked around that
    by hardcoding a single JDK path.
    """
    if not path:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    return (p / "bin" / "java.exe").exists() or (p / "bin" / "java").exists()


def detect_java_home():
    """
    Return a usable JAVA_HOME.

    Order of preference: a JAVA_HOME that really points at a JDK, then the JDK
    owning the `java` on PATH, then the newest JDK in a known install root.
    """
    existing = os.environ.get("JAVA_HOME", "").strip().strip('"')
    if _is_valid_jdk(existing):
        return existing
    if existing:
        logger.warning(
            f"Ignoring JAVA_HOME={existing!r}: it is not a JDK directory "
            f"(no bin/java). Falling back to auto-detection."
        )

    # A `java` already on PATH is the most reliable signal available.
    import shutil as _shutil
    java_exe = _shutil.which("java")
    if java_exe:
        home = Path(java_exe).resolve().parent.parent
        if _is_valid_jdk(home):
            return str(home)

    candidates = []
    for pattern in _JDK_GLOBS:
        candidates.extend(glob.glob(pattern))
    # Newest version last alphabetically is a good enough heuristic (jdk-17 < jdk-22).
    candidates = sorted(c for c in candidates if _is_valid_jdk(c))
    return candidates[-1] if candidates else None


def configure_spark_environment():
    """
    Prepare the process environment for PySpark on Windows.

    Sets JAVA_HOME when it can be discovered, and strips quoting plus the
    stray Windows Installer entries that otherwise make Spark's launcher fail.
    """
    java_home = detect_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        logger.info(f"JAVA_HOME resolved to {java_home}")
    else:
        logger.warning("No JDK found automatically; relying on the ambient JAVA_HOME/PATH.")

    clean_paths = []
    for p in os.environ.get("PATH", "").split(os.pathsep):
        p_clean = p.replace('"', "").strip()
        if p_clean and "msi" not in p_clean.lower():
            clean_paths.append(p_clean)
    os.environ["PATH"] = os.pathsep.join(clean_paths)

    # Make the driver and workers use the same interpreter that launched us.
    import sys
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

RAW_COLUMNS = [
    "transaction_id", "customer_id", "transaction_date", "transaction_time",
    "transaction_type", "account_type", "amount", "balance_before",
    "balance_after", "merchant", "location", "payment_method",
    "device_type", "transaction_status", "is_fraud",
]

TARGET_COL = "is_fraud"

CATEGORICAL_COLS = [
    "transaction_type", "account_type", "payment_method", "device_type", "location",
]

# ---------------------------------------------------------------------------
# Domain thresholds -- derived from the lift analysis in reports/, not guessed.
# ---------------------------------------------------------------------------

# Locations outside India. Measured fraud rate 4.3-5.7% versus ~0.9% domestic
# (roughly 5x lift), which makes this the single strongest non-leaking signal.
INTERNATIONAL_LOCATIONS = {"Dubai", "Singapore", "New York", "London"}

# Fraud rate is 2.85% for hours 00:00-04:59 against 0.45% for the rest of the
# day. Note the original code tested `hour > 23`, which can never be true.
NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 5  # exclusive

# Transactions above this amount show a 9x fraud lift (8.7% versus 0.96%).
HIGH_AMOUNT_THRESHOLD = 100_000.0

# Spend that drains the account to exactly zero.
ZERO_BALANCE_EPS = 1e-9

# ---------------------------------------------------------------------------
# Target leakage
# ---------------------------------------------------------------------------
# `transaction_status` records the bank's OWN fraud verdict: every Declined and
# every Flagged transaction is fraudulent (precision 1.000, recall 0.743), while
# no Pending transaction ever is. Using it as a predictor would mean training on
# the answer, so it is excluded from every deployment feature set. The leakage
# study in src/fraud_detection/leakage_analysis.py quantifies it instead of
# silently dropping the column.
LEAKING_COLS = ["transaction_status"]
LEAKING_STATUS_VALUES = ["Declined", "Flagged"]

# ---------------------------------------------------------------------------
# Cost model for threshold optimisation
# ---------------------------------------------------------------------------
# A fraud model is only useful if its operating threshold reflects what errors
# actually cost. Reviewing a flagged transaction costs analyst time; missing a
# fraud costs the transaction value. Both are configurable via env vars.

COST_PER_REVIEW = float(os.environ.get("COST_PER_REVIEW", 200.0))      # INR per manual review
RECOVERY_RATE = float(os.environ.get("RECOVERY_RATE", 0.90))           # share of a caught fraud recovered

# Fraction of the portfolio an analyst team can realistically review.
REVIEW_CAPACITY_FRACTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]

# ---------------------------------------------------------------------------
# ML training
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
ML_SAMPLE_TARGET = 500_000

# Base numeric features shared by every model.
NUMERIC_FEATURES = [
    "amount", "amount_log", "balance_before", "balance_after", "balance_change",
    "hour", "day_of_week", "day_of_month", "amount_to_balance_ratio",
]

# Engineered binary risk flags and their interactions.
FLAG_FEATURES = [
    "is_night", "is_international", "is_high_amount", "zero_balance_after",
    "intl_x_night", "intl_x_high", "night_x_high",
]

# Customer-relative behavioural features, joined from the 15M-row Spark
# aggregation. Deliberately excludes the per-customer fraud_count, which would
# leak the target.
CUSTOMER_FEATURES = [
    "cust_avg_amount", "cust_txn_count", "cust_unique_merchants",
    "amount_z_vs_customer", "amount_over_cust_mean",
]

# ---------------------------------------------------------------------------
# Clustering / anomaly detection
# ---------------------------------------------------------------------------

KMEANS_K_RANGE = range(2, 9)
KMEANS_OPTIMAL_K = 4
CLUSTER_FEATURES = [
    "total_transaction_amount", "average_transaction_amount", "transaction_count",
    "average_balance_before", "unique_merchants", "fraud_count",
]

ISOLATION_CONTAMINATION = 0.025
ISOLATION_FEATURES = [
    "amount", "balance_before", "balance_after", "balance_change",
    "hour", "is_night", "amount_to_balance_ratio",
]

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

DEFAULT_PORT = int(os.environ.get("PORT", 5000))
MAX_EXPORT_ROWS = 50_000
