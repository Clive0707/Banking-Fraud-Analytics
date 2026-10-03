"""
Entry point for the Banking Fraud Analytics platform.

Runs the stages in order -- PySpark preprocessing, customer segmentation,
anomaly detection, fraud model training -- then serves the dashboard. Each stage
is skipped when its artefacts already exist unless explicitly forced.

    python run.py                      # serve, reusing existing artefacts
    python run.py --all                # rebuild everything, then serve
    python run.py --reprocess --train  # rerun Spark and retrain, then serve
    python run.py --no-serve --all     # batch rebuild with no web server

The Windows JAVA_HOME and PATH handling that used to live at the top of this
file now sits in src/config.py, so the Spark jobs get the same treatment whether
they are launched from here or run standalone.
"""

import argparse
import logging
import shutil
import socket
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src import config  # noqa: E402

config.configure_spark_environment()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BankingAnalyticsRunner")


def stage_banner(name):
    logger.info("=" * 72)
    logger.info(name)
    logger.info("=" * 72)


def setup_data_files():
    """
    Make the raw CSVs available, copying them in from the parent directory if a
    previous checkout left them there.

    The dataset itself is not in the repository -- at 1.76 GB it exceeds
    GitHub's file limit -- so a fresh clone regenerates it with
    scripts/generate_dataset.py.
    """
    config.ensure_dirs()
    parent = BASE_DIR.parent
    for filename in (config.RAW_TRANSACTIONS_CSV.name, config.RAW_CUSTOMERS_CSV.name):
        target = config.RAW_DIR / filename
        source = parent / filename
        if not target.exists() and source.exists():
            logger.info(f"Copying {filename} ({source.stat().st_size / 1e9:.2f} GB) into data/raw ...")
            shutil.copy(source, target)


def find_free_port(preferred, attempts=10):
    """First free port at or after `preferred`."""
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    # Nothing free in the scanned window; say so rather than silently reusing a
    # busy port, which the previous implementation did.
    raise RuntimeError(
        f"No free port between {preferred} and {preferred + attempts - 1}. "
        f"Pass --port to choose another."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Banking Fraud Big Data Analytics & Customer Segmentation"
    )
    parser.add_argument("--reprocess", action="store_true", help="Force the PySpark pipeline to rerun")
    parser.add_argument("--train", action="store_true", help="Force fraud model retraining")
    parser.add_argument("--segment", action="store_true", help="Force K-Means customer segmentation")
    parser.add_argument("--anomalies", action="store_true", help="Force Isolation Forest anomaly detection")
    parser.add_argument("--all", action="store_true", help="Force every stage to rerun")
    parser.add_argument("--spark-ml", action="store_true",
                        help="Also train Spark MLlib models on ALL 15M rows (needs the raw CSV; "
                             "not included in --all because of its runtime)")
    parser.add_argument("--export-full", action="store_true",
                        help="Also write month-partitioned Parquet for all 15M rows")
    parser.add_argument("--no-serve", action="store_true", help="Run the pipeline without starting the web server")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation during training (faster)")
    parser.add_argument("--port", type=int, default=config.DEFAULT_PORT, help="Port for the Flask backend")
    parser.add_argument("--production", action="store_true",
                        help="Serve with waitress instead of the Flask development server")
    args = parser.parse_args()

    setup_data_files()

    force_all = args.all

    # --- Stage 1: PySpark preprocessing -----------------------------------
    if force_all or args.reprocess or not config.SUMMARY_STATS_JSON.exists():
        if not config.RAW_TRANSACTIONS_CSV.exists():
            logger.error(f"Raw dataset not found at {config.RAW_TRANSACTIONS_CSV}.")
            logger.error(
                "The 1.76 GB source file is too large for GitHub, so it is not in the "
                "repository. Regenerate a statistically faithful copy with:"
            )
            logger.error("    python scripts/generate_dataset.py")
            logger.error("Then rerun this command.")
            return 1
        stage_banner("STAGE 1/4  PySpark preprocessing (15M transactions, local[*], no Hadoop)")
        from src.preprocessing.preprocess import run_pyspark_preprocessing
        run_pyspark_preprocessing(export_full=args.export_full)
    else:
        logger.info("Stage 1/4  Spark artefacts present -- skipping (use --reprocess to force).")

    # --- Stage 2: Customer segmentation -----------------------------------
    if force_all or args.segment or not config.CUSTOMER_CLUSTERS_JSON.exists():
        stage_banner("STAGE 2/4  K-Means customer segmentation")
        from src.customer_segmentation.clustering import perform_customer_segmentation
        perform_customer_segmentation(customer_csv_path=config.RAW_CUSTOMERS_CSV)
    else:
        logger.info("Stage 2/4  Cluster artefacts present -- skipping (use --segment to force).")

    # --- Stage 3: Anomaly detection ---------------------------------------
    if force_all or args.anomalies or not config.ANOMALIES_JSON.exists():
        stage_banner("STAGE 3/4  Isolation Forest anomaly detection")
        from src.anomaly_detection.anomaly import perform_anomaly_detection
        perform_anomaly_detection(raw_csv_path=config.RAW_TRANSACTIONS_CSV)
    else:
        logger.info("Stage 3/4  Anomaly artefacts present -- skipping (use --anomalies to force).")

    # --- Stage 4: Fraud model benchmark -----------------------------------
    if force_all or args.train or not config.MODEL_COMPARISON_JSON.exists():
        stage_banner("STAGE 4/4  Fraud model benchmark + leakage audit")
        from src.fraud_detection.train_models import train_fraud_models
        train_fraud_models(run_cv=not args.no_cv)
    else:
        logger.info("Stage 4/4  Model artefacts present -- skipping (use --train to force).")

    # --- Optional stage: Spark MLlib on the complete dataset ---------------
    # The scikit-learn benchmark above trains on a 500k stratified sample because
    # scikit-learn is single-node. This stage trains distributed over all 15M
    # rows so the Big Data claim covers the machine learning, not just the
    # aggregation. It is opt-in because it re-reads the full 1.85 GB CSV.
    if args.spark_ml:
        if not config.RAW_TRANSACTIONS_CSV.exists():
            logger.error(
                f"--spark-ml needs the full dataset at {config.RAW_TRANSACTIONS_CSV}; "
                f"the sampled Parquet is not enough."
            )
            logger.error("Regenerate it with:  python scripts/generate_dataset.py")
            return 1
        stage_banner("STAGE 5/5  Spark MLlib training on ALL 15,000,000 rows")
        from src.fraud_detection.spark_ml import train_spark_models
        train_spark_models()

    if args.no_serve:
        logger.info("Pipeline complete. --no-serve set, exiting without starting the web server.")
        return 0

    # --- Serve -------------------------------------------------------------
    port = find_free_port(args.port)
    from backend.app import create_app
    app = create_app()

    if args.production:
        from waitress import serve
        logger.info(f"Serving with waitress on http://127.0.0.1:{port}/ ...")
        serve(app, host="0.0.0.0", port=port, threads=8)
    else:
        logger.info(f"Starting dashboard on http://127.0.0.1:{port}/ ...")
        logger.info("(development server -- use --production for a waitress WSGI server)")
        app.run(host="0.0.0.0", port=port, debug=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
