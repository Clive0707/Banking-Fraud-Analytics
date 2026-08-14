import os
import sys

# Sanitize PATH and set JAVA_HOME for PySpark local execution on Windows
if os.path.exists(r"C:\Program Files\Java\jdk-22"):
    os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-22"

clean_paths = []
for p in os.environ.get("PATH", "").split(os.pathsep):
    p_clean = p.replace('"', '').strip()
    if p_clean and "msi" not in p_clean.lower():
        clean_paths.append(p_clean)
os.environ["PATH"] = os.pathsep.join(clean_paths)

import shutil
import logging
import argparse
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BankingAnalyticsMasterRunner")

def setup_data_files():
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parent_dir = base_dir.parent

    for filename in ["banking_transactions_15m.csv", "customer_profiles.csv"]:
        target = raw_dir / filename
        if not target.exists():
            source = parent_dir / filename
            if source.exists():
                logger.info(f"Copying {filename} from {source} to {target}...")
                shutil.copy(source, target)

def main():
    parser = argparse.ArgumentParser(description="Banking Fraud Big Data Analytics & Customer Segmentation")
    parser.add_argument("--reprocess", action="store_true", help="Force re-running PySpark data processing")
    parser.add_argument("--train", action="store_true", help="Force re-training ML models")
    parser.add_argument("--port", type=int, default=5000, help="Port to run Flask backend")
    args = parser.parse_args()

    setup_data_files()

    summary_json = base_dir / "data" / "processed" / "summary_stats.json"
    comp_json = base_dir / "models" / "model_comparison.json"

    # Step 1: Run PySpark preprocessing if needed
    if args.reprocess or not summary_json.exists():
        logger.info("Running PySpark Processing Pipeline...")
        from scripts.run_pyspark_pipeline import prepare_and_run
        prepare_and_run()
    else:
        logger.info("PySpark processed cache found. Skipping PySpark preprocessing (use --reprocess to force).")

    # Step 2: Run ML Model Training if needed
    if args.train or not comp_json.exists():
        logger.info("Running ML Model Training Pipeline...")
        from scripts.train_all_models import main as train_models_main
        train_models_main()
    else:
        logger.info("Trained ML models found. Skipping training (use --train to force).")

    # Step 3: Launch Flask Backend
    import socket
    port = args.port
    for p in range(port, port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                port = p
                break

    logger.info(f"Starting Flask Backend on http://127.0.0.1:{port}...")
    from backend.app import create_app
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    main()
