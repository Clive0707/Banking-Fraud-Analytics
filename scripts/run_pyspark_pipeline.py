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
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

from src.preprocessing.preprocess import run_pyspark_preprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PySparkPipelineRunner")

def prepare_and_run():
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    parent_dir = base_dir.parent

    # Copy raw dataset if not present in data/raw
    for filename in ["banking_transactions_15m.csv", "customer_profiles.csv"]:
        target = raw_dir / filename
        if not target.exists():
            source = parent_dir / filename
            if source.exists():
                logger.info(f"Copying {filename} from {source} to {target}...")
                shutil.copy(source, target)
            else:
                logger.warning(f"Source raw data file {source} not found!")

    raw_csv = raw_dir / "banking_transactions_15m.csv"
    if not raw_csv.exists():
        logger.error(f"Cannot run PySpark pipeline. Raw file {raw_csv} missing!")
        sys.exit(1)

    logger.info("Executing PySpark Big Data Processing Pipeline...")
    run_pyspark_preprocessing(str(raw_csv), str(processed_dir))
    logger.info("PySpark pipeline step completed.")

if __name__ == "__main__":
    prepare_and_run()
