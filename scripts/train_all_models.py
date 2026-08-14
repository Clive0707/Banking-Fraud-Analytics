import sys
import logging
from pathlib import Path

# Add src to python path
base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

from src.fraud_detection.train_models import train_fraud_models
from src.customer_segmentation.clustering import perform_customer_segmentation
from src.anomaly_detection.anomaly import perform_anomaly_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainAllModelsRunner")

def main():
    raw_txns = base_dir / "data" / "raw" / "banking_transactions_15m.csv"
    raw_cust = base_dir / "data" / "raw" / "customer_profiles.csv"
    models_dir = base_dir / "models"
    processed_dir = base_dir / "data" / "processed"

    if not raw_txns.exists():
        logger.error("Raw CSV file missing in data/raw! Run run_pyspark_pipeline.py first.")
        sys.exit(1)

    logger.info("==========================================")
    logger.info("1. TRAINING FRAUD CLASSIFICATION MODELS")
    logger.info("==========================================")
    train_fraud_models(str(raw_txns), str(models_dir))

    logger.info("==========================================")
    logger.info("2. PERFORMING K-MEANS CUSTOMER SEGMENTATION")
    logger.info("==========================================")
    perform_customer_segmentation(str(raw_cust), str(models_dir), str(processed_dir))

    logger.info("==========================================")
    logger.info("3. PERFORMING ISOLATION FOREST ANOMALY DETECTION")
    logger.info("==========================================")
    perform_anomaly_detection(str(raw_txns), str(models_dir), str(processed_dir))

    logger.info("All Machine Learning models trained and artifacts generated successfully!")

if __name__ == "__main__":
    main()
