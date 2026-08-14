import os
import json
import joblib
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def perform_anomaly_detection(raw_csv_path, models_dir, processed_dir):
    """
    Unsupervised Anomaly Detection using Isolation Forest on 15M banking transactions (sample partition).
    Calculates anomaly scores, score distribution, top 50 structural outliers, and saves model pickles.
    """
    models_path = Path(models_dir)
    processed_path = Path(processed_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    base_dir = models_path.parent
    sample_parquet = base_dir / "data" / "processed" / "transactions_sample.parquet"

    if sample_parquet.exists():
        logger.info(f"Loading transaction sample from {sample_parquet} for anomaly detection...")
        df = pd.read_parquet(sample_parquet)
    else:
        logger.info(f"Sample parquet not found. Reading from raw dataset {raw_csv_path}...")
        df = pd.read_csv(raw_csv_path, nrows=500000)

    # Summary stats for total 15M transaction count
    summary_file = processed_path / "summary_stats.json"
    total_15m_txns = len(df)
    if summary_file.exists():
        with open(summary_file, "r") as f:
            stats = json.load(f)
            total_15m_txns = stats.get("total_transactions", len(df))

    # Feature Engineering for Anomaly Detection
    df['hour'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S', errors='coerce').dt.hour.fillna(12)
    df['balance_change'] = df['balance_before'] - df['balance_after']
    df['is_unusual_time'] = df['hour'].apply(lambda h: 1 if (h < 5 or h > 23) else 0)
    df['amount_to_balance_ratio'] = np.where(
        df['balance_before'] > 0,
        df['amount'] / (df['balance_before'] + 1.0),
        df['amount']
    )

    feature_cols = ['amount', 'balance_before', 'balance_after', 'balance_change', 'hour', 'is_unusual_time', 'amount_to_balance_ratio']
    X = df[feature_cols].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Fitting Isolation Forest model...")
    # Contamination set to 2.5% to isolate severe structural outliers
    iso = IsolationForest(n_estimators=100, contamination=0.025, random_state=42, n_jobs=-1)
    df['anomaly_label'] = iso.fit_predict(X_scaled)  # -1 for anomaly, 1 for normal
    df['anomaly_score'] = iso.decision_function(X_scaled)  # Lower score = more anomalous

    # Save trained model and scaler (both standard and pipeline names)
    joblib.dump(iso, models_path / "isolation_forest.pkl")
    joblib.dump(iso, models_path / "isolation_forest_pipeline.pkl")
    joblib.dump(scaler, models_path / "isolation_scaler.pkl")

    anomalies_df = df[df['anomaly_label'] == -1]
    sample_anomalies = len(anomalies_df)
    anomaly_percentage = round((sample_anomalies / len(df)) * 100, 2)
    estimated_total_anomalies = int(round(total_15m_txns * (sample_anomalies / len(df))))

    # Score Distribution Histogram Bins (10 bins)
    scores = df['anomaly_score']
    counts, bin_edges = np.histogram(scores, bins=10)
    distribution = []
    for i in range(len(counts)):
        distribution.append({
            "bin_start": round(float(bin_edges[i]), 3),
            "bin_end": round(float(bin_edges[i+1]), 3),
            "count": int(counts[i])
        })

    # Extract Top 50 Most Suspicious Anomalies (lowest anomaly score)
    top_suspicious = anomalies_df.sort_values(by='anomaly_score', ascending=True).head(50)
    suspicious_records = []
    for _, row in top_suspicious.iterrows():
        suspicious_records.append({
            "transaction_id": str(row['transaction_id']),
            "customer_id": int(row['customer_id']),
            "amount": round(float(row['amount']), 2),
            "transaction_type": str(row['transaction_type']),
            "payment_method": str(row['payment_method']),
            "location": str(row['location']),
            "transaction_date": str(row['transaction_date']),
            "transaction_time": str(row['transaction_time']),
            "anomaly_score": round(float(row['anomaly_score']), 4),
            "is_fraud": int(row['is_fraud'])
        })

    payload = {
        "total_transactions": total_15m_txns,
        "total_anomalies": estimated_total_anomalies,
        "anomaly_percentage": anomaly_percentage,
        "score_distribution": distribution,
        "top_suspicious": suspicious_records,
        "note": "Isolation Forest is Unsupervised Anomaly Detection (identifies structural/behavioral outliers without label supervision)."
    }

    with open(processed_path / "anomalies_summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Anomaly detection completed successfully.")
    return payload

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    raw_csv = base_dir / "data" / "raw" / "banking_transactions_15m.csv"
    models_dir = base_dir / "models"
    processed_dir = base_dir / "data" / "processed"
    perform_anomaly_detection(str(raw_csv), str(models_dir), str(processed_dir))
