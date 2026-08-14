import os
import json
import joblib
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def perform_customer_segmentation(customer_csv_path, models_dir, processed_dir):
    """
    K-Means Customer Segmentation derived from 15M transactions customer profiles.
    Calculates Elbow Method (K=2..8), fits optimal K=4, and saves models and cluster statistics.
    """
    models_path = Path(models_dir)
    processed_path = Path(processed_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    cust_parquet_15m = processed_path / "customer_profiles_15m.parquet"
    cust_csv_15m = processed_path / "customer_profiles_15m.csv"

    if cust_parquet_15m.exists():
        logger.info(f"Loading 15M customer profile data from {cust_parquet_15m}...")
        df = pd.read_parquet(cust_parquet_15m)
    elif cust_csv_15m.exists():
        logger.info(f"Loading 15M customer profile data from {cust_csv_15m}...")
        df = pd.read_csv(cust_csv_15m)
    else:
        logger.info(f"Loading fallback customer profile data from {customer_csv_path}...")
        df = pd.read_csv(customer_csv_path)

    features = [
        'total_transaction_amount',
        'average_transaction_amount',
        'transaction_count',
        'average_balance_before',
        'unique_merchants',
        'fraud_count'
    ]

    # Verify all feature columns exist
    X = df[features].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Elbow Method Calculation
    k_range = range(2, 9)
    elbow_data = []
    
    # Subsample for fast elbow evaluation if customer count > 50,000
    if len(X_scaled) > 50000:
        np.random.seed(42)
        elbow_sample_idx = np.random.choice(len(X_scaled), size=50000, replace=False)
        X_elbow_eval = X_scaled[elbow_sample_idx]
    else:
        X_elbow_eval = X_scaled

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_elbow_eval)
        inertia = float(kmeans.inertia_)
        
        # Sample for silhouette score speed
        sil_sample_idx = np.random.choice(len(X_elbow_eval), size=min(5000, len(X_elbow_eval)), replace=False)
        sil = float(silhouette_score(X_elbow_eval[sil_sample_idx], kmeans.labels_[sil_sample_idx]))
        
        elbow_data.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil, 4)
        })

    # Pick optimal K = 4 for distinct behavioral customer segments
    optimal_k = 4
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans_optimal.fit_predict(X_scaled)

    # Save models (both standard and pipeline names)
    joblib.dump(kmeans_optimal, models_path / "kmeans_model.pkl")
    joblib.dump(kmeans_optimal, models_path / "kmeans_pipeline.pkl")
    joblib.dump(scaler, models_path / "kmeans_scaler.pkl")

    # 2. Cluster Statistical Profiling & Dynamic Labeling based on relative metric ranking
    cluster_stats = df.groupby('cluster')[features].mean().round(2).to_dict(orient='index')
    cluster_counts = df['cluster'].value_counts().to_dict()

    # Identify cluster IDs by metric ranks for unambiguous labeling
    highest_spend_cid = max(range(optimal_k), key=lambda c: cluster_stats[c]['average_transaction_amount'])
    remaining_cids = [c for c in range(optimal_k) if c != highest_spend_cid]

    highest_fraud_cid = max(remaining_cids, key=lambda c: cluster_stats[c]['fraud_count'])
    remaining_cids = [c for c in remaining_cids if c != highest_fraud_cid]

    highest_bal_cid = max(remaining_cids, key=lambda c: cluster_stats[c]['average_balance_before'])
    remaining_cids = [c for c in remaining_cids if c != highest_bal_cid]
    standard_cid = remaining_cids[0]

    label_map = {
        highest_spend_cid: "High Spending / VIP Customers",
        highest_fraud_cid: "High Risk / Fraud Prone Customers",
        highest_bal_cid: "High Balance Customers",
        standard_cid: "Standard Retail Customers"
    }

    cluster_profiles = []
    for cid in range(optimal_k):
        stats = cluster_stats[cid]
        count = cluster_counts[cid]
        pct = round((count / len(df)) * 100, 2)
        label = label_map[cid]

        cluster_profiles.append({
            "cluster_id": int(cid),
            "label": label,
            "count": int(count),
            "percentage": pct,
            "stats": stats
        })

    payload = {
        "elbow_curve": elbow_data,
        "optimal_k": optimal_k,
        "total_customers": len(df),
        "cluster_profiles": cluster_profiles
    }

    with open(processed_path / "customer_clusters.json", "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Customer segmentation completed for {len(df):,} customers across {optimal_k} clusters.")
    return payload

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    cust_csv = base_dir / "data" / "raw" / "customer_profiles.csv"
    models_dir = base_dir / "models"
    processed_dir = base_dir / "data" / "processed"
    perform_customer_segmentation(str(cust_csv), str(models_dir), str(processed_dir))
