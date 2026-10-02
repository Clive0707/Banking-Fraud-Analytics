"""
Unsupervised anomaly detection with Isolation Forest.

Changes from the original implementation
---------------------------------------
1. **The detector is now validated against the fraud labels.** The original
   computed anomaly scores and reported how many outliers it found, but never
   checked whether those outliers had anything to do with fraud -- so there was
   no way to tell whether the model was useful or just flagging large amounts.
   Precision, recall and lift against `is_fraud` are now reported. The labels are
   used strictly for evaluation; the model never sees them.

2. **Features come from the shared builder**, so "anomalous" means the same thing
   here as it does to the supervised models.

3. **Score percentiles** accompany the histogram, so a threshold can be chosen
   rather than relying on the `contamination` default.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src import config
from src import features as feat_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def perform_anomaly_detection(raw_csv_path=None, models_dir=None, processed_dir=None):
    # Accept str or Path from any caller.
    models_path = Path(models_dir) if models_dir else config.MODELS_DIR
    processed_path = Path(processed_dir) if processed_dir else config.PROCESSED_DIR
    models_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    # --- Load -------------------------------------------------------------
    if config.TRANSACTIONS_SAMPLE_PARQUET.exists():
        logger.info(f"Loading sample from {config.TRANSACTIONS_SAMPLE_PARQUET.name}...")
        df = pd.read_parquet(config.TRANSACTIONS_SAMPLE_PARQUET)
    elif raw_csv_path:
        logger.info(f"Reading {config.ML_SAMPLE_TARGET:,} rows from {raw_csv_path}...")
        df = pd.read_csv(raw_csv_path, nrows=config.ML_SAMPLE_TARGET)
    else:
        raise FileNotFoundError("No transaction data available for anomaly detection.")

    summary = {}
    if config.SUMMARY_STATS_JSON.exists():
        with open(config.SUMMARY_STATS_JSON) as f:
            summary = json.load(f)
    population_total = summary.get("total_transactions", len(df))

    # --- Features ---------------------------------------------------------
    feat = feat_mod.build_features(df)
    feature_cols = [c for c in config.ISOLATION_FEATURES if c in feat.columns]
    X = feat[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Fit --------------------------------------------------------------
    logger.info(
        f"Fitting Isolation Forest on {len(X):,} rows, {len(feature_cols)} features, "
        f"contamination={config.ISOLATION_CONTAMINATION}..."
    )
    iso = IsolationForest(
        n_estimators=200,
        contamination=config.ISOLATION_CONTAMINATION,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    # The target is never passed to fit -- this is genuinely unsupervised.
    df["anomaly_label"] = iso.fit_predict(X_scaled)        # -1 anomaly, 1 normal
    df["anomaly_score"] = iso.decision_function(X_scaled)  # lower = more anomalous

    joblib.dump(iso, models_path / "isolation_forest_pipeline.pkl", compress=3)
    joblib.dump(scaler, models_path / "isolation_scaler.pkl", compress=3)
    joblib.dump(feature_cols, models_path / "isolation_features.pkl", compress=3)

    anomalies = df[df["anomaly_label"] == -1]
    n_anom = len(anomalies)
    anomaly_pct = n_anom / len(df) * 100
    estimated_total = int(round(population_total * n_anom / len(df)))

    # --- Validate against the fraud labels --------------------------------
    validation = None
    if config.TARGET_COL in df.columns:
        y = df[config.TARGET_COL].astype(int)
        baseline = float(y.mean())
        tp = int(anomalies[config.TARGET_COL].sum())
        total_fraud = int(y.sum())
        precision = tp / n_anom if n_anom else 0.0

        # How well the continuous score ranks fraud, independent of the
        # contamination cut. Scores are inverted because lower means more
        # anomalous.
        from sklearn.metrics import average_precision_score, roc_auc_score
        inverted = -df["anomaly_score"].values

        validation = {
            "note": (
                "Isolation Forest is trained without labels. is_fraud is used here "
                "only to measure whether the structural outliers it finds "
                "correspond to actual fraud."
            ),
            "baseline_fraud_rate_pct": round(baseline * 100, 4),
            "flagged_transactions": n_anom,
            "frauds_among_flagged": tp,
            "precision": round(precision, 6),
            "recall": round(tp / total_fraud, 6) if total_fraud else 0.0,
            "lift_vs_baseline": round(precision / baseline, 3) if baseline else 0.0,
            "roc_auc": round(float(roc_auc_score(y, inverted)), 6),
            "pr_auc": round(float(average_precision_score(y, inverted)), 6),
        }
        logger.info(
            f"Validation against labels: precision={validation['precision']:.4f} "
            f"recall={validation['recall']:.4f} lift={validation['lift_vs_baseline']}x "
            f"ROC-AUC={validation['roc_auc']:.4f}"
        )

    # --- Score distribution ----------------------------------------------
    scores = df["anomaly_score"].values
    counts, edges = np.histogram(scores, bins=20)
    distribution = [
        {
            "bin_start": round(float(edges[i]), 4),
            "bin_end": round(float(edges[i + 1]), 4),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]
    percentiles = {
        f"p{p}": round(float(np.percentile(scores, p)), 4)
        for p in (0.1, 1, 2.5, 5, 10, 25, 50, 75, 90, 99)
    }

    # --- Top outliers -----------------------------------------------------
    top = anomalies.nsmallest(50, "anomaly_score")
    records = []
    for _, row in top.iterrows():
        records.append({
            "transaction_id": str(row["transaction_id"]),
            "customer_id": int(row["customer_id"]),
            "amount": round(float(row["amount"]), 2),
            "balance_before": round(float(row["balance_before"]), 2),
            "balance_after": round(float(row["balance_after"]), 2),
            "transaction_type": str(row["transaction_type"]),
            "payment_method": str(row["payment_method"]),
            "location": str(row["location"]),
            "transaction_date": str(row["transaction_date"]),
            "transaction_time": str(row["transaction_time"]),
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "is_fraud": int(row[config.TARGET_COL]) if config.TARGET_COL in row.index else None,
        })

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "algorithm": "Isolation Forest (unsupervised)",
        "features_used": feature_cols,
        "contamination": config.ISOLATION_CONTAMINATION,
        "rows_scored": int(len(df)),
        "total_transactions": int(population_total),
        "anomalies_in_sample": n_anom,
        "anomaly_percentage": round(anomaly_pct, 4),
        "total_anomalies": estimated_total,
        "extrapolation_note": (
            f"{n_anom:,} outliers in a {len(df):,}-row stratified sample, scaled to "
            f"the {population_total:,}-row population."
        ),
        "score_distribution": distribution,
        "score_percentiles": percentiles,
        "label_validation": validation,
        "top_suspicious": records,
    }

    with open(config.ANOMALIES_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"Anomaly detection complete: {n_anom:,} outliers ({anomaly_pct:.2f}%), "
        f"~{estimated_total:,} projected across the population."
    )
    return payload


if __name__ == "__main__":
    perform_anomaly_detection(raw_csv_path=config.RAW_TRANSACTIONS_CSV)
