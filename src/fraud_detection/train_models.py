import os
import json
import joblib
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_features(df):
    """
    Feature engineering for fraud detection.
    Extracts temporal components, financial deviations, risk ratios, and encodes categoricals.
    Ensures is_fraud is NEVER included in feature matrix X.
    """
    df = df.copy()

    # Temporal features
    if 'transaction_time' in df.columns:
        df['hour'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S', errors='coerce').dt.hour.fillna(12)
    else:
        df['hour'] = 12

    if 'transaction_date' in df.columns:
        date_series = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['day_of_week'] = date_series.dt.dayofweek.fillna(0)
        df['day_of_month'] = date_series.dt.day.fillna(1)
    else:
        df['day_of_week'] = 0
        df['day_of_month'] = 1

    # Financial & Risk indicators
    df['balance_change'] = df['balance_before'] - df['balance_after']
    df['is_unusual_time'] = df['hour'].apply(lambda h: 1 if (h < 5 or h > 23) else 0)
    df['zero_balance_after'] = (df['balance_after'] == 0).astype(int)

    # Balance ratio
    df['amount_to_balance_ratio'] = np.where(
        df['balance_before'] > 0,
        df['amount'] / (df['balance_before'] + 1.0),
        df['amount']
    )

    return df

def train_fraud_models(raw_csv_path, models_dir):
    """
    Trains Logistic Regression, CART Decision Tree, and Random Forest on the 15M dataset (using stratified sample).
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    base_dir = models_path.parent

    sample_parquet = base_dir / "data" / "processed" / "transactions_sample.parquet"
    if sample_parquet.exists():
        logger.info(f"Loading stratified 15M transaction sample from {sample_parquet}...")
        df = pd.read_parquet(sample_parquet)
    else:
        logger.info(f"Sample parquet not found. Reading from raw dataset {raw_csv_path}...")
        # Read sample of 500,000 rows safely
        df = pd.read_csv(raw_csv_path, nrows=500000)

    sample_size = len(df)
    logger.info(f"Training dataset loaded: {sample_size:,} records.")

    logger.info("Extracting features...")
    df_feat = extract_features(df)

    categorical_cols = ['transaction_type', 'account_type', 'payment_method', 'device_type', 'location']
    feature_cols = [
        'amount', 'balance_before', 'balance_after', 'balance_change',
        'hour', 'day_of_week', 'day_of_month', 'is_unusual_time',
        'zero_balance_after', 'amount_to_balance_ratio'
    ]

    encoders = {}
    for col in categorical_cols:
        if col in df_feat.columns:
            le = LabelEncoder()
            df_feat[col + '_encoded'] = le.fit_transform(df_feat[col].astype(str))
            encoders[col] = le
            feature_cols.append(col + '_encoded')

    X = df_feat[feature_cols]
    y = df_feat['is_fraud'].astype(int)

    logger.info(f"Feature matrix shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    # Train/Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save Scaler & Encoders
    joblib.dump(scaler, models_path / "scaler.pkl")
    joblib.dump(encoders, models_path / "encoders.pkl")
    joblib.dump(feature_cols, models_path / "feature_cols.pkl")

    # Models definition
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "CART Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1)
    }

    comparison_results = {}
    model_filename_map = {
        "Logistic Regression": ("logistic_regression.pkl", "logistic_regression_pipeline.pkl"),
        "CART Decision Tree": ("cart_decision_tree.pkl", "cart_decision_tree_pipeline.pkl"),
        "Random Forest": ("random_forest.pkl", "random_forest_pipeline.pkl")
    }

    best_model_name = None
    best_f1 = -1

    for name, model in models.items():
        logger.info(f"Training {name} on 15M dataset sample ({len(X_train):,} train rows)...")
        
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]

        primary_file, pipeline_file = model_filename_map[name]
        comparison_results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "filename": primary_file,
            "pipeline_filename": pipeline_file
        }

        # Save model pickle files (both standard and _pipeline names)
        joblib.dump(model, models_path / primary_file)
        joblib.dump(model, models_path / pipeline_file)

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    comparison_payload = {
        "models": comparison_results,
        "best_model": best_model_name,
        "dataset_total_records": 15000000,
        "training_sample_size": sample_size,
        "test_samples": len(y_test)
    }

    with open(models_path / "model_comparison.json", "w") as f:
        json.dump(comparison_payload, f, indent=2)

    logger.info(f"Model retraining complete. Best model: {best_model_name} (F1: {best_f1:.4f})")
    return comparison_payload

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    raw_csv = base_dir / "data" / "raw" / "banking_transactions_15m.csv"
    models_dir = base_dir / "models"
    train_fraud_models(str(raw_csv), str(models_dir))
