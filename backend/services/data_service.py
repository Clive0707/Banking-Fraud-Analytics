import os
import json
import joblib
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataService:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
        else:
            base_dir = Path(base_dir)

        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self.models_dir = base_dir / "models"

        # Cached memory objects
        self.summary_stats = None
        self.fraud_aggregates = None
        self.customer_clusters = None
        self.anomalies_summary = None
        self.model_comparison = None
        self.transactions_df = None

        # Loaded ML models & artifacts
        self.scaler = None
        self.encoders = None
        self.feature_cols = None
        self.models = {}
        self.best_model_name = None

        self.load_cache()

    def load_cache(self):
        """Loads precomputed JSON data, ML models, and fast sample transactions into memory."""
        logger.info("Loading cached analytics & ML models into memory...")

        # 1. Load JSON Summary Data
        summary_file = self.processed_dir / "summary_stats.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                self.summary_stats = json.load(f)

        fraud_file = self.processed_dir / "fraud_aggregates.json"
        if fraud_file.exists():
            with open(fraud_file, "r") as f:
                self.fraud_aggregates = json.load(f)

        clusters_file = self.processed_dir / "customer_clusters.json"
        if clusters_file.exists():
            with open(clusters_file, "r") as f:
                self.customer_clusters = json.load(f)

        anomalies_file = self.processed_dir / "anomalies_summary.json"
        if anomalies_file.exists():
            with open(anomalies_file, "r") as f:
                self.anomalies_summary = json.load(f)

        comp_file = self.models_dir / "model_comparison.json"
        if comp_file.exists():
            with open(comp_file, "r") as f:
                self.model_comparison = json.load(f)
                self.best_model_name = self.model_comparison.get("best_model", "Random Forest")

        # 2. Load ML Artifacts
        scaler_path = self.models_dir / "scaler.pkl"
        encoders_path = self.models_dir / "encoders.pkl"
        features_path = self.models_dir / "feature_cols.pkl"

        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        if encoders_path.exists():
            self.encoders = joblib.load(encoders_path)
        if features_path.exists():
            self.feature_cols = joblib.load(features_path)

        # Load trained classifiers (checking both primary and pipeline filename conventions)
        for name, primary_file, pipeline_file in [
            ("Logistic Regression", "logistic_regression.pkl", "logistic_regression_pipeline.pkl"),
            ("CART Decision Tree", "cart_decision_tree.pkl", "cart_decision_tree_pipeline.pkl"),
            ("Random Forest", "random_forest.pkl", "random_forest_pipeline.pkl")
        ]:
            path = self.models_dir / primary_file
            if not path.exists():
                path = self.models_dir / pipeline_file

            if path.exists():
                self.models[name] = joblib.load(path)
                logger.info(f"Loaded classifier: {name} from {path.name}")

        # 3. Load sample transaction dataset for Explorer pagination (memory optimized)
        sample_parquet = self.processed_dir / "transactions_sample.parquet"
        processed_parquet = self.processed_dir / "transactions_processed.parquet"

        if sample_parquet.exists():
            logger.info(f"Loading transaction dataset from {sample_parquet} for instant server-side pagination...")
            self.transactions_df = pd.read_parquet(sample_parquet)
        elif processed_parquet.exists():
            logger.info(f"Loading transaction dataset from {processed_parquet}...")
            self.transactions_df = pd.read_parquet(processed_parquet)
        else:
            logger.warning("Processed transaction parquet file not found!")

    def get_summary(self):
        if not self.summary_stats:
            return {"error": "Summary stats not found. Please run PySpark preprocessing first."}
        
        summary = self.summary_stats.copy()
        if self.anomalies_summary:
            summary["detected_anomalies"] = self.anomalies_summary.get("total_anomalies", 0)
        return summary

    def get_fraud_analytics(self):
        if not self.fraud_aggregates:
            return {"error": "Fraud aggregates not found."}
        
        suspicious = []
        if self.transactions_df is not None:
            susp_df = self.transactions_df[self.transactions_df['is_fraud'] == 1].head(20)
            for _, row in susp_df.iterrows():
                suspicious.append({
                    "transaction_id": str(row['transaction_id']),
                    "customer_id": int(row['customer_id']),
                    "transaction_date": str(row['transaction_date']),
                    "transaction_time": str(row['transaction_time']),
                    "transaction_type": str(row['transaction_type']),
                    "amount": round(float(row['amount']), 2),
                    "merchant": str(row['merchant']),
                    "location": str(row['location']),
                    "payment_method": str(row['payment_method']),
                    "device_type": str(row['device_type']),
                    "status": str(row['transaction_status'])
                })

        result = self.fraud_aggregates.copy()
        result["suspicious_transactions"] = suspicious
        return result

    def get_fraud_trends(self):
        if not self.fraud_aggregates:
            return {"error": "Fraud aggregates not found."}
        return {
            "monthly": self.fraud_aggregates.get("monthly_trends", []),
            "daily": self.fraud_aggregates.get("daily_trends", [])
        }

    def get_customer_clusters(self):
        if not self.customer_clusters:
            return {"error": "Customer clusters data not found."}
        return self.customer_clusters

    def get_anomalies(self):
        if not self.anomalies_summary:
            return {"error": "Anomalies summary data not found."}
        return self.anomalies_summary

    def get_model_performance(self):
        if not self.model_comparison:
            return {"error": "Model comparison data not found."}
        return self.model_comparison

    def get_transactions(self, page=1, per_page=25, customer_id=None, transaction_type=None,
                         payment_method=None, location=None, is_fraud=None, min_amount=None, max_amount=None, search=None):
        if self.transactions_df is None:
            return {"error": "Transaction data not loaded."}

        df_filtered = self.transactions_df

        # Apply Filters
        if customer_id:
            df_filtered = df_filtered[df_filtered['customer_id'] == int(customer_id)]
        if transaction_type:
            df_filtered = df_filtered[df_filtered['transaction_type'].str.lower() == transaction_type.lower()]
        if payment_method:
            df_filtered = df_filtered[df_filtered['payment_method'].str.lower() == payment_method.lower()]
        if location:
            df_filtered = df_filtered[df_filtered['location'].str.lower() == location.lower()]
        if is_fraud is not None and is_fraud != '':
            df_filtered = df_filtered[df_filtered['is_fraud'] == int(is_fraud)]
        if min_amount is not None:
            df_filtered = df_filtered[df_filtered['amount'] >= float(min_amount)]
        if max_amount is not None:
            df_filtered = df_filtered[df_filtered['amount'] <= float(max_amount)]

        if search:
            search_str = str(search).lower()
            df_filtered = df_filtered[
                df_filtered['transaction_id'].str.lower().str.contains(search_str) |
                df_filtered['merchant'].str.lower().str.contains(search_str) |
                df_filtered['customer_id'].astype(str).str.contains(search_str)
            ]

        total_records = len(df_filtered)
        total_pages = max(1, (total_records + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_df = df_filtered.iloc[start_idx:end_idx]

        records = []
        for _, row in page_df.iterrows():
            records.append({
                "transaction_id": str(row['transaction_id']),
                "customer_id": int(row['customer_id']),
                "transaction_date": str(row['transaction_date']),
                "transaction_time": str(row['transaction_time']),
                "transaction_type": str(row['transaction_type']),
                "account_type": str(row['account_type']),
                "amount": round(float(row['amount']), 2),
                "balance_before": round(float(row['balance_before']), 2),
                "balance_after": round(float(row['balance_after']), 2),
                "merchant": str(row['merchant']),
                "location": str(row['location']),
                "payment_method": str(row['payment_method']),
                "device_type": str(row['device_type']),
                "transaction_status": str(row['transaction_status']),
                "is_fraud": int(row['is_fraud'])
            })

        return {
            "total_records": total_records,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "data": records
        }

    def predict_fraud(self, input_data, model_name=None):
        """
        Executes single transaction fraud inference using the selected model.
        """
        if not self.models:
            return {"error": "ML models not loaded."}

        target_model_name = model_name if (model_name and model_name in self.models) else self.best_model_name
        model = self.models.get(target_model_name)

        if not model:
            return {"error": f"Model '{target_model_name}' not available."}

        try:
            amount = float(input_data.get('amount', 0.0))
            balance_before = float(input_data.get('balance_before', 0.0))
            balance_after = float(input_data.get('balance_after', 0.0))
            time_str = input_data.get('transaction_time', '12:00:00')
            
            try:
                hour = int(time_str.split(':')[0])
            except Exception:
                hour = 12

            date_str = input_data.get('transaction_date', '2026-01-01')
            try:
                dt = pd.to_datetime(date_str)
                day_of_week = dt.dayofweek
                day_of_month = dt.day
            except Exception:
                day_of_week = 0
                day_of_month = 1

            balance_change = balance_before - balance_after
            is_unusual_time = 1 if (hour < 5 or hour > 23) else 0
            zero_balance_after = 1 if balance_after == 0 else 0
            amount_to_balance_ratio = amount / (balance_before + 1.0) if balance_before > 0 else amount

            row_dict = {
                'amount': amount,
                'balance_before': balance_before,
                'balance_after': balance_after,
                'balance_change': balance_change,
                'hour': hour,
                'day_of_week': day_of_week,
                'day_of_month': day_of_month,
                'is_unusual_time': is_unusual_time,
                'zero_balance_after': zero_balance_after,
                'amount_to_balance_ratio': amount_to_balance_ratio
            }

            categorical_cols = ['transaction_type', 'account_type', 'payment_method', 'device_type', 'location']
            for col in categorical_cols:
                val = str(input_data.get(col, 'Unknown'))
                enc_col = col + '_encoded'
                if self.encoders and col in self.encoders:
                    le = self.encoders[col]
                    if val in le.classes_:
                        encoded_val = int(le.transform([val])[0])
                    else:
                        encoded_val = 0
                else:
                    encoded_val = 0
                row_dict[enc_col] = encoded_val

            feat_vector = []
            for col in self.feature_cols:
                feat_vector.append(row_dict.get(col, 0.0))

            X_input = pd.DataFrame([feat_vector], columns=self.feature_cols)

            if target_model_name == "Logistic Regression" and self.scaler:
                X_input_proc = self.scaler.transform(X_input)
            else:
                X_input_proc = X_input

            proba = float(model.predict_proba(X_input_proc)[:, 1][0])
            pred_class = int(model.predict(X_input_proc)[0])

            prediction_label = "Fraud" if (pred_class == 1 or proba >= 0.5) else "Legitimate"

            if proba > 0.70:
                risk_level = "High"
            elif proba > 0.35:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            return {
                "prediction": prediction_label,
                "fraud_probability": round(proba, 4),
                "fraud_probability_pct": f"{round(proba * 100, 2)}%",
                "risk_level": risk_level,
                "model_used": target_model_name
            }

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"error": f"Inference error: {str(e)}"}
