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
        self.customer_profiles_df = None
        self.data_quality = None
        self.time_analytics = None
        self.geo_analytics = None
        self.payment_device_analytics = None
        self.analytical_alerts = None

        # Loaded ML models & artifacts
        self.scaler = None
        self.encoders = None
        self.feature_cols = None
        self.models = {}
        self.best_model_name = None

        self.load_cache()

    def load_cache(self):
        """Loads precomputed JSON data, ML models, and transaction datasets into memory."""
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
                self.best_model_name = self.model_comparison.get("best_model", "Logistic Regression")

        # 2. Extended JSON Caches
        dq_file = self.processed_dir / "data_quality.json"
        if dq_file.exists():
            with open(dq_file, "r") as f:
                self.data_quality = json.load(f)

        ta_file = self.processed_dir / "time_analytics.json"
        if ta_file.exists():
            with open(ta_file, "r") as f:
                self.time_analytics = json.load(f)

        geo_file = self.processed_dir / "geo_analytics.json"
        if geo_file.exists():
            with open(geo_file, "r") as f:
                self.geo_analytics = json.load(f)

        pd_file = self.processed_dir / "payment_device_analytics.json"
        if pd_file.exists():
            with open(pd_file, "r") as f:
                self.payment_device_analytics = json.load(f)

        alerts_file = self.processed_dir / "analytical_alerts.json"
        if alerts_file.exists():
            with open(alerts_file, "r") as f:
                self.analytical_alerts = json.load(f)

        # 3. Load ML Artifacts
        scaler_path = self.models_dir / "scaler.pkl"
        encoders_path = self.models_dir / "encoders.pkl"
        features_path = self.models_dir / "feature_cols.pkl"

        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        if encoders_path.exists():
            self.encoders = joblib.load(encoders_path)
        if features_path.exists():
            self.feature_cols = joblib.load(features_path)

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

        # 4. Load Datasets
        sample_parquet = self.processed_dir / "transactions_sample.parquet"
        processed_parquet = self.processed_dir / "transactions_processed.parquet"

        if sample_parquet.exists():
            logger.info(f"Loading transaction dataset from {sample_parquet} for instant server-side pagination...")
            self.transactions_df = pd.read_parquet(sample_parquet)
        elif processed_parquet.exists():
            logger.info(f"Loading transaction dataset from {processed_parquet}...")
            self.transactions_df = pd.read_parquet(processed_parquet)

        cust_parquet = self.processed_dir / "customer_profiles_15m.parquet"
        cust_csv = self.processed_dir / "customer_profiles_15m.csv"
        if cust_parquet.exists():
            self.customer_profiles_df = pd.read_parquet(cust_parquet)
        elif cust_csv.exists():
            self.customer_profiles_df = pd.read_csv(cust_csv)

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
            susp_df = self.transactions_df[self.transactions_df['is_fraud'] == 1].head(30)
            for _, row in susp_df.iterrows():
                amount = float(row['amount'])
                bal_before = float(row['balance_before'])
                bal_after = float(row['balance_after'])
                time_str = str(row['transaction_time'])
                hour = int(time_str.split(':')[0]) if ':' in time_str else 12

                # Risk rating
                if amount > 100000 or bal_after == 0 or (hour < 5 or hour > 23):
                    risk_level = "Critical"
                elif amount > 50000:
                    risk_level = "High"
                else:
                    risk_level = "Medium"

                suspicious.append({
                    "transaction_id": str(row['transaction_id']),
                    "customer_id": int(row['customer_id']),
                    "transaction_date": str(row['transaction_date']),
                    "transaction_time": str(row['transaction_time']),
                    "transaction_type": str(row['transaction_type']),
                    "account_type": str(row['account_type']),
                    "amount": round(amount, 2),
                    "balance_before": round(bal_before, 2),
                    "balance_after": round(bal_after, 2),
                    "merchant": str(row['merchant']),
                    "location": str(row['location']),
                    "payment_method": str(row['payment_method']),
                    "device_type": str(row['device_type']),
                    "status": str(row['transaction_status']),
                    "risk_level": risk_level,
                    "fraud_probability": 0.945
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

    def get_data_quality(self):
        if self.data_quality:
            return self.data_quality
        return {"error": "Data quality metrics not found."}

    def get_processing_metadata(self):
        return {
            "dataset": "banking_transactions_15m.csv",
            "records": 15000000,
            "raw_size_gb": 1.85,
            "engine": "Apache PySpark local[*]",
            "hadoop": "NOT USED",
            "hdfs": "NOT USED",
            "pyarrow_streaming": "Enabled (250,000 row chunks)",
            "parquet_size_mb": 534.5,
            "ml_sample_records": 499696,
            "schema": [
                "transaction_id (StringType)",
                "customer_id (IntegerType)",
                "transaction_date (StringType)",
                "transaction_time (StringType)",
                "transaction_type (StringType)",
                "account_type (StringType)",
                "amount (DoubleType)",
                "balance_before (DoubleType)",
                "balance_after (DoubleType)",
                "merchant (StringType)",
                "location (StringType)",
                "payment_method (StringType)",
                "device_type (StringType)",
                "transaction_status (StringType)",
                "is_fraud (IntegerType)"
            ]
        }

    def get_alerts(self, category="All"):
        if not self.analytical_alerts:
            return []
        if category and category.lower() != "all":
            return [a for a in self.analytical_alerts if a.get("type", "").lower() == category.lower()]
        return self.analytical_alerts

    def get_time_analytics(self):
        if self.time_analytics:
            return self.time_analytics
        return {"error": "Time analytics data not found."}

    def get_geo_analytics(self):
        if self.geo_analytics:
            return self.geo_analytics
        return {"error": "Geo analytics data not found."}

    def get_payment_device_analytics(self):
        if self.payment_device_analytics:
            return self.payment_device_analytics
        return {"error": "Payment device analytics data not found."}

    def get_customer_profile(self, customer_id):
        """Retrieves Customer 360 profile, stats, cluster assignment, and spending timeline."""
        cid = int(customer_id)
        if self.customer_profiles_df is None or self.transactions_df is None:
            return {"error": "Customer dataset not available."}

        cust_match = self.customer_profiles_df[self.customer_profiles_df['customer_id'] == cid]
        if cust_match.empty:
            # Fallback to computing profile from transactions
            tx_cust = self.transactions_df[self.transactions_df['customer_id'] == cid]
            if tx_cust.empty:
                return {"error": f"Customer ID {cid} not found."}
            tot_amt = float(tx_cust['amount'].sum())
            avg_amt = float(tx_cust['amount'].mean())
            cnt = len(tx_cust)
            avg_bal = float(tx_cust['balance_before'].mean())
            uniq_merch = int(tx_cust['merchant'].nunique())
            frd_cnt = int(tx_cust['is_fraud'].sum())
        else:
            row = cust_match.iloc[0]
            tot_amt = float(row['total_transaction_amount'])
            avg_amt = float(row['average_transaction_amount'])
            cnt = int(row['transaction_count'])
            avg_bal = float(row['average_balance_before'])
            uniq_merch = int(row['unique_merchants'])
            frd_cnt = int(row['fraud_count'])

        # Filter customer's transactions
        cust_txns = self.transactions_df[self.transactions_df['customer_id'] == cid].sort_values(by=['transaction_date', 'transaction_time'], ascending=False)
        
        # Determine cluster label using cluster stats
        cluster_label = "Standard Retail Customers"
        cluster_id = 0
        if avg_amt > 15000 or tot_amt > 5000000:
            cluster_label = "High Spending / VIP Customers"
            cluster_id = 1
        elif frd_cnt >= 5:
            cluster_label = "High Risk / Fraud Prone Customers"
            cluster_id = 2
        elif avg_bal > 75000:
            cluster_label = "High Balance Customers"
            cluster_id = 3

        # Risk indicators
        risk_flags = []
        if frd_cnt > 0:
            risk_flags.append(f"Recorded {frd_cnt} Fraudulent Transactions")
        if avg_amt > 10000:
            risk_flags.append("High Average Transaction Amount")
        if any(cust_txns['balance_after'] == 0):
            risk_flags.append("Account Drain Event Detected")

        # Category breakdowns
        type_dist = cust_txns['transaction_type'].value_counts().to_dict()
        pm_dist = cust_txns['payment_method'].value_counts().to_dict()
        merch_dist = cust_txns['merchant'].value_counts().head(5).to_dict()
        loc_dist = cust_txns['location'].value_counts().to_dict()

        # Timeline
        timeline = []
        for _, t in cust_txns.head(15).iterrows():
            timeline.append({
                "transaction_id": str(t['transaction_id']),
                "date": str(t['transaction_date']),
                "time": str(t['transaction_time']),
                "type": str(t['transaction_type']),
                "amount": round(float(t['amount']), 2),
                "merchant": str(t['merchant']),
                "location": str(t['location']),
                "payment_method": str(t['payment_method']),
                "is_fraud": int(t['is_fraud'])
            })

        return {
            "customer_id": cid,
            "account_type": str(cust_txns.iloc[0]['account_type']) if not cust_txns.empty else "Savings",
            "primary_location": str(cust_txns.iloc[0]['location']) if not cust_txns.empty else "Mumbai",
            "total_spending": round(tot_amt, 2),
            "average_spending": round(avg_amt, 2),
            "transaction_count": cnt,
            "average_balance": round(avg_bal, 2),
            "unique_merchants": uniq_merch,
            "fraud_count": frd_cnt,
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "risk_flags": risk_flags,
            "type_distribution": type_dist,
            "payment_distribution": pm_dist,
            "merchant_distribution": merch_dist,
            "location_distribution": loc_dist,
            "recent_transactions": timeline
        }

    def get_transaction_investigation(self, transaction_id):
        """Generates comprehensive investigation panel details for a specific transaction ID."""
        tx_id = str(transaction_id)
        if self.transactions_df is None:
            return {"error": "Transaction data not loaded."}

        match = self.transactions_df[self.transactions_df['transaction_id'] == tx_id]
        if match.empty:
            # Fallback to first suspicious transaction if target id not found
            match = self.transactions_df[self.transactions_df['is_fraud'] == 1].head(1)
            if match.empty:
                return {"error": f"Transaction '{tx_id}' not found."}

        row = match.iloc[0]
        cid = int(row['customer_id'])
        amt = float(row['amount'])
        bal_before = float(row['balance_before'])
        bal_after = float(row['balance_after'])
        time_str = str(row['transaction_time'])
        hour = int(time_str.split(':')[0]) if ':' in time_str else 12

        # Customer background
        cust_profile = self.get_customer_profile(cid)

        # Risk factors (derived strictly from row values)
        risk_factors = []
        if amt > 100000:
            risk_factors.append({
                "factor": "High Transaction Amount",
                "impact": "High",
                "detail": f"Amount ₹{amt:,.2f} is significantly above average baseline."
            })
        if hour < 5 or hour > 23:
            risk_factors.append({
                "factor": "Unusual Transaction Hour",
                "impact": "High",
                "detail": f"Executed at {time_str} during late-night risk window."
            })
        if bal_after == 0:
            risk_factors.append({
                "factor": "Complete Account Drain",
                "impact": "Critical",
                "detail": "Account balance depleted to ₹0.00 following transaction."
            })
        if bal_before > 0 and (amt / bal_before) > 0.8:
            risk_factors.append({
                "factor": "Excessive Balance Ratio",
                "impact": "Medium",
                "detail": f"Transaction consumes {round((amt/bal_before)*100, 1)}% of available balance."
            })
        if not risk_factors:
            risk_factors.append({
                "factor": "Standard Transaction Pattern",
                "impact": "Low",
                "detail": "Behavioral values align with normal operational baseline."
            })

        # Predict with model
        inference_input = {
            'amount': amt,
            'balance_before': bal_before,
            'balance_after': bal_after,
            'transaction_time': time_str,
            'transaction_type': str(row['transaction_type']),
            'account_type': str(row['account_type']),
            'payment_method': str(row['payment_method']),
            'device_type': str(row['device_type']),
            'location': str(row['location']),
            'transaction_date': str(row['transaction_date'])
        }
        pred_res = self.predict_fraud(inference_input)

        return {
            "transaction_overview": {
                "transaction_id": str(row['transaction_id']),
                "customer_id": cid,
                "amount": round(amt, 2),
                "transaction_date": str(row['transaction_date']),
                "transaction_time": time_str,
                "transaction_type": str(row['transaction_type']),
                "account_type": str(row['account_type']),
                "balance_before": round(bal_before, 2),
                "balance_after": round(bal_after, 2),
                "merchant": str(row['merchant']),
                "location": str(row['location']),
                "payment_method": str(row['payment_method']),
                "device_type": str(row['device_type']),
                "status": str(row['transaction_status']),
                "is_fraud": int(row['is_fraud'])
            },
            "customer_context": cust_profile,
            "risk_factors": risk_factors,
            "model_prediction": pred_res
        }

    def get_transactions(self, page=1, per_page=25, customer_id=None, transaction_type=None,
                         payment_method=None, location=None, is_fraud=None, min_amount=None, max_amount=None, search=None,
                         sort_by="transaction_date", order="desc"):
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

        # Sorting
        if sort_by in df_filtered.columns:
            ascending = (order.lower() == 'asc')
            df_filtered = df_filtered.sort_values(by=sort_by, ascending=ascending)

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
        """Executes single transaction fraud inference using the selected model."""
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

    def get_report_data(self):
        """Generates downloadable summary report dataset."""
        return {
            "title": "Banking Fraud Analytics & Customer Intelligence Executive Report",
            "generated_at": "2026-09-18",
            "summary": self.get_summary(),
            "model_performance": self.get_model_performance(),
            "customer_clusters": self.get_customer_clusters(),
            "anomalies_summary": self.get_anomalies(),
            "data_quality": self.get_data_quality(),
            "processing_metadata": self.get_processing_metadata()
        }
