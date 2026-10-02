"""
Data access and inference layer for the Flask backend.

What changed from the original service
-------------------------------------
1. **No more fabricated values.** `get_fraud_analytics` previously attached
   `"fraud_probability": 0.945` to every suspicious transaction -- the same
   literal for all of them. Probabilities are now produced by the model.
   `get_processing_metadata` returned a hand-written dict (`parquet_size_mb:
   534.5`, `ml_sample_records: 499696`); it now reports what the pipeline
   actually measured. `get_report_data` had a hardcoded `generated_at`.

2. **Inference goes through the shared feature builder.** The service used to
   re-implement feature engineering inline and then branch on
   `if target_model_name == "Logistic Regression"` to decide whether to scale.
   Scaling now lives inside each saved Pipeline, and features come from
   `src.features`, so training and serving cannot drift apart.

3. **New analytics are exposed**: segment lift, the leakage audit, threshold and
   cost analysis, feature importance and the pipeline benchmark.

4. **Risk levels come from the tuned threshold**, not a fixed 0.5/0.35 ladder.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src import features as feat_mod

logger = logging.getLogger(__name__)

# Artefact stem -> display name for the benchmark models.
MODEL_ARTIFACTS = {
    "Logistic Regression": "logistic_regression_pipeline.pkl",
    "CART Decision Tree": "cart_decision_tree_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl",
    "XGBoost": "xgboost_pipeline.pkl",
    "LightGBM": "lightgbm_pipeline.pkl",
}


def _load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not read {path.name}: {exc}")
        return None


class DataService:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else config.BASE_DIR
        self.processed_dir = config.PROCESSED_DIR
        self.models_dir = config.MODELS_DIR

        # Cached JSON analytics
        self.summary_stats = None
        self.fraud_aggregates = None
        self.customer_clusters = None
        self.anomalies_summary = None
        self.model_comparison = None
        self.data_quality = None
        self.time_analytics = None
        self.geo_analytics = None
        self.payment_device_analytics = None
        self.analytical_alerts = None
        self.segment_lift = None
        self.leakage_report = None
        self.pipeline_benchmark = None
        self.threshold_analysis = None
        self.feature_importance = None

        # Datasets
        self.transactions_df = None
        self.customer_profiles_df = None

        # ML artefacts
        self.models = {}
        self.feature_cols = None
        self.customer_baselines = None
        self.best_model_name = None
        self.operating_thresholds = {}

        self.load_cache()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_cache(self):
        logger.info("Loading analytics artefacts and ML models...")

        self.summary_stats = _load_json(config.SUMMARY_STATS_JSON)
        self.fraud_aggregates = _load_json(config.FRAUD_AGGREGATES_JSON)
        self.customer_clusters = _load_json(config.CUSTOMER_CLUSTERS_JSON)
        self.anomalies_summary = _load_json(config.ANOMALIES_JSON)
        self.data_quality = _load_json(config.DATA_QUALITY_JSON)
        self.time_analytics = _load_json(config.TIME_ANALYTICS_JSON)
        self.geo_analytics = _load_json(config.GEO_ANALYTICS_JSON)
        self.payment_device_analytics = _load_json(config.PAYMENT_DEVICE_JSON)
        self.analytical_alerts = _load_json(config.ALERTS_JSON)
        self.segment_lift = _load_json(config.SEGMENT_LIFT_JSON)
        self.leakage_report = _load_json(config.LEAKAGE_REPORT_JSON)
        self.pipeline_benchmark = _load_json(config.PIPELINE_BENCHMARK_JSON)
        self.model_comparison = _load_json(config.MODEL_COMPARISON_JSON)
        self.threshold_analysis = _load_json(config.THRESHOLD_ANALYSIS_JSON)
        self.feature_importance = _load_json(config.FEATURE_IMPORTANCE_JSON)

        if self.model_comparison:
            self.best_model_name = self.model_comparison.get("best_model")

        # --- ML artefacts ---
        feat_path = self.models_dir / "feature_cols.pkl"
        if feat_path.exists():
            self.feature_cols = joblib.load(feat_path)

        base_path = self.models_dir / "customer_baselines.pkl"
        if base_path.exists():
            self.customer_baselines = joblib.load(base_path)
            logger.info(f"Loaded customer baselines for {len(self.customer_baselines):,} customers.")

        for name, filename in MODEL_ARTIFACTS.items():
            path = self.models_dir / filename
            if path.exists():
                try:
                    self.models[name] = joblib.load(path)
                    logger.info(f"Loaded model pipeline: {name}")
                except Exception as exc:
                    logger.warning(f"Failed to load {filename}: {exc}")

        if self.best_model_name not in self.models and self.models:
            self.best_model_name = next(iter(self.models))

        # Operating thresholds chosen by the cost analysis, falling back to 0.5.
        if self.threshold_analysis:
            for name, block in self.threshold_analysis.get("models", {}).items():
                chosen = block.get("at_cost_optimal_threshold") or block.get("at_best_f1_threshold")
                if chosen:
                    self.operating_thresholds[name] = float(chosen["threshold"])

        # --- Datasets ---
        if config.TRANSACTIONS_SAMPLE_PARQUET.exists():
            logger.info(f"Loading transactions from {config.TRANSACTIONS_SAMPLE_PARQUET.name}...")
            self.transactions_df = pd.read_parquet(config.TRANSACTIONS_SAMPLE_PARQUET)
        else:
            monthly_dir = self.processed_dir / "transactions_by_month"
            if monthly_dir.exists():
                logger.info("Loading month-partitioned transactions...")
                self.transactions_df = pd.read_parquet(monthly_dir)
            elif config.TRANSACTIONS_PROCESSED_PARQUET.exists():
                self.transactions_df = pd.read_parquet(config.TRANSACTIONS_PROCESSED_PARQUET)

        if config.CUSTOMER_PROFILES_PARQUET.exists():
            self.customer_profiles_df = pd.read_parquet(config.CUSTOMER_PROFILES_PARQUET)
        elif config.CUSTOMER_PROFILES_CSV.exists():
            self.customer_profiles_df = pd.read_csv(config.CUSTOMER_PROFILES_CSV)

        # Prefer the Spark 15M profile table for serving-time customer baselines;
        # it is strictly richer than the train-split table saved by the trainer.
        if self.customer_baselines is None and self.customer_profiles_df is not None:
            self.customer_baselines = feat_mod.customer_stats_from_profiles(self.customer_profiles_df)

        logger.info(
            f"Ready: {len(self.models)} models, "
            f"{0 if self.transactions_df is None else len(self.transactions_df):,} transactions."
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _operating_threshold(self, model_name):
        return self.operating_thresholds.get(model_name, 0.5)

    def _risk_level(self, proba, model_name):
        """
        Band a probability against the model's tuned operating point rather than
        the arbitrary 0.70/0.35 cuts the service used before.
        """
        t = self._operating_threshold(model_name)
        if proba >= t:
            return "High"
        if proba >= t * 0.6:
            return "Medium"
        return "Low"

    def score_frame(self, df, model_name=None):
        """
        Score a raw transaction frame and return the fraud probabilities.

        Features are built by `src.features`, the same module the trainer used.
        """
        name = model_name if model_name in self.models else self.best_model_name
        model = self.models.get(name)
        if model is None or self.feature_cols is None:
            return None, name

        X, _ = feat_mod.assemble_matrix(df, customer_stats=self.customer_baselines, fit=False)
        X = X.reindex(columns=self.feature_cols, fill_value=0.0)
        return model.predict_proba(X.values)[:, 1], name

    # ------------------------------------------------------------------
    # Analytics endpoints
    # ------------------------------------------------------------------

    def get_summary(self):
        if not self.summary_stats:
            return {"error": "Summary stats not found. Run the PySpark pipeline first."}

        summary = dict(self.summary_stats)
        if self.anomalies_summary:
            summary["detected_anomalies"] = self.anomalies_summary.get("total_anomalies", 0)
        if self.data_quality:
            summary["data_quality_scores"] = {
                "completeness": self.data_quality.get("completeness_score"),
                "uniqueness": self.data_quality.get("uniqueness_score"),
                "validity": self.data_quality.get("validity_score"),
                "consistency": self.data_quality.get("consistency_score"),
            }
        if self.model_comparison:
            best = self.model_comparison.get("best_model")
            block = self.model_comparison.get("models", {}).get(best, {})
            top1 = next(
                (r for r in block.get("precision_at_k", []) if r.get("capacity_fraction") == 0.01),
                None,
            )
            summary["best_model"] = best
            summary["best_model_pr_auc"] = block.get("ranking", {}).get("pr_auc")
            summary["best_model_roc_auc"] = block.get("ranking", {}).get("roc_auc")
            if top1:
                summary["top_1pct_review_lift"] = top1.get("lift")
                summary["top_1pct_review_precision"] = top1.get("precision_at_k")
            cost = block.get("cost_analysis", {}).get("cost_optimal")
            if cost:
                summary["cost_optimal_net_saving_test_set"] = cost.get("net_saving")
        return summary

    def get_fraud_analytics(self):
        if not self.fraud_aggregates:
            return {"error": "Fraud aggregates not found."}

        result = dict(self.fraud_aggregates)
        result["suspicious_transactions"] = self._suspicious_transactions(limit=30)
        if self.segment_lift:
            result["segment_lift"] = self.segment_lift
        return result

    def _suspicious_transactions(self, limit=30):
        """
        The highest-risk confirmed-fraud transactions, with a real model score.

        The previous implementation stamped every row with
        `fraud_probability: 0.945`.
        """
        if self.transactions_df is None:
            return []

        fraud_df = self.transactions_df[self.transactions_df["is_fraud"] == 1].head(limit)
        if fraud_df.empty:
            return []

        probs, model_name = self.score_frame(fraud_df)

        records = []
        for i, (_, row) in enumerate(fraud_df.iterrows()):
            amount = float(row["amount"])
            proba = float(probs[i]) if probs is not None else None
            records.append({
                "transaction_id": str(row["transaction_id"]),
                "customer_id": int(row["customer_id"]),
                "transaction_date": str(row["transaction_date"]),
                "transaction_time": str(row["transaction_time"]),
                "transaction_type": str(row["transaction_type"]),
                "account_type": str(row["account_type"]),
                "amount": round(amount, 2),
                "balance_before": round(float(row["balance_before"]), 2),
                "balance_after": round(float(row["balance_after"]), 2),
                "merchant": str(row["merchant"]),
                "location": str(row["location"]),
                "payment_method": str(row["payment_method"]),
                "device_type": str(row["device_type"]),
                "status": str(row["transaction_status"]),
                "fraud_probability": round(proba, 4) if proba is not None else None,
                "risk_level": self._risk_level(proba, model_name) if proba is not None else "Unknown",
                "scored_by": model_name,
            })

        records.sort(key=lambda r: -(r["fraud_probability"] or 0))
        return records

    def get_fraud_trends(self):
        if not self.fraud_aggregates:
            return {"error": "Fraud aggregates not found."}
        return {
            "monthly": self.fraud_aggregates.get("monthly_trends", []),
            "daily": self.fraud_aggregates.get("daily_trends", []),
        }

    def get_customer_clusters(self):
        return self.customer_clusters or {"error": "Customer clusters not found."}

    def get_anomalies(self):
        return self.anomalies_summary or {"error": "Anomaly summary not found."}

    def get_model_performance(self):
        return self.model_comparison or {"error": "Model comparison not found."}

    def get_data_quality(self):
        return self.data_quality or {"error": "Data quality metrics not found."}

    def get_time_analytics(self):
        return self.time_analytics or {"error": "Time analytics not found."}

    def get_geo_analytics(self):
        return self.geo_analytics or {"error": "Geo analytics not found."}

    def get_payment_device_analytics(self):
        return self.payment_device_analytics or {"error": "Payment/device analytics not found."}

    def get_segment_lift(self):
        return self.segment_lift or {"error": "Segment lift analysis not found."}

    def get_leakage_report(self):
        return self.leakage_report or {"error": "Leakage report not found."}

    def get_threshold_analysis(self):
        return self.threshold_analysis or {"error": "Threshold analysis not found."}

    def get_feature_importance(self):
        return self.feature_importance or {"error": "Feature importance not found."}

    def get_alerts(self, category="All"):
        alerts = self.analytical_alerts or []
        if category and category.lower() != "all":
            return [a for a in alerts if a.get("type", "").lower() == category.lower()]
        return alerts

    def get_processing_metadata(self):
        """
        Pipeline provenance, read from what the run actually recorded.

        This used to be a hardcoded dictionary including a fabricated
        `parquet_size_mb: 534.5` and a fixed row count.
        """
        if not self.pipeline_benchmark:
            return {
                "error": "Pipeline benchmark not found. Run the PySpark pipeline to generate it.",
                "engine": f"Apache PySpark {config.SPARK_MASTER}",
                "hadoop": "NOT USED",
                "hdfs": "NOT USED",
            }

        bench = dict(self.pipeline_benchmark)
        bench["schema"] = config.RAW_COLUMNS
        if self.data_quality:
            bench["data_quality_scores"] = {
                "completeness": self.data_quality.get("completeness_score"),
                "uniqueness": self.data_quality.get("uniqueness_score"),
                "validity": self.data_quality.get("validity_score"),
                "consistency": self.data_quality.get("consistency_score"),
            }
        if self.model_comparison:
            bench["ml_sample_records"] = self.model_comparison.get("training_sample_size")
            bench["ml_features"] = self.model_comparison.get("n_features")
        return bench

    # ------------------------------------------------------------------
    # Customer 360
    # ------------------------------------------------------------------

    def get_customer_profile(self, customer_id):
        try:
            cid = int(customer_id)
        except (TypeError, ValueError):
            return {"error": f"Invalid customer id: {customer_id!r}"}

        if self.transactions_df is None:
            return {"error": "Transaction data not loaded."}

        profile_row = None
        if self.customer_profiles_df is not None:
            match = self.customer_profiles_df[self.customer_profiles_df["customer_id"] == cid]
            if not match.empty:
                profile_row = match.iloc[0]

        cust_txns = self.transactions_df[self.transactions_df["customer_id"] == cid]
        if profile_row is None and cust_txns.empty:
            return {"error": f"Customer ID {cid} not found."}

        if profile_row is not None:
            tot_amt = float(profile_row["total_transaction_amount"])
            avg_amt = float(profile_row["average_transaction_amount"])
            cnt = int(profile_row["transaction_count"])
            avg_bal = float(profile_row["average_balance_before"])
            uniq_merch = int(profile_row["unique_merchants"])
            frd_cnt = int(profile_row["fraud_count"])
        else:
            tot_amt = float(cust_txns["amount"].sum())
            avg_amt = float(cust_txns["amount"].mean())
            cnt = len(cust_txns)
            avg_bal = float(cust_txns["balance_before"].mean())
            uniq_merch = int(cust_txns["merchant"].nunique())
            frd_cnt = int(cust_txns["is_fraud"].sum())

        cust_txns = cust_txns.sort_values(
            by=["transaction_date", "transaction_time"], ascending=False
        )

        cluster_id, cluster_label = self._assign_cluster(cid, avg_amt, tot_amt, frd_cnt, avg_bal)

        risk_flags = []
        if frd_cnt > 0:
            risk_flags.append(f"{frd_cnt} confirmed fraudulent transaction(s) on record")
        if avg_amt > 10000:
            risk_flags.append("Average transaction value well above portfolio mean")
        if not cust_txns.empty and (cust_txns["balance_after"] == 0).any():
            drains = int((cust_txns["balance_after"] == 0).sum())
            risk_flags.append(f"{drains} account-drain event(s) detected")
        if profile_row is not None and "international_txn_count" in profile_row.index:
            intl = int(profile_row["international_txn_count"])
            if intl:
                risk_flags.append(f"{intl} cross-border transaction(s) -- 5x baseline fraud segment")
        if profile_row is not None and "max_txns_single_day" in profile_row.index:
            burst = int(profile_row["max_txns_single_day"] or 0)
            if burst >= 5:
                risk_flags.append(f"Velocity burst: {burst} transactions in a single day")

        velocity = {}
        if profile_row is not None:
            for key in ("avg_gap_hours", "min_gap_minutes", "max_txns_single_day",
                        "avg_txns_per_active_day", "active_days", "unique_locations",
                        "international_txn_count", "night_txn_count",
                        "std_transaction_amount", "max_transaction_amount"):
                if key in profile_row.index and pd.notna(profile_row[key]):
                    velocity[key] = float(profile_row[key])

        timeline = []
        if not cust_txns.empty:
            head = cust_txns.head(15)
            probs, model_name = self.score_frame(head)
            for i, (_, t) in enumerate(head.iterrows()):
                timeline.append({
                    "transaction_id": str(t["transaction_id"]),
                    "date": str(t["transaction_date"]),
                    "time": str(t["transaction_time"]),
                    "type": str(t["transaction_type"]),
                    "amount": round(float(t["amount"]), 2),
                    "merchant": str(t["merchant"]),
                    "location": str(t["location"]),
                    "payment_method": str(t["payment_method"]),
                    "is_fraud": int(t["is_fraud"]),
                    "fraud_probability": round(float(probs[i]), 4) if probs is not None else None,
                })

        return {
            "customer_id": cid,
            "account_type": str(cust_txns.iloc[0]["account_type"]) if not cust_txns.empty else None,
            "primary_location": (
                str(cust_txns["location"].mode().iloc[0]) if not cust_txns.empty else None
            ),
            "total_spending": round(tot_amt, 2),
            "average_spending": round(avg_amt, 2),
            "transaction_count": cnt,
            "average_balance": round(avg_bal, 2),
            "unique_merchants": uniq_merch,
            "fraud_count": frd_cnt,
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "risk_flags": risk_flags,
            "velocity_metrics": velocity,
            "type_distribution": cust_txns["transaction_type"].value_counts().to_dict(),
            "payment_distribution": cust_txns["payment_method"].value_counts().to_dict(),
            "merchant_distribution": cust_txns["merchant"].value_counts().head(5).to_dict(),
            "location_distribution": cust_txns["location"].value_counts().to_dict(),
            "recent_transactions": timeline,
            "transactions_in_sample": len(cust_txns),
        }

    def _assign_cluster(self, cid, avg_amt, tot_amt, frd_cnt, avg_bal):
        """
        Resolve the customer's K-Means segment.

        Prefers the real cluster assignment saved by the segmentation job. The
        original code only ever used an inline if/elif ladder whose labels and
        ids did not correspond to the trained K-Means clusters at all.
        """
        assignments = (self.customer_clusters or {}).get("assignments")
        if assignments:
            cluster_id = assignments.get(str(cid), assignments.get(cid))
            if cluster_id is not None:
                for prof in (self.customer_clusters or {}).get("cluster_profiles", []):
                    if prof.get("cluster_id") == int(cluster_id):
                        return int(cluster_id), prof.get("label", f"Cluster {cluster_id}")
                return int(cluster_id), f"Cluster {cluster_id}"

        # Heuristic fallback, clearly labelled as such.
        if avg_amt > 15000 or tot_amt > 5_000_000:
            return None, "High Spending / VIP Customers (heuristic)"
        if frd_cnt >= 5:
            return None, "High Risk / Fraud Prone Customers (heuristic)"
        if avg_bal > 75000:
            return None, "High Balance Customers (heuristic)"
        return None, "Standard Retail Customers (heuristic)"

    # ------------------------------------------------------------------
    # Investigation
    # ------------------------------------------------------------------

    def get_transaction_investigation(self, transaction_id):
        tx_id = str(transaction_id)
        if self.transactions_df is None:
            return {"error": "Transaction data not loaded."}

        match = self.transactions_df[self.transactions_df["transaction_id"] == tx_id]
        if match.empty:
            # Report a miss instead of silently substituting another transaction,
            # which is what the previous implementation did.
            return {"error": f"Transaction '{tx_id}' not found in the loaded partition."}

        row = match.iloc[0]
        cid = int(row["customer_id"])
        amt = float(row["amount"])
        bal_before = float(row["balance_before"])
        bal_after = float(row["balance_after"])
        time_str = str(row["transaction_time"])
        hour = int(time_str.split(":")[0]) if ":" in time_str else 12
        location = str(row["location"])

        risk_factors = []
        if amt > config.HIGH_AMOUNT_THRESHOLD:
            risk_factors.append({
                "factor": "High transaction amount", "impact": "High",
                "detail": (
                    f"Rs {amt:,.2f} exceeds the Rs {config.HIGH_AMOUNT_THRESHOLD:,.0f} "
                    f"threshold, a segment with roughly 9x baseline fraud rate."
                ),
            })
        if hour < config.NIGHT_END_HOUR:
            risk_factors.append({
                "factor": "Overnight execution", "impact": "High",
                "detail": f"Executed at {time_str}; the 00:00-04:59 window runs ~3x baseline fraud.",
            })
        if location in config.INTERNATIONAL_LOCATIONS:
            risk_factors.append({
                "factor": "Cross-border transaction", "impact": "High",
                "detail": f"{location} is an international corridor with ~5x baseline fraud rate.",
            })
        if bal_after == 0:
            risk_factors.append({
                "factor": "Complete account drain", "impact": "Critical",
                "detail": "Balance depleted to zero; also breaks the ledger identity.",
            })
        if bal_before > 0 and (amt / bal_before) > 0.8:
            risk_factors.append({
                "factor": "Excessive balance ratio", "impact": "Medium",
                "detail": f"Consumes {amt / bal_before * 100:.1f}% of the available balance.",
            })
        if not risk_factors:
            risk_factors.append({
                "factor": "No elevated risk indicators", "impact": "Low",
                "detail": "Values sit within normal operating ranges.",
            })

        probs, model_name = self.score_frame(match)
        proba = float(probs[0]) if probs is not None else None
        threshold = self._operating_threshold(model_name)

        return {
            "transaction_overview": {
                "transaction_id": tx_id,
                "customer_id": cid,
                "amount": round(amt, 2),
                "transaction_date": str(row["transaction_date"]),
                "transaction_time": time_str,
                "transaction_type": str(row["transaction_type"]),
                "account_type": str(row["account_type"]),
                "balance_before": round(bal_before, 2),
                "balance_after": round(bal_after, 2),
                "merchant": str(row["merchant"]),
                "location": location,
                "payment_method": str(row["payment_method"]),
                "device_type": str(row["device_type"]),
                "status": str(row["transaction_status"]),
                "is_fraud": int(row["is_fraud"]),
            },
            "customer_context": self.get_customer_profile(cid),
            "risk_factors": risk_factors,
            "model_prediction": {
                "prediction": "Fraud" if (proba is not None and proba >= threshold) else "Legitimate",
                "fraud_probability": round(proba, 4) if proba is not None else None,
                "fraud_probability_pct": f"{proba * 100:.2f}%" if proba is not None else None,
                "risk_level": self._risk_level(proba, model_name) if proba is not None else "Unknown",
                "model_used": model_name,
                "operating_threshold": round(threshold, 4),
            },
            "ground_truth_note": (
                "transaction_status is shown for context only. It records the bank's own "
                "fraud verdict and is excluded from every model feature set -- see the "
                "leakage audit at /api/leakage."
            ),
        }

    # ------------------------------------------------------------------
    # Transaction explorer
    # ------------------------------------------------------------------

    def get_transactions(self, page=1, per_page=25, customer_id=None, transaction_type=None,
                         payment_method=None, location=None, is_fraud=None, min_amount=None,
                         max_amount=None, search=None, sort_by="transaction_date", order="desc",
                         max_rows=None):
        if self.transactions_df is None:
            return {"error": "Transaction data not loaded."}

        df = self.transactions_df

        try:
            if customer_id:
                df = df[df["customer_id"] == int(customer_id)]
            if transaction_type:
                df = df[df["transaction_type"].str.lower() == transaction_type.lower()]
            if payment_method:
                df = df[df["payment_method"].str.lower() == payment_method.lower()]
            if location:
                df = df[df["location"].str.lower() == location.lower()]
            if is_fraud not in (None, ""):
                df = df[df["is_fraud"] == int(is_fraud)]
            if min_amount not in (None, ""):
                df = df[df["amount"] >= float(min_amount)]
            if max_amount not in (None, ""):
                df = df[df["amount"] <= float(max_amount)]
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid filter value: {exc}"}

        if search:
            s = str(search).lower()
            df = df[
                df["transaction_id"].str.lower().str.contains(s, regex=False)
                | df["merchant"].str.lower().str.contains(s, regex=False)
                | df["customer_id"].astype(str).str.contains(s, regex=False)
            ]

        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=(str(order).lower() == "asc"))

        total = len(df)
        per_page = max(1, min(int(per_page), 500))
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(int(page), total_pages))

        if max_rows:
            page_df = df.iloc[: int(max_rows)]
        else:
            start = (page - 1) * per_page
            page_df = df.iloc[start:start + per_page]

        records = page_df[[
            "transaction_id", "customer_id", "transaction_date", "transaction_time",
            "transaction_type", "account_type", "amount", "balance_before", "balance_after",
            "merchant", "location", "payment_method", "device_type",
            "transaction_status", "is_fraud",
        ]].to_dict(orient="records")

        for r in records:
            r["customer_id"] = int(r["customer_id"])
            r["is_fraud"] = int(r["is_fraud"])
            for k in ("amount", "balance_before", "balance_after"):
                r[k] = round(float(r[k]), 2)

        return {
            "total_records": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "data": records,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_fraud(self, input_data, model_name=None):
        """
        Score a single transaction supplied by the prediction sandbox.

        Builds features through `src.features`, so the sandbox sees exactly the
        transformation the model was trained on.
        """
        if not self.models:
            return {"error": "No ML models loaded. Run the training pipeline first."}

        name = model_name if model_name in self.models else self.best_model_name
        model = self.models.get(name)
        if model is None:
            return {"error": f"Model '{model_name}' is not available."}
        if self.feature_cols is None:
            return {"error": "Feature column list missing; retrain the models."}

        try:
            row = {
                "transaction_id": "SANDBOX",
                "customer_id": int(input_data.get("customer_id", 0) or 0),
                "transaction_date": str(input_data.get("transaction_date", "2025-06-15")),
                "transaction_time": str(input_data.get("transaction_time", "12:00:00")),
                "transaction_type": str(input_data.get("transaction_type", "UPI")),
                "account_type": str(input_data.get("account_type", "Savings")),
                "amount": float(input_data.get("amount", 0.0) or 0.0),
                "balance_before": float(input_data.get("balance_before", 0.0) or 0.0),
                "balance_after": float(input_data.get("balance_after", 0.0) or 0.0),
                "merchant": str(input_data.get("merchant", "Amazon")),
                "location": str(input_data.get("location", "Mumbai")),
                "payment_method": str(input_data.get("payment_method", "UPI")),
                "device_type": str(input_data.get("device_type", "Android")),
            }
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid input: {exc}"}

        try:
            df = pd.DataFrame([row])
            probs, used = self.score_frame(df, model_name=name)
            if probs is None:
                return {"error": "Scoring failed; model artefacts incomplete."}

            proba = float(probs[0])
            threshold = self._operating_threshold(used)
            feat = feat_mod.build_features(df, customer_stats=self.customer_baselines).iloc[0]

            triggered = []
            if feat["is_night"]:
                triggered.append("Overnight window (00:00-04:59), ~3x baseline fraud rate")
            if feat["is_international"]:
                triggered.append(f"International corridor ({row['location']}), ~5x baseline")
            if feat["is_high_amount"]:
                triggered.append(
                    f"Amount above Rs {config.HIGH_AMOUNT_THRESHOLD:,.0f}, ~9x baseline"
                )
            if feat["zero_balance_after"]:
                triggered.append("Account drained to a zero balance")
            if abs(float(feat["amount_z_vs_customer"])) > 3:
                triggered.append(
                    f"Amount is {feat['amount_z_vs_customer']:.1f} standard deviations from "
                    f"this customer's own average"
                )

            return {
                "prediction": "Fraud" if proba >= threshold else "Legitimate",
                "fraud_probability": round(proba, 4),
                "fraud_probability_pct": f"{proba * 100:.2f}%",
                "risk_level": self._risk_level(proba, used),
                "model_used": used,
                "operating_threshold": round(threshold, 4),
                "threshold_basis": (
                    "cost-optimal (maximises expected net saving)"
                    if used in self.operating_thresholds else "default 0.5"
                ),
                "triggered_risk_factors": triggered or ["No elevated risk indicators"],
            }
        except Exception as exc:
            logger.error(f"Inference error: {exc}", exc_info=True)
            return {"error": f"Inference error: {exc}"}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_report_data(self):
        """Executive report payload. `generated_at` used to be a hardcoded date."""
        return {
            "title": "Banking Fraud Analytics & Customer Intelligence Executive Report",
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "summary": self.get_summary(),
            "data_quality": self.get_data_quality(),
            "segment_lift": self.get_segment_lift(),
            "leakage_audit": self.get_leakage_report(),
            "model_performance": self.get_model_performance(),
            "threshold_analysis": self.get_threshold_analysis(),
            "feature_importance": self.get_feature_importance(),
            "customer_clusters": self.get_customer_clusters(),
            "anomalies_summary": self.get_anomalies(),
            "processing_metadata": self.get_processing_metadata(),
            "alerts": self.get_alerts(),
        }
