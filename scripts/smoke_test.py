"""
End-to-end pipeline smoke test.

Generates a small synthetic dataset with the same schema and signal structure as
the real one, then runs every stage -- PySpark preprocessing, K-Means
segmentation, Isolation Forest anomaly detection and the fraud model benchmark --
into a temporary directory, and asserts the artefacts are present and coherent.

This is what CI runs: it proves the pipeline executes without needing the 1.85 GB
raw file in the repository.

    python scripts/smoke_test.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

ROWS = 60_000
CUSTOMERS = 500


def generate_csv(path):
    """Synthesise transactions carrying the signal structure the pipeline expects."""
    rng = np.random.RandomState(42)
    n = ROWS

    locations = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Dubai", "Singapore", "London", "New York"]
    loc = rng.choice(locations, n, p=[0.24, 0.2, 0.18, 0.14, 0.06, 0.06, 0.06, 0.06])
    hour = rng.randint(0, 24, n)
    amount = np.round(np.clip(rng.lognormal(7.6, 1.3, n), 50, 250_000), 2)

    is_intl = np.isin(loc, ["Dubai", "Singapore", "London", "New York"])
    is_night = hour < 5
    is_high = amount > 100_000

    p = 0.004 + 0.02 * is_night + 0.035 * is_intl + 0.06 * is_high
    is_fraud = (rng.random_sample(n) < p).astype(int)

    balance_before = np.round(rng.uniform(1_000, 300_000, n), 2)
    drained = rng.random_sample(n) < 0.05
    balance_after = np.round(np.where(drained, 0.0, np.maximum(0, balance_before - amount)), 2)

    status = np.where(
        is_fraud == 1,
        rng.choice(["Flagged", "Declined", "Completed"], n, p=[0.55, 0.2, 0.25]),
        rng.choice(["Completed", "Pending"], n, p=[0.97, 0.03]),
    )

    dates = pd.to_datetime(rng.randint(0, 365, n), unit="D", origin="2025-01-01")

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:09d}" for i in range(n)],
        "customer_id": rng.randint(100_000, 100_000 + CUSTOMERS, n),
        "transaction_date": dates.strftime("%Y-%m-%d"),
        "transaction_time": [f"{h:02d}:{m:02d}:{s:02d}" for h, m, s in
                             zip(hour, rng.randint(0, 60, n), rng.randint(0, 60, n), strict=True)],
        "transaction_type": rng.choice(
            ["UPI", "Card Payment", "Bank Transfer", "ATM Withdrawal", "Online Purchase"], n),
        "account_type": rng.choice(["Savings", "Current", "Salary"], n),
        "amount": amount,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "merchant": rng.choice(["Amazon", "Flipkart", "Swiggy", "Zomato", "IRCTC", "DMart"], n),
        "location": loc,
        "payment_method": rng.choice(["UPI", "Credit Card", "Debit Card", "Net Banking", "ATM"], n),
        "device_type": rng.choice(["Android", "iOS", "Windows", "Mac", "ATM"], n),
        "transaction_status": status,
        "is_fraud": is_fraud,
    })
    df.to_csv(path, index=False)
    return df


def point_config_at(tmp):
    """Redirect every config output path into the temporary workspace."""
    from src import config

    config.PROCESSED_DIR = tmp / "processed"
    config.MODELS_DIR = tmp / "models"
    config.RAW_DIR = tmp / "raw"
    for d in (config.PROCESSED_DIR, config.MODELS_DIR, config.RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)

    names = {
        "SUMMARY_STATS_JSON": "summary_stats.json",
        "FRAUD_AGGREGATES_JSON": "fraud_aggregates.json",
        "DATA_QUALITY_JSON": "data_quality.json",
        "TIME_ANALYTICS_JSON": "time_analytics.json",
        "GEO_ANALYTICS_JSON": "geo_analytics.json",
        "PAYMENT_DEVICE_JSON": "payment_device_analytics.json",
        "ALERTS_JSON": "analytical_alerts.json",
        "SEGMENT_LIFT_JSON": "segment_lift.json",
        "LEAKAGE_REPORT_JSON": "leakage_report.json",
        "PIPELINE_BENCHMARK_JSON": "pipeline_benchmark.json",
        "CUSTOMER_CLUSTERS_JSON": "customer_clusters.json",
        "ANOMALIES_JSON": "anomalies_summary.json",
        "TRANSACTIONS_SAMPLE_PARQUET": "transactions_sample.parquet",
        "CUSTOMER_PROFILES_PARQUET": "customer_profiles_15m.parquet",
        "CUSTOMER_PROFILES_CSV": "customer_profiles_15m.csv",
        "TRANSACTIONS_PROCESSED_PARQUET": "transactions_processed.parquet",
    }
    for attr, filename in names.items():
        setattr(config, attr, config.PROCESSED_DIR / filename)

    for attr, filename in {
        "MODEL_COMPARISON_JSON": "model_comparison.json",
        "THRESHOLD_ANALYSIS_JSON": "threshold_analysis.json",
        "FEATURE_IMPORTANCE_JSON": "feature_importance.json",
        "SPARK_ML_COMPARISON_JSON": "spark_ml_comparison.json",
    }.items():
        setattr(config, attr, config.MODELS_DIR / filename)
    config.SPARK_ML_MODEL_DIR = config.MODELS_DIR / "spark_ml"

    config.ML_SAMPLE_TARGET = ROWS
    return config


def check(label, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}{(' -- ' + detail) if detail else ''}")
    return bool(condition)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="bfa_smoke_"))
    print(f"Smoke test workspace: {tmp}")
    ok = True

    try:
        config = point_config_at(tmp)
        csv_path = config.RAW_DIR / "transactions.csv"

        print(f"\n[1/6] Generating {ROWS:,} synthetic transactions...")
        source = generate_csv(csv_path)
        print(f"      fraud rate {source['is_fraud'].mean() * 100:.3f}%")

        print("\n[2/6] PySpark preprocessing...")
        import src.preprocessing.preprocess as pp
        pp.config = config
        result = pp.run_pyspark_preprocessing(
            raw_data_path=csv_path, processed_data_dir=config.PROCESSED_DIR
        )

        summary, quality, lift = result["summary"], result["quality"], result["lift"]
        ok &= check("row count matches input", summary["total_transactions"] == ROWS)
        ok &= check("data quality was measured", quality.get("measured") is True)
        ok &= check("consistency check ran", "consistency_check" in quality)
        ok &= check("segment lift computed", len(lift["segments"]) > 0,
                    f"{len(lift['segments'])} segments")
        night = next((s for s in lift["segments"] if s["segment"].startswith("Night")), None)
        ok &= check("overnight segment shows elevated fraud",
                    night is not None and night["lift_vs_baseline"] > 1.2,
                    f"lift {night['lift_vs_baseline']}x" if night else "missing")
        ok &= check("stratified sample written", config.TRANSACTIONS_SAMPLE_PARQUET.exists())
        ok &= check("alerts generated", config.ALERTS_JSON.exists())
        ok &= check("benchmark recorded", config.PIPELINE_BENCHMARK_JSON.exists())

        print("\n[3/6] K-Means customer segmentation...")
        import src.customer_segmentation.clustering as cl
        cl.config = config
        clusters = cl.perform_customer_segmentation()
        ok &= check("clusters produced", len(clusters["cluster_profiles"]) >= 2)
        ok &= check("assignments saved", len(clusters.get("assignments", {})) > 0)
        ok &= check("quality verdict present", "segmentation_quality" in clusters)

        print("\n[4/6] Isolation Forest anomaly detection...")
        import src.anomaly_detection.anomaly as an
        an.config = config
        anomalies = an.perform_anomaly_detection()
        ok &= check("anomalies detected", anomalies["anomalies_in_sample"] > 0)
        ok &= check("validated against labels", anomalies.get("label_validation") is not None)

        print("\n[5/6] Fraud model benchmark + leakage audit...")
        import src.fraud_detection.train_models as tm
        tm.config = config
        comparison = tm.train_fraud_models(models_dir=config.MODELS_DIR, run_cv=False)

        models = comparison["models"]
        ok &= check("models trained", len(models) >= 3, f"{len(models)} models")
        ok &= check("selection metric is pr_auc", comparison["selection_metric"] == "pr_auc")
        ok &= check("no-skill baseline reported", "no_skill_baseline" in comparison)

        prevalence = comparison["no_skill_baseline"]["prevalence"]
        best = models[comparison["best_model"]]
        ok &= check("best model beats a random scorer",
                    best["ranking"]["pr_auc"] > prevalence,
                    f"PR-AUC {best['ranking']['pr_auc']} vs random {prevalence}")
        ok &= check("cost analysis produced", "cost_analysis" in best)
        ok &= check("threshold tuning produced", "at_best_f1_threshold" in best)

        leak = comparison["leakage_summary"]
        ok &= check("leakage detected", "LEAKING" in str(leak.get("verdict")))
        ok &= check("leaking rule has perfect precision",
                    leak["zero_model_rule"]["precision"] == 1.0)
        ok &= check("transaction_status excluded from features",
                    not any("status" in f.lower() for f in comparison["feature_names"]))
        ok &= check("target excluded from features",
                    "is_fraud" not in comparison["feature_names"])

        print("\n[6/6] Spark MLlib on the full dataset...")
        # Logistic regression only: the tree ensembles are far slower, and this
        # stage exists to prove the distributed path runs end to end rather than
        # to benchmark it.
        import src.fraud_detection.spark_ml as sml
        sml.config = config
        spark_res = sml.train_spark_models(
            raw_path=csv_path, include_gbt=False, include_rf=False, save_models=False
        )
        ok &= check("trained on the full dataset",
                    spark_res["trained_on_full_dataset"] is True)
        ok &= check("used every row", spark_res["dataset_rows"] == ROWS,
                    f"{spark_res['dataset_rows']:,} rows")
        ok &= check("train + test accounts for the dataset",
                    spark_res["train_rows"] + spark_res["test_rows"] == ROWS)
        ok &= check("class weighting applied",
                    spark_res["class_weighting"]["weight_positive"] > 1)
        sp_best = spark_res["models"][spark_res["best_model"]]
        ok &= check("Spark model produces a ranking",
                    sp_best["ranking"]["pr_auc"] > 0,
                    f"PR-AUC {sp_best['ranking']['pr_auc']}")
        ok &= check("top-slice lift above 1x",
                    any(r["lift"] > 1 for r in sp_best["precision_at_k"]))
        ok &= check("compared against the sampled benchmark",
                    spark_res["comparison_with_sampled_sklearn"]["available"] is True)
        ok &= check("transaction_status excluded from Spark features",
                    all("status" not in c.lower()
                        for c in sml.NUMERIC_COLS + sml.FLAG_COLS
                        + sml.CUSTOMER_COLS + sml.CATEGORICAL_COLS))

        print("\n" + "=" * 60)
        print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
        print("=" * 60)
        return 0 if ok else 1

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
