"""
Distributed fraud classification with Spark MLlib over all 15,000,000 rows.

Why this exists
---------------
The scikit-learn benchmark in `train_models.py` trains on a reproducible
stratified sample of ~500,000 rows, because scikit-learn is single-node and the
full matrix does not fit comfortably in local RAM. That is a defensible
engineering choice, but it leaves a fair criticism of a Big Data project
unanswered: the aggregation layer used all 15M rows while the machine learning
layer used 3% of them.

This module answers it. Spark MLlib trains on the **entire 15M-row dataset** --
feature engineering, train/test split, class weighting, fitting and evaluation
all run distributed across `local[*]` partitions, with nothing collected to the
driver except the final metrics.

NO-HADOOP DIRECTIVE: Spark runs in local standalone mode. No Hadoop, HDFS or
YARN is used.

What it reports
---------------
Results are written in the same metric vocabulary as the sklearn benchmark
(PR-AUC, ROC-AUC, precision/recall at a tuned threshold, precision@k and lift),
so the two can be compared directly. The headline question is whether 15M rows
actually buys accuracy over 500k -- the comparison block answers it with
measurements rather than assumption.
"""

import json
import logging
import shutil
import time

from src import config

config.configure_spark_environment()

from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import (  # noqa: E402
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.ml.feature import (  # noqa: E402
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import DoubleType  # noqa: E402

from src.preprocessing.preprocess import SCHEMA, enrich, get_spark_session  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Categoricals that are one-hot encoded. `location` is deliberately represented
# by the is_international flag instead, matching the sklearn feature set.
CATEGORICAL_COLS = ["transaction_type", "account_type", "payment_method", "device_type"]

NUMERIC_COLS = [
    "amount", "amount_log", "balance_before", "balance_after", "balance_change",
    "hour", "day_of_week", "day_of_month", "amount_to_balance_ratio",
]

FLAG_COLS = [
    "is_night", "is_international", "is_high_amount", "zero_balance_after",
    "intl_x_night", "intl_x_high", "night_x_high",
]

CUSTOMER_COLS = [
    "cust_avg_amount", "cust_std_amount", "cust_txn_count",
    "cust_unique_merchants", "amount_z_vs_customer", "amount_over_cust_mean",
]

# Review capacities for precision@k, matching the sklearn benchmark.
CAPACITY_FRACTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def engineer_features(df):
    """
    Extend the shared `enrich` transformations with the interaction terms and
    the log amount, so the Spark feature set mirrors `src/features.py`.
    """
    return (
        enrich(df)
        .withColumn("amount_log", F.log1p(F.greatest(F.col("amount"), F.lit(0.0))))
        .withColumn("intl_x_night", F.col("is_international") * F.col("is_night"))
        .withColumn("intl_x_high", F.col("is_international") * F.col("is_high_amount"))
        .withColumn("night_x_high", F.col("is_night") * F.col("is_high_amount"))
    )


def attach_customer_baselines(train_df, test_df):
    """
    Join per-customer behavioural baselines, computed on the TRAINING split only.

    Computing them across the whole dataset would let each test row contribute to
    its own baseline. Spark makes the correct version cheap: one groupBy on the
    training partition, then a broadcast-friendly left join onto both splits.
    """
    baselines = train_df.groupBy("customer_id").agg(
        F.avg("amount").alias("cust_avg_amount"),
        F.coalesce(F.stddev("amount"), F.lit(0.0)).alias("cust_std_amount"),
        F.count("*").cast(DoubleType()).alias("cust_txn_count"),
        F.countDistinct("merchant").cast(DoubleType()).alias("cust_unique_merchants"),
    )
    baselines.cache()

    def join_and_derive(d):
        out = d.join(baselines, "customer_id", "left")
        # Customers unseen in training fall back to neutral values.
        out = (
            out
            .withColumn("cust_avg_amount", F.coalesce(F.col("cust_avg_amount"), F.lit(0.0)))
            .withColumn("cust_std_amount", F.coalesce(F.col("cust_std_amount"), F.lit(0.0)))
            .withColumn("cust_txn_count", F.coalesce(F.col("cust_txn_count"), F.lit(0.0)))
            .withColumn("cust_unique_merchants",
                        F.coalesce(F.col("cust_unique_merchants"), F.lit(0.0)))
            .withColumn(
                "amount_z_vs_customer",
                F.when(F.col("cust_std_amount") > 0,
                       (F.col("amount") - F.col("cust_avg_amount")) / F.col("cust_std_amount"))
                .otherwise(F.lit(0.0)),
            )
            .withColumn(
                "amount_over_cust_mean",
                F.when(F.col("cust_avg_amount") > 0,
                       F.col("amount") / F.col("cust_avg_amount"))
                .otherwise(F.lit(1.0)),
            )
        )
        return out

    return join_and_derive(train_df), join_and_derive(test_df), baselines


def build_feature_stages():
    """StringIndexer -> OneHotEncoder -> VectorAssembler for the shared matrix."""
    stages = []
    encoded = []
    for col in CATEGORICAL_COLS:
        idx = f"{col}_idx"
        vec = f"{col}_vec"
        stages.append(StringIndexer(inputCol=col, outputCol=idx, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx], outputCols=[vec], handleInvalid="keep"))
        encoded.append(vec)

    assembler = VectorAssembler(
        inputCols=NUMERIC_COLS + FLAG_COLS + CUSTOMER_COLS + encoded,
        outputCol="features_raw",
        handleInvalid="keep",
    )
    stages.append(assembler)
    return stages, encoded


def add_class_weights(df, label_col="is_fraud"):
    """
    Balanced class weights as an explicit column.

    Spark's LogisticRegression and GBTClassifier both accept `weightCol`, which
    is how class imbalance is handled distributed -- there is no
    `class_weight="balanced"` convenience parameter as in scikit-learn.
    """
    counts = df.groupBy(label_col).count().collect()
    by_label = {int(r[label_col]): r["count"] for r in counts}
    neg, pos = by_label.get(0, 0), by_label.get(1, 0)
    total = neg + pos
    # Weight each class inversely to its frequency, mirroring sklearn's formula.
    w_pos = total / (2.0 * pos) if pos else 1.0
    w_neg = total / (2.0 * neg) if neg else 1.0
    logger.info(f"Class weights -- fraud: {w_pos:.2f}, legitimate: {w_neg:.4f} "
                f"(pos={pos:,} neg={neg:,})")
    return (
        df.withColumn(
            "class_weight",
            F.when(F.col(label_col) == 1, F.lit(w_pos)).otherwise(F.lit(w_neg)),
        ),
        {"positives": pos, "negatives": neg, "weight_positive": round(w_pos, 4),
         "weight_negative": round(w_neg, 6)},
    )


def precision_at_k_spark(scored, fractions=None, score_col="fraud_probability",
                         label_col="is_fraud"):
    """
    precision@k and lift without collecting or fully sorting the test set.

    `approxQuantile` finds the score cutoff for each review capacity in one pass,
    then a single aggregation counts hits above each cutoff. This keeps the
    computation distributed even on a multi-million-row test set, where an
    exact global sort would be the expensive part.
    """
    fractions = fractions or CAPACITY_FRACTIONS
    total = scored.count()
    positives = scored.filter(F.col(label_col) == 1).count()
    prevalence = positives / total if total else 0.0

    # Top k% by score == scores above the (1-k) quantile.
    quantiles = [1.0 - f for f in fractions]
    cutoffs = scored.approxQuantile(score_col, quantiles, 0.0005)

    exprs = []
    for i, cutoff in enumerate(cutoffs):
        flagged = F.col(score_col) >= F.lit(cutoff)
        exprs.append(F.sum(F.when(flagged, 1).otherwise(0)).alias(f"n_{i}"))
        exprs.append(
            F.sum(F.when(flagged & (F.col(label_col) == 1), 1).otherwise(0)).alias(f"tp_{i}")
        )
    row = scored.select(*exprs).collect()[0].asDict()

    rows = []
    for i, frac in enumerate(fractions):
        n, tp = int(row[f"n_{i}"]), int(row[f"tp_{i}"])
        prec = tp / n if n else 0.0
        rows.append({
            "capacity_fraction": frac,
            "capacity_pct": round(frac * 100, 3),
            "reviewed": n,
            "frauds_caught": tp,
            "precision_at_k": round(prec, 6),
            "recall_at_k": round(tp / positives, 6) if positives else 0.0,
            "lift": round(prec / prevalence, 4) if prevalence else 0.0,
        })
    return rows, {"test_rows": total, "test_positives": positives,
                  "prevalence": round(prevalence, 6)}


def metrics_at_threshold_spark(scored, threshold, score_col="fraud_probability",
                               label_col="is_fraud"):
    """Confusion matrix and point metrics at one threshold, computed distributed."""
    pred = F.col(score_col) >= F.lit(threshold)
    row = scored.select(
        F.sum(F.when(~pred & (F.col(label_col) == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when(pred & (F.col(label_col) == 0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when(~pred & (F.col(label_col) == 1), 1).otherwise(0)).alias("fn"),
        F.sum(F.when(pred & (F.col(label_col) == 1), 1).otherwise(0)).alias("tp"),
    ).collect()[0].asDict()

    tn, fp, fn, tp = (int(row[k]) for k in ("tn", "fp", "fn", "tp"))
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round((tp + tn) / total, 6) if total else 0.0,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "alerts_raised": tp + fp,
    }


def best_f1_threshold_spark(scored, n_points=40, score_col="fraud_probability",
                            label_col="is_fraud"):
    """
    Sweep candidate thresholds in one aggregation and return the best F1.

    Candidates come from score quantiles rather than a uniform grid, so they
    concentrate where the scores actually are.
    """
    qs = [i / n_points for i in range(1, n_points)]
    candidates = sorted(set(scored.approxQuantile(score_col, qs, 0.001)))
    candidates = [c for c in candidates if 0.0 < c < 1.0]
    if not candidates:
        return 0.5, None

    exprs = []
    for i, t in enumerate(candidates):
        pred = F.col(score_col) >= F.lit(t)
        exprs.append(F.sum(F.when(pred & (F.col(label_col) == 1), 1).otherwise(0)).alias(f"tp_{i}"))
        exprs.append(F.sum(F.when(pred, 1).otherwise(0)).alias(f"pp_{i}"))
    exprs.append(F.sum(F.col(label_col)).alias("total_pos"))
    row = scored.select(*exprs).collect()[0].asDict()

    total_pos = int(row["total_pos"])
    best_t, best_f1 = 0.5, -1.0
    for i, t in enumerate(candidates):
        tp, pp = int(row[f"tp_{i}"]), int(row[f"pp_{i}"])
        if pp == 0 or total_pos == 0:
            continue
        precision, recall = tp / pp, tp / total_pos
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, round(best_f1, 6)


def build_model_zoo(include_gbt=True, include_rf=True):
    """
    Spark MLlib estimators. All accept the explicit class-weight column.

    Tree depths and iteration counts are kept modest: these fit on 12M training
    rows on a single machine, where each extra tree costs a full pass.
    """
    zoo = {
        "Spark Logistic Regression": LogisticRegression(
            featuresCol="features", labelCol=config.TARGET_COL,
            weightCol="class_weight", maxIter=50, regParam=0.0, elasticNetParam=0.0,
        ),
    }
    if include_rf:
        zoo["Spark Random Forest"] = RandomForestClassifier(
            featuresCol="features", labelCol=config.TARGET_COL,
            weightCol="class_weight", numTrees=40, maxDepth=10,
            subsamplingRate=0.5, seed=config.RANDOM_SEED,
        )
    if include_gbt:
        zoo["Spark GBT"] = GBTClassifier(
            featuresCol="features", labelCol=config.TARGET_COL,
            weightCol="class_weight", maxIter=20, maxDepth=5,
            subsamplingRate=0.5, stepSize=0.15, seed=config.RANDOM_SEED,
        )
    return zoo


def train_spark_models(raw_path=None, include_gbt=True, include_rf=True, save_models=True):
    """Train and evaluate Spark MLlib classifiers on the full 15M-row dataset."""
    raw_path = raw_path or config.RAW_TRANSACTIONS_CSV
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Spark MLlib training reads the "
            f"full CSV rather than the sampled Parquet."
        )

    t_start = time.perf_counter()
    spark = get_spark_session()
    logger.info(f"Spark {spark.version} started in {config.SPARK_MASTER} mode (no Hadoop).")

    # --- Load the complete dataset ---------------------------------------
    logger.info(f"Reading the full dataset from {raw_path.name}...")
    df = spark.read.csv(str(raw_path), header=True, schema=SCHEMA).dropDuplicates(["transaction_id"])
    df = engineer_features(df)

    # transaction_status is dropped here as well as in the sklearn path: it
    # records the bank's own fraud verdict and leaks the target.
    df = df.drop(*config.LEAKING_COLS)

    total_rows = df.count()
    logger.info(f"Training on the complete dataset: {total_rows:,} rows.")
    t_load = time.perf_counter() - t_start

    # --- Split, then fit customer baselines on train only -----------------
    train_df, test_df = df.randomSplit([1 - config.TEST_SIZE, config.TEST_SIZE],
                                       seed=config.RANDOM_SEED)
    logger.info("Computing per-customer baselines on the training split...")
    train_df, test_df, baselines = attach_customer_baselines(train_df, test_df)

    train_df, weight_info = add_class_weights(train_df)
    test_df = test_df.withColumn("class_weight", F.lit(1.0))

    train_df.cache()
    test_df.cache()
    n_train, n_test = train_df.count(), test_df.count()
    logger.info(f"Train {n_train:,} rows / test {n_test:,} rows.")

    # --- Shared feature pipeline -----------------------------------------
    feature_stages, _ = build_feature_stages()
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=False, withStd=True)
    feature_pipeline = Pipeline(stages=feature_stages + [scaler]).fit(train_df)

    train_feat = feature_pipeline.transform(train_df).select(
        config.TARGET_COL, "class_weight", "features"
    ).cache()
    test_feat = feature_pipeline.transform(test_df).select(
        config.TARGET_COL, "features"
    ).cache()
    train_feat.count(), test_feat.count()

    n_features = int(train_feat.select("features").head()["features"].size)
    logger.info(f"Assembled feature vector of {n_features} dimensions.")

    pr_eval = BinaryClassificationEvaluator(
        labelCol=config.TARGET_COL, rawPredictionCol="probability",
        metricName="areaUnderPR",
    )
    roc_eval = BinaryClassificationEvaluator(
        labelCol=config.TARGET_COL, rawPredictionCol="probability",
        metricName="areaUnderROC",
    )

    # Native Spark function rather than a Python UDF: extracting the positive
    # class probability for millions of test rows through a UDF would serialise
    # every row to the Python worker and back.
    def positive_class_prob(col):
        return vector_to_array(col)[1].cast(DoubleType())

    # --- Train -------------------------------------------------------------
    results = {}
    for name, estimator in build_model_zoo(include_gbt, include_rf).items():
        logger.info(f"Training {name} on {n_train:,} rows...")
        t0 = time.perf_counter()
        model = estimator.fit(train_feat)
        train_secs = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred = model.transform(test_feat)
        pr_auc = float(pr_eval.evaluate(pred))
        roc_auc = float(roc_eval.evaluate(pred))

        scored = pred.select(
            config.TARGET_COL,
            positive_class_prob(F.col("probability")).alias("fraud_probability"),
        ).cache()
        scored.count()
        eval_secs = time.perf_counter() - t1

        pak, test_info = precision_at_k_spark(scored)
        at_default = metrics_at_threshold_spark(scored, 0.5)
        t_best, f1_best = best_f1_threshold_spark(scored)
        at_best = metrics_at_threshold_spark(scored, t_best)

        prevalence = test_info["prevalence"]
        results[name] = {
            "trained_on_rows": n_train,
            "tested_on_rows": n_test,
            "n_features": n_features,
            "ranking": {
                "pr_auc": round(pr_auc, 6),
                "roc_auc": round(roc_auc, 6),
                "pr_auc_lift_over_random": round(pr_auc / prevalence, 4) if prevalence else 0.0,
            },
            "at_default_threshold": at_default,
            "at_best_f1_threshold": at_best,
            "precision_at_k": pak,
            "timing": {
                "train_seconds": round(train_secs, 2),
                "evaluate_seconds": round(eval_secs, 2),
            },
        }

        if save_models:
            out = config.SPARK_ML_MODEL_DIR / name.replace(" ", "_").lower()
            try:
                if out.exists():
                    shutil.rmtree(out)
                out.parent.mkdir(parents=True, exist_ok=True)
                model.write().overwrite().save(str(out))
                results[name]["artifact"] = str(out.relative_to(config.BASE_DIR))
            except Exception as exc:
                # Saving Spark models needs writable local storage; never fail
                # the benchmark over it.
                logger.warning(f"Could not persist {name}: {exc}")

        top1 = next(r for r in pak if r["capacity_fraction"] == 0.01)
        logger.info(
            f"  {name}: PR-AUC={pr_auc:.4f} (random={prevalence:.4f}) ROC-AUC={roc_auc:.4f} "
            f"| top-1% precision={top1['precision_at_k']:.4f} lift={top1['lift']:.2f}x "
            f"| {train_secs:.1f}s"
        )
        scored.unpersist()

    best_name = max(results, key=lambda n: results[n]["ranking"]["pr_auc"])
    total_secs = time.perf_counter() - t_start

    payload = {
        "generated_at": __import__("pandas").Timestamp.now().isoformat(timespec="seconds"),
        "engine": f"Apache Spark MLlib {spark.version} ({config.SPARK_MASTER})",
        "hadoop": "NOT USED",
        "trained_on_full_dataset": True,
        "dataset_rows": total_rows,
        "train_rows": n_train,
        "test_rows": n_test,
        "n_features": n_features,
        "class_weighting": weight_info,
        "selection_metric": "pr_auc",
        "best_model": best_name,
        "models": results,
        "timing": {
            "load_and_engineer_seconds": round(t_load, 2),
            "total_seconds": round(total_secs, 2),
        },
        "notes": [
            "Every stage -- feature engineering, split, weighting, fitting and "
            "evaluation -- runs distributed across local[*] partitions. Nothing "
            "but the final metrics is collected to the driver.",
            "Per-customer baselines are computed on the training split only, so "
            "test rows do not contribute to their own baselines.",
            "transaction_status is dropped before training; it records the bank's "
            "own fraud verdict and leaks the target.",
            "precision@k uses approxQuantile to find score cutoffs, avoiding a "
            "full global sort of the multi-million-row test set.",
        ],
    }

    payload["comparison_with_sampled_sklearn"] = _compare_with_sklearn(payload)

    config.SPARK_ML_COMPARISON_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SPARK_ML_COMPARISON_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"Spark MLlib benchmark complete in {total_secs:.1f}s. "
        f"Best: {best_name} (PR-AUC {results[best_name]['ranking']['pr_auc']:.4f})"
    )

    train_feat.unpersist()
    test_feat.unpersist()
    baselines.unpersist()
    spark.stop()
    return payload


def _compare_with_sklearn(spark_payload):
    """
    Put the full-dataset Spark result next to the 500k-sample sklearn result.

    The interesting question for a Big Data project is not which library wins but
    whether 30x more training data actually improves the model.
    """
    if not config.MODEL_COMPARISON_JSON.exists():
        return {"available": False, "reason": "sklearn benchmark has not been run"}

    with open(config.MODEL_COMPARISON_JSON) as f:
        sk = json.load(f)

    sk_best_name = sk.get("best_model")
    sk_best = sk.get("models", {}).get(sk_best_name, {})
    sp_best_name = spark_payload["best_model"]
    sp_best = spark_payload["models"][sp_best_name]

    sk_pr = sk_best.get("ranking", {}).get("pr_auc")
    sp_pr = sp_best["ranking"]["pr_auc"]

    def top1(block):
        return next((r for r in block.get("precision_at_k", [])
                     if r.get("capacity_fraction") == 0.01), {})

    sk_rows = sk.get("train_rows")
    sp_rows = spark_payload["train_rows"]

    delta_pct = round((sp_pr - sk_pr) / sk_pr * 100, 2) if sk_pr else None
    verdict = (
        "inconclusive" if delta_pct is None
        else "more data helps" if delta_pct > 5
        else "more data hurts" if delta_pct < -5
        else "no material gain from the extra data"
    )

    return {
        "available": True,
        "sampled_sklearn": {
            "model": sk_best_name,
            "train_rows": sk_rows,
            "pr_auc": sk_pr,
            "roc_auc": sk_best.get("ranking", {}).get("roc_auc"),
            "top_1pct_lift": top1(sk_best).get("lift"),
        },
        "full_dataset_spark": {
            "model": sp_best_name,
            "train_rows": sp_rows,
            "pr_auc": sp_pr,
            "roc_auc": sp_best["ranking"]["roc_auc"],
            "top_1pct_lift": top1(sp_best).get("lift"),
        },
        "training_data_multiple": round(sp_rows / sk_rows, 1) if sk_rows else None,
        "pr_auc_change_pct": delta_pct,
        "verdict": verdict,
        "interpretation": (
            f"Spark MLlib trained on {sp_rows:,} rows against scikit-learn's "
            f"{sk_rows:,}-row stratified sample "
            f"({round(sp_rows / sk_rows, 1) if sk_rows else '?'}x more data). "
            f"PR-AUC changed by {delta_pct}%, which indicates {verdict}. A flat "
            f"result is itself informative: it shows the 500k stratified sample "
            f"already sits on the plateau of the learning curve for this feature "
            f"set, which is what justifies sampling for the single-node models."
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spark MLlib fraud training on all 15M rows")
    parser.add_argument("--no-gbt", action="store_true", help="Skip the GBT model (slowest)")
    parser.add_argument("--no-rf", action="store_true", help="Skip the Random Forest model")
    args = parser.parse_args()

    train_spark_models(include_gbt=not args.no_gbt, include_rf=not args.no_rf)
