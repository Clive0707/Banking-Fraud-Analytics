"""
Target-leakage audit for the banking transactions dataset.

The finding
-----------
`transaction_status` is a perfect fraud predictor:

    status      legit    fraud    fraud rate
    Declined        0      955        100.0%
    Flagged         0    2,622        100.0%
    Completed  479,816    1,235         0.26%
    Pending     15,068        0         0.00%

The rule `status in {Declined, Flagged}` achieves **precision 1.000 and recall
0.743** with no model at all. That is not a feature, it is the answer: the status
field records the bank's own fraud verdict, which only exists *after* the fraud
decision has been made. A model trained on it would score near-perfectly offline
and be worthless in production, where the status of an incoming transaction is
still `Pending`.

Why this module exists rather than a silent column drop
------------------------------------------------------
The original pipeline happened to exclude `transaction_status` from its feature
list, but nothing recorded why, and nothing stopped a later change from adding it
back. Quantifying the leak turns an accident into a documented, reproducible
control -- and the contrast between the leaky and clean models is the clearest
way to show what honest fraud-model evaluation looks like.

The module also runs a generic scan for *any* column that separates the target
suspiciously well, so a future dataset change gets caught automatically.
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)

# A single categorical level that is this pure and this common is almost
# certainly downstream of the label rather than upstream of it.
SUSPICIOUS_PURITY = 0.98
SUSPICIOUS_MIN_SUPPORT = 50


def status_leakage_report(df):
    """Quantify the `transaction_status` leak as a crosstab plus a rule score."""
    status_col, target = "transaction_status", config.TARGET_COL
    if status_col not in df.columns:
        return {"available": False, "reason": f"{status_col} not present in dataset"}

    ct = pd.crosstab(df[status_col], df[target])
    for c in (0, 1):
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[[0, 1]]

    breakdown = []
    for status, row in ct.iterrows():
        legit, fraud = int(row[0]), int(row[1])
        total = legit + fraud
        breakdown.append({
            "status": str(status),
            "legitimate": legit,
            "fraudulent": fraud,
            "total": total,
            "fraud_rate_pct": round(fraud / total * 100, 4) if total else 0.0,
        })
    breakdown.sort(key=lambda r: -r["fraud_rate_pct"])

    flagged = df[status_col].isin(config.LEAKING_STATUS_VALUES)
    total_fraud = int(df[target].sum())
    tp = int(df.loc[flagged, target].sum())
    n_flagged = int(flagged.sum())

    rule = {
        "rule": f"transaction_status in {sorted(config.LEAKING_STATUS_VALUES)}",
        "transactions_matched": n_flagged,
        "frauds_matched": tp,
        "precision": round(tp / n_flagged, 6) if n_flagged else 0.0,
        "recall": round(tp / total_fraud, 6) if total_fraud else 0.0,
        "false_positives": n_flagged - tp,
    }

    return {
        "available": True,
        "column": status_col,
        "breakdown": breakdown,
        "zero_model_rule": rule,
        "verdict": (
            "LEAKING -- excluded from all deployment features"
            if rule["precision"] >= SUSPICIOUS_PURITY
            else "No leak detected"
        ),
        "explanation": (
            "transaction_status encodes the outcome of the bank's existing fraud "
            "controls, so it is only populated after a fraud decision has been "
            "taken. At scoring time an incoming transaction is still Pending, "
            "which means the field cannot be used for prediction even though it "
            "separates the training labels almost perfectly."
        ),
    }


def _is_categorical_like(series):
    """
    True for columns that behave as categories, across pandas versions.

    Testing `dtype == object` is not portable: pandas 3 gives string columns a
    dedicated `str` dtype, so that check silently excluded every categorical
    column and the scan returned no findings at all. A leak detector that
    quietly finds nothing is worse than one that fails loudly, so this tests
    what the column *is not* -- numeric, boolean or temporal -- rather than
    enumerating the dtypes it might be.
    """
    from pandas.api import types as ptypes

    return not (
        ptypes.is_numeric_dtype(series)
        or ptypes.is_bool_dtype(series)
        or ptypes.is_datetime64_any_dtype(series)
        or ptypes.is_timedelta64_dtype(series)
    )


def scan_categorical_purity(df, exclude=None):
    """
    Generic leak detector: flag categorical levels that are almost pure in the
    target and common enough to matter.
    """
    target = config.TARGET_COL
    exclude = set(exclude or []) | {target, "transaction_id", "customer_id"}
    candidates = [c for c in df.columns
                  if c not in exclude and _is_categorical_like(df[c])]

    findings = []
    for col in candidates:
        grp = df.groupby(col, observed=True)[target].agg(["mean", "count"])
        hits = grp[(grp["count"] >= SUSPICIOUS_MIN_SUPPORT) &
                   ((grp["mean"] >= SUSPICIOUS_PURITY) | (grp["mean"] <= 1 - SUSPICIOUS_PURITY))]
        # A level that is purely *negative* is only interesting if the base rate
        # is not already near zero, which it is here, so keep positives only.
        hits = hits[hits["mean"] >= SUSPICIOUS_PURITY]
        for level, row in hits.iterrows():
            findings.append({
                "column": col,
                "level": str(level),
                "support": int(row["count"]),
                "fraud_rate_pct": round(float(row["mean"]) * 100, 4),
            })

    return {
        "columns_scanned": candidates,
        "purity_threshold_pct": SUSPICIOUS_PURITY * 100,
        "min_support": SUSPICIOUS_MIN_SUPPORT,
        "suspicious_levels": sorted(findings, key=lambda r: -r["fraud_rate_pct"]),
    }


def leaky_vs_clean_comparison(df, random_state=None):
    """
    Train the same estimator twice -- once with the leaking status column, once
    without -- so the inflation is measured rather than asserted.

    Uses a shallow decision tree: it is fast, and a leak this clean needs only
    one split to exploit, so the contrast is unmistakable.
    """
    from sklearn.tree import DecisionTreeClassifier

    from src import features as feat_mod
    from src.fraud_detection import metrics as metric_mod

    random_state = config.RANDOM_SEED if random_state is None else random_state
    target = config.TARGET_COL

    y = df[target].astype(int).values
    X_clean, clean_cols = feat_mod.assemble_matrix(df)

    # The leaky variant adds one-hot status columns on top of the clean matrix.
    status_dummies = pd.get_dummies(
        df["transaction_status"], prefix="status", drop_first=False
    ).astype("int64")
    X_leaky = pd.concat([X_clean.reset_index(drop=True),
                         status_dummies.reset_index(drop=True)], axis=1)

    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=config.TEST_SIZE,
        random_state=random_state, stratify=y
    )

    results = {}
    for tag, X in (("clean_no_status", X_clean), ("leaky_with_status", X_leaky)):
        Xv = X.values if hasattr(X, "values") else X
        model = DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=random_state
        )
        model.fit(Xv[idx_tr], y[idx_tr])
        score = model.predict_proba(Xv[idx_te])[:, 1]

        rank = metric_mod.ranking_metrics(y[idx_te], score)
        point = metric_mod.metrics_at_threshold(y[idx_te], score, 0.5)
        results[tag] = {
            "n_features": int(Xv.shape[1]),
            "roc_auc": rank["roc_auc"],
            "pr_auc": rank["pr_auc"],
            "precision": point["precision"],
            "recall": point["recall"],
            "f1_score": point["f1_score"],
            "confusion_matrix": point["confusion_matrix"],
        }

    clean, leaky = results["clean_no_status"], results["leaky_with_status"]
    results["inflation"] = {
        "pr_auc_multiple": round(leaky["pr_auc"] / clean["pr_auc"], 2) if clean["pr_auc"] else None,
        "f1_multiple": round(leaky["f1_score"] / clean["f1_score"], 2) if clean["f1_score"] else None,
        "precision_delta": round(leaky["precision"] - clean["precision"], 6),
        "interpretation": (
            "The leaky model's headline metrics are the ones a naive benchmark "
            "would publish. They are unreachable in production because the "
            "feature driving them is not available at scoring time."
        ),
    }
    return results


def run_leakage_analysis(df, output_path=None):
    """Run the full audit and persist it as JSON."""
    logger.info("Running target-leakage audit...")

    report = {
        "dataset_rows_analysed": int(len(df)),
        "target": config.TARGET_COL,
        "status_leakage": status_leakage_report(df),
        "categorical_purity_scan": scan_categorical_purity(df),
        "excluded_from_features": config.LEAKING_COLS,
    }

    try:
        report["leaky_vs_clean"] = leaky_vs_clean_comparison(df)
    except Exception as exc:  # the audit must never block the pipeline
        logger.warning(f"Leaky-vs-clean comparison failed: {exc}")
        report["leaky_vs_clean"] = {"error": str(exc)}

    output_path = output_path or config.LEAKAGE_REPORT_JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    rule = report["status_leakage"].get("zero_model_rule", {})
    logger.info(
        f"Leakage audit complete: status rule precision={rule.get('precision')} "
        f"recall={rule.get('recall')} -> {report['status_leakage'].get('verdict')}"
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    sample = pd.read_parquet(config.TRANSACTIONS_SAMPLE_PARQUET)
    run_leakage_analysis(sample)
