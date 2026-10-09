"""
Canonical feature engineering for the fraud models.

Before this module existed the same derivations were written out four separate
times -- in the Spark pipeline, the model trainer, the anomaly detector and the
Flask inference path -- and they had already drifted apart. A single transaction
could therefore be featurised one way at training time and a different way at
prediction time, which is the classic training/serving skew bug.

Everything now routes through `build_features`, so a model trained on these
columns is served the identical columns.
"""

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def _parse_hour(df):
    """Hour of day (0-23) from a HH:MM:SS string column, defaulting to midday."""
    if "transaction_time" not in df.columns:
        return pd.Series(12, index=df.index, dtype="int64")
    parsed = pd.to_datetime(df["transaction_time"], format="%H:%M:%S", errors="coerce")
    return parsed.dt.hour.fillna(12).astype("int64")


def _parse_calendar(df):
    """Day-of-week (Mon=0) and day-of-month from the date column."""
    if "transaction_date" not in df.columns:
        zeros = pd.Series(0, index=df.index, dtype="int64")
        return zeros, pd.Series(1, index=df.index, dtype="int64")
    parsed = pd.to_datetime(df["transaction_date"], errors="coerce")
    dow = parsed.dt.dayofweek.fillna(0).astype("int64")
    dom = parsed.dt.day.fillna(1).astype("int64")
    return dow, dom


def build_features(df, customer_stats=None):
    """
    Add every engineered feature column to a copy of `df` and return it.

    `customer_stats` is the per-customer behavioural table aggregated by Spark
    across all 15M rows, indexed by customer_id with columns
    cust_avg_amount / cust_std_amount / cust_txn_count / cust_unique_merchants.
    It is optional: without it the customer-relative features fall back to
    neutral values so the column set stays identical either way.

    Note that `transaction_status` is never read here. It encodes the bank's own
    fraud verdict and is handled separately by the leakage study.
    """
    out = df.copy()

    # --- Temporal ---------------------------------------------------------
    out["hour"] = _parse_hour(out)
    out["day_of_week"], out["day_of_month"] = _parse_calendar(out)

    # --- Financial --------------------------------------------------------
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["balance_before"] = pd.to_numeric(out["balance_before"], errors="coerce").fillna(0.0)
    out["balance_after"] = pd.to_numeric(out["balance_after"], errors="coerce").fillna(0.0)

    out["balance_change"] = out["balance_before"] - out["balance_after"]
    out["amount_log"] = np.log1p(out["amount"].clip(lower=0))
    out["amount_to_balance_ratio"] = np.where(
        out["balance_before"] > 0,
        out["amount"] / (out["balance_before"] + 1.0),
        out["amount"],
    )

    # --- Risk flags -------------------------------------------------------
    # Measured lifts versus the 0.96% baseline: night 3.0x, international 5.0x,
    # high amount 9.1x. See reports/segment_lift.json.
    out["is_night"] = (
        (out["hour"] >= config.NIGHT_START_HOUR) & (out["hour"] < config.NIGHT_END_HOUR)
    ).astype("int64")

    if "location" in out.columns:
        out["is_international"] = (
            out["location"].astype(str).isin(config.INTERNATIONAL_LOCATIONS).astype("int64")
        )
    else:
        out["is_international"] = 0

    out["is_high_amount"] = (out["amount"] > config.HIGH_AMOUNT_THRESHOLD).astype("int64")
    out["zero_balance_after"] = (out["balance_after"].abs() <= config.ZERO_BALANCE_EPS).astype("int64")

    # Interactions: night-and-international reaches 6.8% fraud (7.0x lift),
    # which neither flag achieves alone, so the product carries real signal.
    out["intl_x_night"] = out["is_international"] * out["is_night"]
    out["intl_x_high"] = out["is_international"] * out["is_high_amount"]
    out["night_x_high"] = out["is_night"] * out["is_high_amount"]

    # --- Customer-relative behaviour -------------------------------------
    out = _attach_customer_features(out, customer_stats)

    return out


def _attach_customer_features(out, customer_stats):
    """
    Join the Spark-computed per-customer baseline and derive deviation features.

    "Is this amount unusual *for this customer*" is a far stronger fraud signal
    than the raw amount, and it is the feature the original pipeline computed
    (customer_profiles_15m.parquet) but never actually fed to the models.
    """
    neutral = {
        "cust_avg_amount": out["amount"].median() if len(out) else 0.0,
        "cust_std_amount": 0.0,
        "cust_txn_count": 0.0,
        "cust_unique_merchants": 0.0,
    }

    if customer_stats is None or "customer_id" not in out.columns:
        for col, val in neutral.items():
            out[col] = val
    else:
        stats = customer_stats.copy()
        if stats.index.name != "customer_id":
            if "customer_id" in stats.columns:
                stats = stats.set_index("customer_id")
        wanted = ["cust_avg_amount", "cust_std_amount", "cust_txn_count", "cust_unique_merchants"]
        for col in wanted:
            if col not in stats.columns:
                stats[col] = neutral[col]
        out = out.join(stats[wanted], on="customer_id")
        for col, val in neutral.items():
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(val)

    std = out["cust_std_amount"].replace(0, np.nan)
    out["amount_z_vs_customer"] = ((out["amount"] - out["cust_avg_amount"]) / std).fillna(0.0)
    mean = out["cust_avg_amount"].replace(0, np.nan)
    out["amount_over_cust_mean"] = (out["amount"] / mean).fillna(1.0)

    # Guard against infinities reaching the estimators.
    for col in ("amount_z_vs_customer", "amount_over_cust_mean", "amount_to_balance_ratio"):
        out[col] = out[col].replace([np.inf, -np.inf], 0.0)

    # Bound the ratio and deviation features to the range actually observed in
    # the data. No real transaction is affected -- the limits sit outside the
    # observed extremes -- but it stops the model extrapolating far beyond its
    # evidence. Without this a sandbox transaction producing a 444-sigma
    # deviation (against a maximum of 57 in 499,684 real rows) drove a linear
    # model to 99.9% confidence from a region it had never seen.
    for col, (lo, hi) in config.FEATURE_CLIPS.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lo, upper=hi)

    return out


def encode_categoricals(df, fit=True, encoder_columns=None):
    """
    One-hot encode the low-cardinality categoricals.

    The original pipeline used LabelEncoder, which turns `location` into an
    arbitrary integer ordinal: it implies Ahmedabad < Bengaluru < Chennai, an
    ordering that carries no meaning. Linear models therefore could not use the
    5x international fraud lift at all, and trees had to waste splits
    reconstructing it. One-hot removes the fake ordering.

    `location` itself is intentionally dropped in favour of the `is_international`
    flag plus the per-location lift table, which keeps the matrix narrow and the
    signal explicit.
    """
    cat_cols = [c for c in ["transaction_type", "account_type", "payment_method", "device_type"]
                if c in df.columns]
    dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=True).astype("int64")

    if not fit and encoder_columns is not None:
        # Align an inference frame to the exact training column set.
        dummies = dummies.reindex(columns=encoder_columns, fill_value=0)

    return dummies


def assemble_matrix(df, customer_stats=None, fit=True, encoder_columns=None):
    """
    Produce the final (X, feature_names) pair used for training and inference.
    """
    feat = build_features(df, customer_stats=customer_stats)
    dummies = encode_categoricals(feat, fit=fit, encoder_columns=encoder_columns)

    base_cols = config.NUMERIC_FEATURES + config.FLAG_FEATURES + config.CUSTOMER_FEATURES
    missing = [c for c in base_cols if c not in feat.columns]
    if missing:
        raise ValueError(f"Feature builder did not produce expected columns: {missing}")

    X = pd.concat([feat[base_cols], dummies], axis=1)
    X = X.astype("float64")
    return X, list(X.columns)


def customer_stats_from_profiles(profiles_df):
    """
    Adapt the Spark customer aggregation table to the names used by the feature
    builder. Deliberately ignores `fraud_count`: a per-customer fraud tally
    computed over the full dataset includes the test rows' own labels, so using
    it as a predictor would leak the target.
    """
    if profiles_df is None or profiles_df.empty:
        return None

    stats = pd.DataFrame(index=profiles_df["customer_id"].values)
    stats.index.name = "customer_id"
    stats["cust_avg_amount"] = profiles_df["average_transaction_amount"].values
    stats["cust_txn_count"] = profiles_df["transaction_count"].values
    stats["cust_unique_merchants"] = profiles_df["unique_merchants"].values
    if "std_transaction_amount" in profiles_df.columns:
        stats["cust_std_amount"] = profiles_df["std_transaction_amount"].values
    else:
        # Older artefacts predate the std aggregation; the deviation feature
        # degrades to zero rather than failing.
        stats["cust_std_amount"] = 0.0
    return stats
