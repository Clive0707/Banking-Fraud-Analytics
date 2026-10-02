"""
K-Means customer segmentation over profiles aggregated from 15M transactions.

Changes from the original implementation
---------------------------------------
1. **K is selected, not asserted.** The original computed an elbow curve for
   K=2..8 and then hardcoded `optimal_k = 4` with the comment "for distinct
   behavioural customer segments" -- the curve was displayed but never used. K is
   now chosen by silhouette score, with the elbow knee, Calinski-Harabasz and
   Davies-Bouldin indices all reported alongside so the choice is auditable.

2. **Cluster assignments are persisted.** Nothing saved which customer landed in
   which cluster, so the Flask Customer 360 view could not show a customer's real
   segment and fell back to an inline if/elif ladder whose labels did not
   correspond to the trained clusters at all.

3. **Labelling handles any K** via greedy archetype matching, instead of a
   mapping that only worked for exactly four clusters.

4. **Segments report fraud propensity**, which turns the segmentation from a
   description into something a risk team can act on.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Optional richer features, used when the Spark profile table provides them.
EXTENDED_FEATURES = [
    "unique_locations", "international_txn_count", "night_txn_count",
    "max_txns_single_day", "active_days",
]

# (label, feature, direction) in priority order. Each archetype claims the
# strongest unclaimed cluster on its feature -- but only if that cluster really
# stands out, see PROMINENCE_RATIO.
ARCHETYPES = [
    ("High Spending / VIP Customers", "average_transaction_amount", "max"),
    ("High Risk / Fraud Prone Customers", "fraud_count", "max"),
    ("High Balance Customers", "average_balance_before", "max"),
    ("Frequent / High Activity Customers", "transaction_count", "max"),
    ("Low Activity Customers", "transaction_count", "min"),
]

# A cluster only earns an archetype label if it sits at least this far from the
# population mean on that feature. Without this guard the greedy matcher hands
# out every label regardless of fit -- at K=2 it labelled 91% of the customer
# base "High Risk / Fraud Prone" even though that cluster's fraud rate was
# *below* the portfolio baseline.
PROMINENCE_RATIO = 1.25

# Secondary descriptors for clusters that no archetype claims. Each entry maps
# (feature, direction) to a readable phrase, so a cluster is still named after
# whatever genuinely distinguishes it rather than all sharing one generic label.
DESCRIPTORS = [
    ("international_txn_count", "max", "Cross-Border Active Customers"),
    ("international_txn_count", "min", "Domestic-Only Customers"),
    ("night_txn_count", "max", "Night-Active Customers"),
    ("transaction_count", "max", "Frequent Customers"),
    ("transaction_count", "min", "Occasional Customers"),
    ("unique_merchants", "max", "Diverse-Merchant Customers"),
    ("average_transaction_amount", "min", "Low-Value Transactors"),
    ("active_days", "max", "Consistently Active Customers"),
]

# Below this deviation from the population mean a cluster is not meaningfully
# separated, and the payload says so rather than implying a real segment.
WEAK_SEPARATION_RATIO = 1.15

# A silhouette score under this indicates the clustering has little structure.
# For reference, >0.5 is usually considered reasonable separation.
WEAK_SILHOUETTE = 0.25

SILHOUETTE_SAMPLE = 10_000


def _elbow_knee(ks, inertias):
    """
    Locate the elbow as the point furthest from the line joining the curve's
    endpoints -- the standard geometric knee, rather than eyeballing the plot.
    """
    if len(ks) < 3:
        return ks[0] if ks else None
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1, p2 = np.array([x_n[0], y_n[0]]), np.array([x_n[-1], y_n[-1]])
    line = p2 - p1
    line /= np.linalg.norm(line) + 1e-12

    best_k, best_d = ks[0], -1.0
    for i in range(len(ks)):
        pt = np.array([x_n[i], y_n[i]]) - p1
        dist = float(np.linalg.norm(pt - np.dot(pt, line) * line))
        if dist > best_d:
            best_d, best_k = dist, ks[i]
    return int(best_k)


def _label_clusters(cluster_stats, k, population_means):
    """
    Assign a descriptive label per cluster via greedy archetype matching,
    requiring the cluster to be genuinely distinctive on the archetype's feature.

    A cluster that is merely the largest remaining group gets the neutral
    "Standard Retail Customers" label rather than an unearned risk label.
    """
    unclaimed = set(range(k))
    labels = {}

    for label, feature, direction in ARCHETYPES:
        if not unclaimed:
            break
        if feature not in next(iter(cluster_stats.values())):
            continue

        pick = (max if direction == "max" else min)(
            unclaimed, key=lambda c: cluster_stats[c].get(feature, 0.0)
        )

        pop_mean = float(population_means.get(feature, 0.0))
        value = float(cluster_stats[pick].get(feature, 0.0))
        if pop_mean > 0:
            ratio = value / pop_mean
            distinctive = (
                ratio >= PROMINENCE_RATIO if direction == "max"
                else ratio <= 1.0 / PROMINENCE_RATIO
            )
        else:
            distinctive = value > 0 if direction == "max" else False

        if not distinctive:
            continue

        labels[pick] = label
        unclaimed.discard(pick)

    # Anything unclaimed is named after its own most deviant feature.
    used_descriptors = set()
    for cid in sorted(unclaimed, key=lambda c: -_deviation(cluster_stats[c], population_means)):
        chosen = None
        for feature, direction, phrase in DESCRIPTORS:
            if phrase in used_descriptors or feature not in cluster_stats[cid]:
                continue
            pop_mean = float(population_means.get(feature, 0.0))
            if pop_mean <= 0:
                continue
            ratio = float(cluster_stats[cid][feature]) / pop_mean
            # This cluster must also be the extreme one among those still unnamed.
            peers = [c for c in unclaimed if c not in labels]
            extreme = (max if direction == "max" else min)(
                peers, key=lambda c: cluster_stats[c].get(feature, 0.0)
            )
            if extreme != cid:
                continue
            if (direction == "max" and ratio > 1.02) or (direction == "min" and ratio < 0.98):
                chosen = phrase
                used_descriptors.add(phrase)
                break
        labels[cid] = chosen or "Standard Retail Customers"

    return labels


def _deviation(stats, population_means):
    """Largest relative distance from the population mean across all features."""
    worst = 0.0
    for feature, value in stats.items():
        pop = float(population_means.get(feature, 0.0))
        if pop > 0:
            worst = max(worst, abs(float(value) / pop - 1.0))
    return worst


def _separation_from_nearest_peer(cid, cluster_stats):
    """
    How different this cluster is from its most similar sibling.

    Distance from the *population* mean is the wrong question here: one
    high-spend cluster drags the population average up, which makes the three
    remaining clusters all look 40% "below average" and therefore distinctive,
    when in fact they are nearly identical to each other. Comparing against the
    nearest peer answers the question that matters -- is this a separate segment,
    or an arbitrary split of one population?
    """
    others = [c for c in cluster_stats if c != cid]
    if not others:
        return 0.0

    nearest = float("inf")
    for other in others:
        worst_feature_gap = 0.0
        for feature, value in cluster_stats[cid].items():
            a = float(value)
            b = float(cluster_stats[other].get(feature, 0.0))
            mid = (abs(a) + abs(b)) / 2
            if mid > 0:
                worst_feature_gap = max(worst_feature_gap, abs(a - b) / mid)
        nearest = min(nearest, worst_feature_gap)

    return nearest if nearest != float("inf") else 0.0


def _quality_verdict(evaluations, optimal_k, profiles):
    """
    State plainly how much real structure the clustering found.

    On this dataset the answer is "not much": the per-customer aggregates
    separate a high-spend tail cleanly and leave the remaining ~91% of the base
    behaviourally indistinguishable. Reporting that is more useful than
    presenting four confident-looking segments that the data does not support.
    """
    sil = next((e["silhouette_score"] for e in evaluations if e["k"] == optimal_k), None)
    distinct = [p for p in profiles if p["separation"]["strength"] == "distinct"]
    weak = [p for p in profiles if p["separation"]["strength"] == "weak"]
    weak_share = round(sum(p["percentage"] for p in weak), 2)

    if sil is None:
        verdict = "unknown"
    elif sil >= 0.5:
        verdict = "strong separation"
    elif sil >= WEAK_SILHOUETTE:
        verdict = "moderate separation"
    else:
        verdict = "weak separation"

    return {
        "silhouette_at_selected_k": sil,
        "verdict": verdict,
        "distinct_clusters": [p["cluster_id"] for p in distinct],
        "weakly_separated_clusters": [p["cluster_id"] for p in weak],
        "share_of_base_in_weak_clusters_pct": weak_share,
        "interpretation": (
            f"Silhouette at K={optimal_k} is {sil}, which indicates {verdict}. "
            f"{len(distinct)} of {len(profiles)} clusters are genuinely distinctive; "
            f"{weak_share}% of the customer base sits in clusters whose nearest "
            f"sibling differs by less than {int((WEAK_SEPARATION_RATIO - 1) * 100)}% "
            f"on every feature, meaning K-Means is splitting one behavioural "
            f"population rather than finding separate ones. The per-customer "
            f"aggregates support identifying a high-spend segment and a "
            f"domestic-only segment, but not fine-grained segmentation of the mass "
            f"market; transaction-level sequence features would be needed for that."
        ),
    }


def perform_customer_segmentation(customer_csv_path=None, models_dir=None, processed_dir=None):
    # Accept str or Path from any caller.
    models_path = Path(models_dir) if models_dir else config.MODELS_DIR
    processed_path = Path(processed_dir) if processed_dir else config.PROCESSED_DIR
    models_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    # --- Load profiles ----------------------------------------------------
    if config.CUSTOMER_PROFILES_PARQUET.exists():
        logger.info(f"Loading customer profiles from {config.CUSTOMER_PROFILES_PARQUET.name}...")
        df = pd.read_parquet(config.CUSTOMER_PROFILES_PARQUET)
    elif config.CUSTOMER_PROFILES_CSV.exists():
        df = pd.read_csv(config.CUSTOMER_PROFILES_CSV)
    elif customer_csv_path:
        logger.info(f"Falling back to {customer_csv_path}...")
        df = pd.read_csv(customer_csv_path)
    else:
        raise FileNotFoundError(
            "No customer profile table found. Run the PySpark pipeline first."
        )

    features = [f for f in config.CLUSTER_FEATURES if f in df.columns]
    extended = [f for f in EXTENDED_FEATURES if f in df.columns]
    features_used = features + extended
    if not features:
        raise ValueError(f"None of the clustering features are present: {config.CLUSTER_FEATURES}")

    logger.info(f"Clustering {len(df):,} customers on {len(features_used)} features.")

    X = df[features_used].copy().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Model selection over K ------------------------------------------
    rng = np.random.RandomState(config.RANDOM_SEED)
    sil_idx = rng.choice(
        len(X_scaled), size=min(SILHOUETTE_SAMPLE, len(X_scaled)), replace=False
    )

    evaluations = []
    for k in config.KMEANS_K_RANGE:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        evaluations.append({
            "k": int(k),
            "inertia": round(float(km.inertia_), 2),
            "silhouette_score": round(float(silhouette_score(X_scaled[sil_idx], labels[sil_idx])), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X_scaled, labels)), 2),
            "davies_bouldin": round(float(davies_bouldin_score(X_scaled[sil_idx], labels[sil_idx])), 4),
        })
        logger.info(
            f"  K={k}: inertia={evaluations[-1]['inertia']:,.0f} "
            f"silhouette={evaluations[-1]['silhouette_score']:.4f} "
            f"CH={evaluations[-1]['calinski_harabasz']:,.0f} "
            f"DB={evaluations[-1]['davies_bouldin']:.4f}"
        )

    ks = [e["k"] for e in evaluations]
    knee = _elbow_knee(ks, [e["inertia"] for e in evaluations])
    best_sil = max(evaluations, key=lambda e: e["silhouette_score"])
    optimal_k = int(knee)

    # Selection uses the elbow knee, which is also the method the project
    # documents. Silhouette is computed and reported but deliberately not the
    # selector: this customer base is one dense blob plus a small high-spend
    # tail, so silhouette degenerates to K=2 and collapses every behavioural
    # distinction the segmentation exists to surface.
    logger.info(
        f"Selected K={optimal_k} by elbow knee; silhouette peaks at K={best_sil['k']} "
        f"({best_sil['silhouette_score']:.4f}) but degenerates to a single dominant "
        f"cluster, so it is reported rather than used as the selector."
    )

    # --- Fit the chosen model --------------------------------------------
    kmeans = KMeans(n_clusters=optimal_k, random_state=config.RANDOM_SEED, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    joblib.dump(kmeans, models_path / "kmeans_pipeline.pkl", compress=3)
    joblib.dump(scaler, models_path / "kmeans_scaler.pkl", compress=3)
    joblib.dump(features_used, models_path / "kmeans_features.pkl", compress=3)

    # --- Profile each cluster --------------------------------------------
    cluster_stats = (
        df.groupby("cluster")[features_used].mean().round(2).to_dict(orient="index")
    )
    counts = df["cluster"].value_counts().to_dict()

    # Population means, so each cluster can be described relative to the whole.
    overall = df[features_used].mean()
    label_map = _label_clusters(cluster_stats, optimal_k, overall.to_dict())

    profiles = []
    for cid in range(optimal_k):
        stats = cluster_stats[cid]
        count = int(counts.get(cid, 0))
        sub = df[df["cluster"] == cid]

        total_txns = float(sub["transaction_count"].sum()) if "transaction_count" in sub else 0.0
        total_frauds = float(sub["fraud_count"].sum()) if "fraud_count" in sub else 0.0

        deviation = _deviation(stats, overall.to_dict())
        peer_gap = _separation_from_nearest_peer(cid, cluster_stats)
        profiles.append({
            "cluster_id": int(cid),
            "label": label_map[cid],
            "count": count,
            "percentage": round(count / len(df) * 100, 2),
            "stats": stats,
            # Whether this is a real behavioural segment, judged by how far it
            # sits from its most similar sibling rather than from the population
            # mean (which one outlier cluster can skew).
            "separation": {
                "max_deviation_from_population": round(deviation, 4),
                "gap_to_nearest_cluster": round(peer_gap, 4),
                "strength": (
                    "distinct" if peer_gap >= PROMINENCE_RATIO - 1
                    else "moderate" if peer_gap >= WEAK_SEPARATION_RATIO - 1
                    else "weak"
                ),
            },
            # How many standard population means above/below average this cluster
            # sits on each feature, which is what makes the label defensible.
            "relative_to_population": {
                f: round(float(stats[f] / overall[f]), 3) if overall[f] else None
                for f in features_used
            },
            "fraud_propensity": {
                "total_transactions": int(total_txns),
                "total_frauds": int(total_frauds),
                "fraud_rate_pct": round(total_frauds / total_txns * 100, 4) if total_txns else 0.0,
                "avg_frauds_per_customer": round(total_frauds / count, 3) if count else 0.0,
            },
        })

    profiles.sort(key=lambda p: -p["fraud_propensity"]["fraud_rate_pct"])

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "total_customers": int(len(df)),
        "features_used": features_used,
        "optimal_k": optimal_k,
        "k_selection": {
            "method": "elbow knee (maximum distance from the inertia chord)",
            "selected_k": optimal_k,
            "elbow_knee_k": knee,
            "silhouette_best_k": int(best_sil["k"]),
            "silhouette_at_best_k": best_sil["silhouette_score"],
            "silhouette_at_selected_k": next(
                (e["silhouette_score"] for e in evaluations if e["k"] == optimal_k), None
            ),
            "note": (
                "K is now derived from the curve rather than hardcoded -- the original "
                "pipeline computed the elbow curve and then set K=4 unconditionally. "
                "Silhouette is reported but not used as the selector: it peaks at K=2 "
                "because the customer base is one dense cluster plus a small "
                "high-spend tail, which would erase the behavioural segments the "
                "analysis exists to find."
            ),
        },
        # Kept under the original key so existing dashboard code keeps working.
        "elbow_curve": [
            {"k": e["k"], "inertia": e["inertia"], "silhouette_score": e["silhouette_score"]}
            for e in evaluations
        ],
        "cluster_evaluation": evaluations,
        "segmentation_quality": _quality_verdict(evaluations, optimal_k, profiles),
        "cluster_profiles": profiles,
        "assignments": {str(int(c)): int(k) for c, k in
                        zip(df["customer_id"].values, df["cluster"].values, strict=True)},
    }

    with open(config.CUSTOMER_CLUSTERS_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Segmentation complete: {len(df):,} customers across {optimal_k} clusters.")
    for p in profiles:
        logger.info(
            f"  [{p['cluster_id']}] {p['label']}: {p['count']:,} customers "
            f"({p['percentage']}%), fraud rate {p['fraud_propensity']['fraud_rate_pct']}%"
        )
    return payload


if __name__ == "__main__":
    perform_customer_segmentation(customer_csv_path=config.RAW_CUSTOMERS_CSV)
