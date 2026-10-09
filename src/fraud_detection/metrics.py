"""
Evaluation metrics appropriate for a 0.96%-prevalence fraud problem.

Why this module exists
----------------------
The original benchmark reported accuracy, precision, recall, F1 and ROC-AUC at a
fixed 0.5 threshold. For this dataset that is actively misleading:

* **Accuracy is worse than useless.** Predicting "never fraud" scores 99.04%
  accuracy. Every trained model scored 76-81%, i.e. every model looked *worse
  than a constant predictor* on the headline metric.
* **ROC-AUC is over-optimistic under heavy imbalance.** The false-positive rate
  denominator is dominated by the 99% negative class, so large absolute numbers
  of false alarms barely move the curve. PR-AUC (average precision) is the
  honest summary.
* **A fixed 0.5 threshold is arbitrary.** With `class_weight="balanced"` the
  scores are already re-centred, so 0.5 produced 21,356 false positives to catch
  611 frauds. The threshold is a business decision, not a default.

What replaces it
----------------
* PR-AUC plus an explicit no-skill baseline for every metric.
* precision@k / recall@k at realistic analyst review capacities, which is how a
  fraud team actually consumes a risk score.
* Lift and cumulative-gain curves: the model's real value is *ranking*.
* A cost curve that converts each threshold into expected rupees saved, and the
  cost-optimal operating point that falls out of it.
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config

logger = logging.getLogger(__name__)

# A reliability bin needs this many rows before its probability is treated as
# evidence-backed rather than extrapolation.
MIN_BIN_SUPPORT = 30


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def no_skill_baseline(y_true):
    """
    Metrics for the trivial classifiers, so every model number has a reference.

    `always_negative` is the one that matters: it is what "99% accuracy" really
    means on an imbalanced problem.
    """
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    positives = int(y_true.sum())
    prevalence = positives / n if n else 0.0

    return {
        "prevalence": round(prevalence, 6),
        "prevalence_pct": round(prevalence * 100, 4),
        "positives": positives,
        "negatives": n - positives,
        "always_negative": {
            "accuracy": round(1.0 - prevalence, 6),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        },
        "always_positive": {
            "accuracy": round(prevalence, 6),
            "precision": round(prevalence, 6),
            "recall": 1.0,
            "f1_score": round(2 * prevalence / (1 + prevalence), 6) if prevalence else 0.0,
        },
        # A random scorer's PR-AUC equals the prevalence; ROC-AUC equals 0.5.
        "random_scorer": {"pr_auc": round(prevalence, 6), "roc_auc": 0.5},
    }


# ---------------------------------------------------------------------------
# Threshold-free summaries
# ---------------------------------------------------------------------------

def ranking_metrics(y_true, y_score):
    """ROC-AUC and PR-AUC, the two threshold-independent summaries."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    prevalence = y_true.mean() if len(y_true) else 0.0
    pr_auc = float(average_precision_score(y_true, y_score))

    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6),
        "pr_auc": round(pr_auc, 6),
        # How many times better than a random scorer, whose PR-AUC is the prevalence.
        "pr_auc_lift_over_random": round(pr_auc / prevalence, 4) if prevalence else 0.0,
    }


def metrics_at_threshold(y_true, y_score, threshold):
    """Point metrics for one operating threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts_raised": int(tp + fp),
        # The number a fraud analyst asks for first.
        "alerts_per_true_fraud": round(float((tp + fp) / tp), 2) if tp else None,
    }


def best_f1_threshold(y_true, y_score):
    """The threshold maximising F1, found by sweeping the PR curve."""
    y_true = np.asarray(y_true).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more element than thresholds.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    if len(f1) == 0 or np.all(np.isnan(f1)):
        return 0.5
    return float(thresholds[int(np.nanargmax(f1))])


# ---------------------------------------------------------------------------
# Capacity-based metrics: precision@k, lift, cumulative gain
# ---------------------------------------------------------------------------

def precision_at_k(y_true, y_score, fractions=None):
    """
    Precision and recall when only the top-k riskiest transactions are reviewed.

    This is the metric that makes a weak classifier useful. Even a model with an
    F1 of 0.06 is worth deploying if reviewing the riskiest 1% of transactions
    surfaces several times more fraud than reviewing 1% at random.
    """
    fractions = fractions or config.REVIEW_CAPACITY_FRACTIONS
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    n = len(y_true)
    total_pos = int(y_true.sum())
    prevalence = total_pos / n if n else 0.0

    order = np.argsort(-y_score, kind="mergesort")  # stable: ties keep input order
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)

    rows = []
    for frac in fractions:
        k = max(1, int(round(frac * n)))
        k = min(k, n)
        tp = int(cum_tp[k - 1])
        prec = tp / k
        rows.append({
            "capacity_fraction": frac,
            "capacity_pct": round(frac * 100, 3),
            "reviewed": k,
            "frauds_caught": tp,
            "precision_at_k": round(prec, 6),
            "recall_at_k": round(tp / total_pos, 6) if total_pos else 0.0,
            # Lift: how much richer this slice is in fraud than the portfolio.
            "lift": round(prec / prevalence, 4) if prevalence else 0.0,
        })
    return rows


def gain_curve(y_true, y_score, n_bins=20):
    """
    Cumulative gain curve: fraud captured versus portfolio reviewed.

    Returned at `n_bins` points so it can be shipped straight to Chart.js.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    total_pos = int(y_true.sum())

    order = np.argsort(-y_score, kind="mergesort")
    cum_tp = np.cumsum(y_true[order])

    points = [{"reviewed_pct": 0.0, "fraud_captured_pct": 0.0, "lift": 0.0}]
    for i in range(1, n_bins + 1):
        frac = i / n_bins
        k = max(1, int(round(frac * n)))
        captured = cum_tp[k - 1] / total_pos if total_pos else 0.0
        points.append({
            "reviewed_pct": round(frac * 100, 2),
            "fraud_captured_pct": round(float(captured) * 100, 3),
            # Against the diagonal, where reviewing x% catches x% of fraud.
            "lift": round(float(captured) / frac, 4) if frac else 0.0,
        })
    return points


# ---------------------------------------------------------------------------
# Cost-sensitive thresholding
# ---------------------------------------------------------------------------

def cost_curve(y_true, y_score, amounts, n_points=60,
               cost_per_review=None, recovery_rate=None):
    """
    Translate every threshold into expected rupees saved.

    Model
    -----
    Flagging a transaction costs `cost_per_review` in analyst time, whether or
    not it turns out to be fraud. Catching a fraud recovers `recovery_rate` of
    its value; missing one loses the full value.

        net_saving(t) = recovery_rate * value_of_frauds_caught(t)
                        - cost_per_review * alerts_raised(t)

    The do-nothing policy saves zero by construction, so a positive net saving
    is the amount the model earns over not deploying it at all.
    """
    cost_per_review = config.COST_PER_REVIEW if cost_per_review is None else cost_per_review
    recovery_rate = config.RECOVERY_RATE if recovery_rate is None else recovery_rate

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    total_fraud_value = float(amounts[y_true == 1].sum())

    lo, hi = float(np.min(y_score)), float(np.max(y_score))
    if hi <= lo:
        thresholds = np.array([lo])
    else:
        thresholds = np.linspace(lo, hi, n_points)

    curve = []
    for t in thresholds:
        flagged = y_score >= t
        n_alerts = int(flagged.sum())
        caught_value = float(amounts[flagged & (y_true == 1)].sum())
        review_cost = cost_per_review * n_alerts
        net = recovery_rate * caught_value - review_cost
        curve.append({
            "threshold": round(float(t), 6),
            "alerts_raised": n_alerts,
            "frauds_caught": int((flagged & (y_true == 1)).sum()),
            "fraud_value_caught": round(caught_value, 2),
            "review_cost": round(review_cost, 2),
            "net_saving": round(net, 2),
        })

    best = max(curve, key=lambda r: r["net_saving"]) if curve else None

    return {
        "assumptions": {
            "cost_per_review_inr": cost_per_review,
            "recovery_rate": recovery_rate,
            "total_fraud_value_in_test_set": round(total_fraud_value, 2),
        },
        "curve": curve,
        "cost_optimal": best,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_report(y_true, y_score, n_bins=10):
    """
    How close the predicted score is to an actual probability.

    This matters because `class_weight="balanced"` re-weights the 1% positive
    class roughly 100x, which shifts the log-odds by a constant and inflates
    every output. The raw score is a good *ranking* but a poor *probability*:
    uncalibrated, rows scored 0.75 carried a true fraud rate near 3%. Reporting
    a score of 0.99 as "99% probability of fraud" is simply wrong, so the
    pipeline calibrates and this quantifies the result.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    mean_predicted = float(y_score.mean()) if len(y_score) else 0.0
    brier = float(np.mean((y_score - y_true) ** 2)) if len(y_true) else 0.0

    # Equal-width reliability bins.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    gaps = []
    weights = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (y_score >= lo) & (y_score < hi if i < n_bins - 1 else y_score <= hi)
        n = int(sel.sum())
        if n == 0:
            continue
        pred = float(y_score[sel].mean())
        actual = float(y_true[sel].mean())
        bins.append({
            "bin_start": round(float(lo), 3),
            "bin_end": round(float(hi), 3),
            "count": n,
            "mean_predicted": round(pred, 6),
            "actual_fraud_rate": round(actual, 6),
            "gap": round(pred - actual, 6),
        })
        gaps.append(abs(pred - actual))
        weights.append(n)

    # Expected calibration error: bin gaps weighted by population.
    ece = float(np.average(gaps, weights=weights)) if gaps else 0.0

    # The highest score the test set actually supports. Calibration can only
    # correct regions it observed: on this dataset 99.8% of rows score under
    # 0.1, and above 0.5 there are 3 rows in 99,937. A prediction beyond this
    # point is extrapolation, however well calibrated the bulk of the range is,
    # and the dashboard labels it as such rather than quoting false confidence.
    supported = [b for b in bins if b["count"] >= MIN_BIN_SUPPORT]
    supported_max = supported[-1]["bin_end"] if supported else 0.0

    return {
        "brier_score": round(brier, 6),
        "expected_calibration_error": round(ece, 6),
        "supported_max_probability": round(float(supported_max), 4),
        "min_rows_for_support": MIN_BIN_SUPPORT,
        "mean_predicted": round(mean_predicted, 6),
        "actual_prevalence": round(prevalence, 6),
        # >1 means the model overstates fraud probability on average.
        "inflation_factor": round(mean_predicted / prevalence, 3) if prevalence else None,
        "reliability_bins": bins,
    }


# ---------------------------------------------------------------------------
# Curve exports for the dashboard
# ---------------------------------------------------------------------------

def roc_points(y_true, y_score, max_points=120):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = _downsample_index(len(fpr), max_points)
    return [{"fpr": round(float(fpr[i]), 5), "tpr": round(float(tpr[i]), 5)} for i in idx]


def pr_points(y_true, y_score, max_points=120):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    idx = _downsample_index(len(precision), max_points)
    return [{"recall": round(float(recall[i]), 5), "precision": round(float(precision[i]), 5)}
            for i in idx]


def _downsample_index(length, max_points):
    """Evenly spaced indices, always keeping the first and last point."""
    if length <= max_points:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, max_points).astype(int).tolist()))


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_score, amounts=None, default_threshold=0.5):
    """
    Produce the complete evaluation payload for one model.

    Reports the default 0.5 threshold (for continuity with the original
    benchmark), the F1-optimal threshold, and -- when transaction amounts are
    supplied -- the cost-optimal threshold.
    """
    payload = {
        "ranking": ranking_metrics(y_true, y_score),
        "calibration": calibration_report(y_true, y_score),
        "at_default_threshold": metrics_at_threshold(y_true, y_score, default_threshold),
        "precision_at_k": precision_at_k(y_true, y_score),
        "gain_curve": gain_curve(y_true, y_score),
        "roc_curve": roc_points(y_true, y_score),
        "pr_curve": pr_points(y_true, y_score),
    }

    t_f1 = best_f1_threshold(y_true, y_score)
    payload["at_best_f1_threshold"] = metrics_at_threshold(y_true, y_score, t_f1)

    if amounts is not None:
        cost = cost_curve(y_true, y_score, amounts)
        payload["cost_analysis"] = cost
        if cost["cost_optimal"]:
            t_cost = cost["cost_optimal"]["threshold"]
            payload["at_cost_optimal_threshold"] = metrics_at_threshold(y_true, y_score, t_cost)
            payload["at_cost_optimal_threshold"]["net_saving"] = cost["cost_optimal"]["net_saving"]

    return payload
