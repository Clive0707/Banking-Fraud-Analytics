"""
Tests for the imbalanced-classification metrics.

The point of these is that the metrics behave correctly on a 1%-prevalence
problem, where accuracy and ROC-AUC both mislead.
"""

import numpy as np
import pytest

from src.fraud_detection import metrics as m


@pytest.fixture
def imbalanced():
    """1,000 rows at 1% prevalence with a weakly informative score."""
    rng = np.random.RandomState(7)
    y = np.zeros(1000, dtype=int)
    y[:10] = 1
    rng.shuffle(y)
    # Positives score a little higher on average, with heavy overlap.
    score = rng.beta(2, 5, 1000) + y * 0.25
    return y, np.clip(score, 0, 1)


class TestNoSkillBaseline:
    def test_always_negative_accuracy_equals_one_minus_prevalence(self, imbalanced):
        y, _ = imbalanced
        base = m.no_skill_baseline(y)
        assert base["always_negative"]["accuracy"] == pytest.approx(1 - y.mean(), abs=1e-9)

    def test_always_negative_beats_a_weak_model_on_accuracy(self, imbalanced):
        """
        The headline finding the original benchmark missed: a constant predictor
        scores higher accuracy than a model operating at a balanced threshold.
        """
        y, score = imbalanced
        base = m.no_skill_baseline(y)
        model = m.metrics_at_threshold(y, score, 0.2)
        assert base["always_negative"]["accuracy"] > model["accuracy"]

    def test_random_scorer_pr_auc_is_prevalence(self, imbalanced):
        y, _ = imbalanced
        base = m.no_skill_baseline(y)
        assert base["random_scorer"]["pr_auc"] == pytest.approx(y.mean(), abs=1e-6)


class TestRankingMetrics:
    def test_perfect_scores_reach_one(self):
        y = np.array([0, 0, 1, 1])
        r = m.ranking_metrics(y, np.array([0.1, 0.2, 0.9, 0.95]))
        assert r["roc_auc"] == pytest.approx(1.0)
        assert r["pr_auc"] == pytest.approx(1.0)

    def test_pr_auc_lift_reported_against_random(self, imbalanced):
        y, score = imbalanced
        r = m.ranking_metrics(y, score)
        assert r["pr_auc_lift_over_random"] == pytest.approx(r["pr_auc"] / y.mean(), rel=1e-3)


class TestPrecisionAtK:
    def test_recall_is_monotonic_in_capacity(self, imbalanced):
        y, score = imbalanced
        rows = m.precision_at_k(y, score, fractions=[0.01, 0.05, 0.1, 0.5])
        recalls = [r["recall_at_k"] for r in rows]
        assert recalls == sorted(recalls)

    def test_perfect_ranking_gives_full_precision_at_top(self):
        y = np.array([1] * 10 + [0] * 990)
        score = np.concatenate([np.ones(10), np.zeros(990)])
        rows = m.precision_at_k(y, score, fractions=[0.01])
        assert rows[0]["precision_at_k"] == pytest.approx(1.0)
        assert rows[0]["lift"] == pytest.approx(100.0)

    def test_lift_is_one_for_an_uninformative_score(self):
        rng = np.random.RandomState(3)
        y = np.zeros(10000, dtype=int)
        y[:100] = 1
        rng.shuffle(y)
        rows = m.precision_at_k(y, rng.random_sample(10000), fractions=[0.5])
        # At half the portfolio a random ranking should land near baseline.
        assert rows[0]["lift"] == pytest.approx(1.0, abs=0.35)


class TestGainCurve:
    def test_curve_starts_at_origin_and_ends_complete(self, imbalanced):
        y, score = imbalanced
        points = m.gain_curve(y, score, n_bins=10)
        assert points[0]["reviewed_pct"] == 0 and points[0]["fraud_captured_pct"] == 0
        assert points[-1]["reviewed_pct"] == pytest.approx(100.0)
        assert points[-1]["fraud_captured_pct"] == pytest.approx(100.0, abs=0.01)

    def test_capture_is_monotonic(self, imbalanced):
        y, score = imbalanced
        captured = [p["fraud_captured_pct"] for p in m.gain_curve(y, score, n_bins=20)]
        assert captured == sorted(captured)


class TestCostCurve:
    def test_optimal_point_maximises_net_saving(self, imbalanced):
        y, score = imbalanced
        amounts = np.full(len(y), 50_000.0)
        res = m.cost_curve(y, score, amounts, n_points=25)
        best = max(r["net_saving"] for r in res["curve"])
        assert res["cost_optimal"]["net_saving"] == pytest.approx(best)

    def test_expensive_reviews_shrink_the_alert_budget(self, imbalanced):
        """Raising the review cost should never increase the number of alerts."""
        y, score = imbalanced
        amounts = np.full(len(y), 50_000.0)
        cheap = m.cost_curve(y, score, amounts, n_points=40, cost_per_review=10)
        dear = m.cost_curve(y, score, amounts, n_points=40, cost_per_review=5000)
        assert dear["cost_optimal"]["alerts_raised"] <= cheap["cost_optimal"]["alerts_raised"]

    def test_net_saving_accounting_is_consistent(self, imbalanced):
        y, score = imbalanced
        amounts = np.full(len(y), 1000.0)
        res = m.cost_curve(y, score, amounts, n_points=10,
                           cost_per_review=100, recovery_rate=0.5)
        for row in res["curve"]:
            expected = 0.5 * row["fraud_value_caught"] - 100 * row["alerts_raised"]
            assert row["net_saving"] == pytest.approx(expected, abs=0.01)


class TestThresholdSelection:
    def test_best_f1_threshold_beats_default(self, imbalanced):
        y, score = imbalanced
        t = m.best_f1_threshold(y, score)
        assert m.metrics_at_threshold(y, score, t)["f1_score"] >= \
               m.metrics_at_threshold(y, score, 0.5)["f1_score"]

    def test_confusion_matrix_totals_match(self, imbalanced):
        y, score = imbalanced
        res = m.metrics_at_threshold(y, score, 0.3)
        tn, fp = res["confusion_matrix"][0]
        fn, tp = res["confusion_matrix"][1]
        assert tn + fp + fn + tp == len(y)
        assert tp + fn == int(y.sum())


class TestEvaluateModel:
    def test_payload_contains_every_section(self, imbalanced):
        y, score = imbalanced
        out = m.evaluate_model(y, score, amounts=np.full(len(y), 1000.0))
        for key in ("ranking", "at_default_threshold", "at_best_f1_threshold",
                    "precision_at_k", "gain_curve", "roc_curve", "pr_curve",
                    "cost_analysis", "at_cost_optimal_threshold"):
            assert key in out, f"missing {key}"

    def test_curves_are_downsampled_for_transport(self, imbalanced):
        y, score = imbalanced
        out = m.evaluate_model(y, score)
        assert len(out["roc_curve"]) <= 120
        assert len(out["pr_curve"]) <= 120
