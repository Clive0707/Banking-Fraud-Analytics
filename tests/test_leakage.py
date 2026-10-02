"""
Tests for the target-leakage audit.

The audit is a standing control, not a one-off observation: if a future dataset
change reintroduces a column that predicts the target almost perfectly, these
tests ensure it is detected and reported rather than silently trained on.
"""

import pytest

from src import config
from src.fraud_detection import leakage_analysis as la


class TestStatusLeakageReport:
    def test_detects_the_leak(self, synthetic_transactions):
        report = la.status_leakage_report(synthetic_transactions)
        assert report["available"]
        assert "LEAKING" in report["verdict"]

    def test_rule_has_perfect_precision(self, synthetic_transactions):
        """Declined and Flagged occur only on fraud, so precision must be 1.0."""
        rule = la.status_leakage_report(synthetic_transactions)["zero_model_rule"]
        assert rule["precision"] == pytest.approx(1.0)
        assert rule["false_positives"] == 0

    def test_breakdown_covers_every_status(self, synthetic_transactions):
        report = la.status_leakage_report(synthetic_transactions)
        seen = {b["status"] for b in report["breakdown"]}
        assert seen == set(synthetic_transactions["transaction_status"].unique())

    def test_missing_column_is_handled(self, synthetic_transactions):
        df = synthetic_transactions.drop(columns=["transaction_status"])
        assert la.status_leakage_report(df)["available"] is False


class TestPurityScan:
    def test_flags_the_pure_levels(self, synthetic_transactions):
        scan = la.scan_categorical_purity(synthetic_transactions)
        flagged = {(f["column"], f["level"]) for f in scan["suspicious_levels"]}
        assert ("transaction_status", "Flagged") in flagged
        assert ("transaction_status", "Declined") in flagged

    def test_does_not_flag_ordinary_columns(self, synthetic_transactions):
        scan = la.scan_categorical_purity(synthetic_transactions)
        cols = {f["column"] for f in scan["suspicious_levels"]}
        assert "location" not in cols
        assert "payment_method" not in cols

    def test_clean_dataset_produces_no_findings(self, synthetic_transactions):
        df = synthetic_transactions.drop(columns=["transaction_status"])
        assert la.scan_categorical_purity(df)["suspicious_levels"] == []


class TestLeakyVsClean:
    def test_leak_inflates_the_metrics(self, synthetic_transactions):
        res = la.leaky_vs_clean_comparison(synthetic_transactions)
        assert res["leaky_with_status"]["pr_auc"] > res["clean_no_status"]["pr_auc"]
        assert res["inflation"]["pr_auc_multiple"] > 1.0

    def test_leaky_model_uses_more_features(self, synthetic_transactions):
        res = la.leaky_vs_clean_comparison(synthetic_transactions)
        assert res["leaky_with_status"]["n_features"] > res["clean_no_status"]["n_features"]


class TestFullAudit:
    def test_writes_a_complete_report(self, synthetic_transactions, tmp_path):
        out = tmp_path / "leakage_report.json"
        report = la.run_leakage_analysis(synthetic_transactions, output_path=out)
        assert out.exists()
        for key in ("status_leakage", "categorical_purity_scan",
                    "excluded_from_features", "leaky_vs_clean"):
            assert key in report
        assert report["excluded_from_features"] == config.LEAKING_COLS
