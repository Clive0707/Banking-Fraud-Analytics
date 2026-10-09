"""
Tests for the data service's startup contract.

These exist because a refactor once left the tail of `load_cache` stranded
inside another method, after a `return`. Everything downstream of the model
loop silently stopped running: transactions, customer profiles and the tuned
operating thresholds were never loaded. The dashboard answered "Transaction
data not loaded" and every customer lookup 404'd.

The existing API tests did not catch it because they *skip* when data is
absent, which is exactly the state the bug produced. These assert instead:
when an artefact is present on disk, the service must actually load it.
"""

import pytest

from backend.services.data_service import DataService
from src import config


@pytest.fixture(scope="module")
def service():
    return DataService()


def _artefacts_present():
    return (
        config.TRANSACTIONS_SAMPLE_PARQUET.exists()
        or config.TRANSACTIONS_PROCESSED_PARQUET.exists()
        or (config.PROCESSED_DIR / "transactions_by_month").exists()
    )


class TestLoadCacheCompletes:
    """
    Every stage of load_cache must run. A stage that silently stops leaves the
    object half-initialised, which is far harder to notice than a crash.
    """

    def test_transactions_are_loaded_when_the_file_exists(self, service):
        if not _artefacts_present():
            pytest.skip("no transaction artefact in this environment")
        assert service.transactions_df is not None, (
            "transaction artefact exists on disk but load_cache did not load it"
        )
        assert len(service.transactions_df) > 0

    def test_customer_profiles_are_loaded_when_the_file_exists(self, service):
        if not (config.CUSTOMER_PROFILES_PARQUET.exists()
                or config.CUSTOMER_PROFILES_CSV.exists()):
            pytest.skip("no customer profile artefact in this environment")
        assert service.customer_profiles_df is not None, (
            "customer profile artefact exists but load_cache did not load it"
        )
        assert len(service.customer_profiles_df) > 0

    def test_operating_thresholds_are_loaded(self, service):
        if not config.THRESHOLD_ANALYSIS_JSON.exists():
            pytest.skip("threshold analysis not generated")
        assert service.operating_thresholds, (
            "threshold analysis exists but no operating thresholds were loaded; "
            "predictions would silently fall back to 0.5"
        )

    def test_tuned_threshold_is_actually_used(self, service):
        """A loaded threshold must reach the scoring path, not just the dict."""
        if not service.operating_thresholds or not service.models:
            pytest.skip("models or thresholds unavailable")
        name = service.best_model_name
        if name not in service.operating_thresholds:
            pytest.skip(f"no tuned threshold for {name}")
        assert service._operating_threshold(name) == pytest.approx(
            service.operating_thresholds[name]
        )


class TestCustomerLookup:
    def test_a_known_customer_resolves(self, service):
        if service.customer_profiles_df is None:
            pytest.skip("customer profiles unavailable")
        cid = int(service.customer_profiles_df["customer_id"].iloc[0])
        profile = service.get_customer_profile(cid)
        assert "error" not in profile, profile.get("error")
        assert profile["customer_id"] == cid

    def test_an_unknown_customer_reports_not_found(self, service):
        if service.customer_profiles_df is None:
            pytest.skip("customer profiles unavailable")
        assert "error" in service.get_customer_profile(999_999_999)

    def test_a_non_numeric_id_is_rejected(self, service):
        assert "error" in service.get_customer_profile("not-a-number")


class TestTransactionListing:
    def test_unfiltered_listing_returns_rows(self, service):
        if service.transactions_df is None:
            pytest.skip("transaction data unavailable")
        res = service.get_transactions(page=1, per_page=10)
        assert "error" not in res
        assert res["total_records"] > 0
        assert len(res["data"]) == 10

    def test_listing_exposes_the_data_key_the_frontend_reads(self, service):
        """
        The explorer reads `data`. It once read `transactions`, which does not
        exist in the payload, so the table showed "no matching transactions"
        for every query.
        """
        if service.transactions_df is None:
            pytest.skip("transaction data unavailable")
        res = service.get_transactions(page=1, per_page=3)
        assert "data" in res
        assert isinstance(res["data"], list)


class TestCalibration:
    """
    Predicted probabilities must mean what they say.

    With 1% prevalence and balanced class weights the raw score is inflated
    roughly 39x: rows scored 0.75 carried a true fraud rate near 3%. The
    pipeline calibrates on a held-out split, and the benchmark records both the
    before and after so the correction is auditable.
    """

    def test_benchmark_reports_calibration(self, service):
        comp = service.model_comparison
        if not comp or not comp.get("models"):
            pytest.skip("benchmark not generated")
        best = comp["models"][comp["best_model"]]
        if "calibration" not in best:
            pytest.skip("artefacts predate calibration; retrain with run.py --train")
        assert "expected_calibration_error" in best["calibration"]
        assert "reliability_bins" in best["calibration"]

    def test_calibrated_scores_track_the_base_rate(self, service):
        comp = service.model_comparison
        if not comp or not comp.get("models"):
            pytest.skip("benchmark not generated")
        best = comp["models"][comp["best_model"]]
        cal = best.get("calibration")
        if not cal or cal.get("inflation_factor") is None:
            pytest.skip("artefacts predate calibration")
        # A calibrated model's average prediction should sit near the actual
        # prevalence. Uncalibrated it was about 39x too high.
        assert cal["inflation_factor"] < 3.0, (
            f"mean prediction is {cal['inflation_factor']}x the base rate; "
            f"scores are not calibrated"
        )

    def test_calibration_preserves_ranking(self, service):
        """Platt scaling is monotonic, so the model must still rank well."""
        comp = service.model_comparison
        if not comp or not comp.get("models"):
            pytest.skip("benchmark not generated")
        best = comp["models"][comp["best_model"]]
        prevalence = comp.get("no_skill_baseline", {}).get("prevalence")
        if not prevalence:
            pytest.skip("baseline unavailable")
        assert best["ranking"]["pr_auc"] > prevalence, (
            "calibration must not degrade ranking below a random scorer"
        )
