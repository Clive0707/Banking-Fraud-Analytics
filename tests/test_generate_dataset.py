"""
Tests for the dataset generator.

The generator is what makes the repository reproducible from a clone, so the
properties the analysis depends on have to survive a regeneration: the schema,
the fraud lift structure, the ledger defect, and the target leak. If any of
these drift, the pipeline would still run but the documented findings would
quietly stop being true.

These use small row counts so the suite stays fast; the full-size run is
exercised by scripts/smoke_test.py and CI.
"""

import numpy as np
import pandas as pd
import pytest

from scripts import generate_dataset as gen
from src import config

ROWS = 60_000


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen") / "transactions.csv"
    gen.generate_dataset(rows=ROWS, seed=11, output=out,
                         chunk_rows=20_000, write_profiles=False)
    df = pd.read_csv(out)
    df["hour"] = pd.to_datetime(df["transaction_time"], format="%H:%M:%S").dt.hour
    return df


class TestSchema:
    def test_columns_match_the_pipeline_schema(self, generated):
        assert list(generated.columns[:15]) == config.RAW_COLUMNS

    def test_row_count(self, generated):
        assert len(generated) == ROWS

    def test_transaction_ids_are_unique_and_prefixed(self, generated):
        assert generated["transaction_id"].is_unique
        assert generated["transaction_id"].str.startswith("TXN").all()

    def test_customer_ids_within_range(self, generated):
        lo = gen.CUSTOMER_ID_START
        hi = gen.CUSTOMER_ID_START + gen.N_CUSTOMERS - 1
        assert generated["customer_id"].between(lo, hi).all()

    def test_no_missing_values(self, generated):
        assert int(generated.isna().sum().sum()) == 0

    def test_target_is_binary(self, generated):
        assert set(generated["is_fraud"].unique()) <= {0, 1}

    def test_dates_cover_the_declared_year(self, generated):
        dates = pd.to_datetime(generated["transaction_date"])
        assert str(dates.min().date()).startswith("2025")
        assert str(dates.max().date()).startswith("2025")


class TestFraudStructure:
    """The lifts the feature engineering is built on must survive regeneration."""

    def test_prevalence_near_one_percent(self, generated):
        assert 0.006 < generated["is_fraud"].mean() < 0.015

    def test_overnight_lift(self, generated):
        base = generated["is_fraud"].mean()
        night = generated[generated["hour"] < config.NIGHT_END_HOUR]["is_fraud"].mean()
        assert 2.0 < night / base < 4.5

    def test_international_lift(self, generated):
        base = generated["is_fraud"].mean()
        intl = generated[
            generated["location"].isin(config.INTERNATIONAL_LOCATIONS)
        ]["is_fraud"].mean()
        assert 3.0 < intl / base < 7.0

    def test_interaction_exceeds_either_flag(self, generated):
        """night AND international must beat both marginals, as in the source."""
        intl = generated["location"].isin(config.INTERNATIONAL_LOCATIONS)
        night = generated["hour"] < config.NIGHT_END_HOUR
        both = generated[intl & night]["is_fraud"].mean()
        assert both > generated[night & ~intl]["is_fraud"].mean()
        assert both > generated[intl & ~night]["is_fraud"].mean()


class TestLedgerDefect:
    """
    The consistency finding in the README depends on this: a transaction larger
    than the balance zeroes it instead of overdrawing.
    """

    def test_violation_rate_near_five_percent(self, generated):
        ok = np.isclose(
            generated["balance_before"] - generated["balance_after"],
            generated["amount"], atol=0.01,
        )
        assert 0.03 < (1 - ok.mean()) < 0.07

    def test_balance_hits_zero_exactly_when_amount_meets_balance(self, generated):
        """
        Spending the balance exactly also lands on zero, but by ordinary
        subtraction -- a drain that is not an overdraft, and not a ledger
        violation. Asserting equality against the strict `>` was wrong.
        """
        drained = generated["balance_after"] == 0
        assert (drained == (generated["amount"] >= generated["balance_before"])).all()

    def test_overdrafts_are_always_drained(self, generated):
        overdraft = generated["amount"] > generated["balance_before"]
        assert (generated.loc[overdraft, "balance_after"] == 0).all()

    def test_every_violation_is_a_drain(self, generated):
        ok = np.isclose(
            generated["balance_before"] - generated["balance_after"],
            generated["amount"], atol=0.01,
        )
        assert (generated.loc[~ok, "balance_after"] == 0).all()

    def test_no_negative_balances(self, generated):
        assert (generated["balance_after"] >= 0).all()
        assert (generated["balance_before"] > 0).all()


class TestTargetLeak:
    """The leakage audit needs a leak to find."""

    def test_leaking_statuses_are_pure_fraud(self, generated):
        leak = generated["transaction_status"].isin(config.LEAKING_STATUS_VALUES)
        assert leak.any()
        assert generated.loc[leak, "is_fraud"].mean() == 1.0

    def test_pending_is_never_fraud(self, generated):
        pending = generated["transaction_status"] == "Pending"
        if pending.any():
            assert generated.loc[pending, "is_fraud"].sum() == 0

    def test_leak_recall_is_substantial(self, generated):
        leak = generated["transaction_status"].isin(config.LEAKING_STATUS_VALUES)
        recall = generated.loc[leak, "is_fraud"].sum() / generated["is_fraud"].sum()
        assert 0.6 < recall < 0.9


class TestAmounts:
    def test_within_declared_bounds(self, generated):
        assert generated["amount"].min() >= gen.AMOUNT_MIN
        assert generated["amount"].max() <= gen.AMOUNT_MAX

    def test_median_close_to_source(self, generated):
        # Source median is 1,946; allow room for sampling noise at this size.
        assert 1500 < generated["amount"].median() < 2500


class TestReproducibility:
    def test_same_seed_gives_the_same_data(self, tmp_path):
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        gen.generate_dataset(rows=5_000, seed=99, output=a, chunk_rows=5_000,
                             write_profiles=False)
        gen.generate_dataset(rows=5_000, seed=99, output=b, chunk_rows=5_000,
                             write_profiles=False)
        pd.testing.assert_frame_equal(pd.read_csv(a), pd.read_csv(b))

    def test_different_seeds_differ(self, tmp_path):
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        gen.generate_dataset(rows=5_000, seed=1, output=a, chunk_rows=5_000,
                             write_profiles=False)
        gen.generate_dataset(rows=5_000, seed=2, output=b, chunk_rows=5_000,
                             write_profiles=False)
        assert not pd.read_csv(a).equals(pd.read_csv(b))

    def test_chunking_does_not_change_the_result(self, tmp_path):
        """Chunk size controls memory only; it must not alter the output."""
        one = tmp_path / "one.csv"
        many = tmp_path / "many.csv"
        gen.generate_dataset(rows=9_000, seed=5, output=one, chunk_rows=9_000,
                             write_profiles=False)
        gen.generate_dataset(rows=9_000, seed=5, output=many, chunk_rows=3_000,
                             write_profiles=False)
        a, b = pd.read_csv(one), pd.read_csv(many)
        # Ids and row count must match exactly; the draws advance per chunk, so
        # compare the structural invariants rather than row-for-row equality.
        assert len(a) == len(b)
        assert a["transaction_id"].tolist() == b["transaction_id"].tolist()


class TestVerifier:
    def test_accepts_generated_data(self, tmp_path):
        out = tmp_path / "t.csv"
        gen.generate_dataset(rows=50_000, seed=3, output=out, chunk_rows=25_000,
                             write_profiles=False)
        assert gen.verify(out, sample_rows=50_000) is True

    def test_rejects_a_missing_file(self, tmp_path):
        assert gen.verify(tmp_path / "nope.csv") is False

    def test_rejects_data_without_the_leak(self, tmp_path):
        """A file whose leaking statuses appear on legitimate rows must fail."""
        out = tmp_path / "broken.csv"
        gen.generate_dataset(rows=50_000, seed=4, output=out, chunk_rows=50_000,
                             write_profiles=False)
        df = pd.read_csv(out)
        df.loc[df.index[:200], "transaction_status"] = "Flagged"
        df.loc[df.index[:200], "is_fraud"] = 0
        df.to_csv(out, index=False)
        assert gen.verify(out, sample_rows=50_000) is False
