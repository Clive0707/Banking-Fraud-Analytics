"""
Feature engineering tests.

These guard the invariants that matter most: the target never leaks into the
feature matrix, and the training and serving paths produce identical columns.
"""

import numpy as np
import pandas as pd

from src import config
from src import features as feat_mod


class TestRiskFlags:
    def test_night_flag_covers_midnight_to_five(self, synthetic_transactions):
        feat = feat_mod.build_features(synthetic_transactions)
        night = feat[feat["is_night"] == 1]["hour"]
        day = feat[feat["is_night"] == 0]["hour"]
        assert night.max() < config.NIGHT_END_HOUR
        assert day.min() >= config.NIGHT_END_HOUR

    def test_hour_23_is_not_night(self):
        """
        The original rule was `hour < 5 or hour > 23`, whose second clause can
        never fire because hours top out at 23.
        """
        df = pd.DataFrame([{
            "transaction_id": "T1", "customer_id": 1,
            "transaction_date": "2025-06-15", "transaction_time": "23:30:00",
            "transaction_type": "UPI", "account_type": "Savings",
            "amount": 100.0, "balance_before": 1000.0, "balance_after": 900.0,
            "merchant": "Amazon", "location": "Mumbai",
            "payment_method": "UPI", "device_type": "Android",
        }])
        assert feat_mod.build_features(df)["is_night"].iloc[0] == 0

    def test_international_flag_matches_config(self, synthetic_transactions):
        feat = feat_mod.build_features(synthetic_transactions)
        flagged = set(feat[feat["is_international"] == 1]["location"].unique())
        assert flagged <= config.INTERNATIONAL_LOCATIONS

    def test_high_amount_flag_uses_threshold(self, synthetic_transactions):
        feat = feat_mod.build_features(synthetic_transactions)
        assert (feat.loc[feat["is_high_amount"] == 1, "amount"] > config.HIGH_AMOUNT_THRESHOLD).all()
        assert (feat.loc[feat["is_high_amount"] == 0, "amount"] <= config.HIGH_AMOUNT_THRESHOLD).all()

    def test_interactions_are_products(self, synthetic_transactions):
        feat = feat_mod.build_features(synthetic_transactions)
        assert (feat["intl_x_night"] == feat["is_international"] * feat["is_night"]).all()
        assert (feat["night_x_high"] == feat["is_night"] * feat["is_high_amount"]).all()


class TestNoLeakage:
    def test_target_absent_from_matrix(self, synthetic_transactions, customer_stats):
        X, names = feat_mod.assemble_matrix(synthetic_transactions, customer_stats=customer_stats)
        assert config.TARGET_COL not in names
        assert not any(config.TARGET_COL in n for n in names)

    def test_transaction_status_absent_from_matrix(self, synthetic_transactions, customer_stats):
        """transaction_status predicts the target perfectly and must never appear."""
        _, names = feat_mod.assemble_matrix(synthetic_transactions, customer_stats=customer_stats)
        assert not any("status" in n.lower() for n in names)

    def test_customer_stats_exclude_fraud_count(self):
        profiles = pd.DataFrame({
            "customer_id": [1, 2],
            "average_transaction_amount": [100.0, 200.0],
            "transaction_count": [10, 20],
            "unique_merchants": [3, 4],
            "fraud_count": [5, 0],
        })
        stats = feat_mod.customer_stats_from_profiles(profiles)
        assert "fraud_count" not in stats.columns


class TestTrainServeConsistency:
    def test_single_row_matches_batch_columns(self, synthetic_transactions, customer_stats):
        """
        A one-row inference frame must produce the exact training column set.
        This is the invariant the old inline feature code in the Flask service
        could not guarantee.
        """
        X_train, names = feat_mod.assemble_matrix(
            synthetic_transactions, customer_stats=customer_stats
        )
        one = synthetic_transactions.head(1)
        X_one, _ = feat_mod.assemble_matrix(one, customer_stats=customer_stats, fit=False)
        X_one = X_one.reindex(columns=names, fill_value=0.0)
        assert list(X_one.columns) == list(names)
        assert X_one.shape[1] == X_train.shape[1]

    def test_unseen_category_does_not_crash(self, customer_stats):
        df = pd.DataFrame([{
            "transaction_id": "T1", "customer_id": 999999,
            "transaction_date": "2025-06-15", "transaction_time": "12:00:00",
            "transaction_type": "Crypto Swap",     # never seen in training
            "account_type": "Offshore",
            "amount": 500.0, "balance_before": 1000.0, "balance_after": 500.0,
            "merchant": "Unknown", "location": "Reykjavik",
            "payment_method": "Wire", "device_type": "Linux",
        }])
        X, _ = feat_mod.assemble_matrix(df, customer_stats=customer_stats, fit=False)
        assert len(X) == 1
        assert np.isfinite(X.values).all()

    def test_no_infinities_from_zero_balances(self, customer_stats):
        df = pd.DataFrame([{
            "transaction_id": "T1", "customer_id": 1,
            "transaction_date": "2025-06-15", "transaction_time": "12:00:00",
            "transaction_type": "UPI", "account_type": "Savings",
            "amount": 5000.0, "balance_before": 0.0, "balance_after": 0.0,
            "merchant": "Amazon", "location": "Mumbai",
            "payment_method": "UPI", "device_type": "Android",
        }])
        X, _ = feat_mod.assemble_matrix(df, customer_stats=customer_stats, fit=False)
        assert np.isfinite(X.values).all()


class TestDerivedValues:
    def test_zero_balance_flag(self, synthetic_transactions):
        feat = feat_mod.build_features(synthetic_transactions)
        assert (feat.loc[feat["zero_balance_after"] == 1, "balance_after"] == 0).all()

    def test_balance_change_is_difference(self, synthetic_transactions):
        feat = feat_mod.build_features(synthetic_transactions)
        expected = feat["balance_before"] - feat["balance_after"]
        pd.testing.assert_series_equal(feat["balance_change"], expected, check_names=False)

    def test_amount_log_is_monotonic(self, synthetic_transactions):
        feat = feat_mod.build_features(synthetic_transactions)
        order_amount = feat["amount"].rank(method="first")
        order_log = feat["amount_log"].rank(method="first")
        assert (order_amount == order_log).all()
