"""Shared pytest fixtures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))


@pytest.fixture(scope="session")
def synthetic_transactions():
    """
    A small transaction frame with the same schema and signal structure as the
    real dataset: an overnight band, international locations, a high-value tail,
    account drains, and a transaction_status column that leaks the target.

    Built in-process so the tests never depend on the 1.85 GB raw file.
    """
    rng = np.random.RandomState(0)
    # Large enough that each leaking status level clears the purity scan's
    # minimum-support bar, so the leakage tests exercise the real code path.
    n = 20000

    locations = ["Mumbai", "Delhi", "Bengaluru", "Dubai", "Singapore", "London"]
    loc = rng.choice(locations, n, p=[0.3, 0.25, 0.25, 0.08, 0.07, 0.05])
    hour = rng.randint(0, 24, n)
    amount = np.round(rng.lognormal(7.5, 1.2, n), 2)

    is_intl = np.isin(loc, ["Dubai", "Singapore", "London"])
    is_night = hour < 5
    is_high = amount > 100_000

    # Fraud probability rises with each risk factor, mirroring the measured lifts.
    p = 0.005 + 0.02 * is_night + 0.03 * is_intl + 0.05 * is_high
    is_fraud = (rng.random_sample(n) < p).astype(int)

    balance_before = np.round(rng.uniform(1000, 200000, n), 2)
    drained = rng.random_sample(n) < 0.05
    balance_after = np.where(drained, 0.0, np.maximum(0, balance_before - amount))

    # Leakage, deliberately: Declined/Flagged occur only on fraud.
    status = np.where(
        is_fraud == 1,
        rng.choice(["Flagged", "Declined", "Completed"], n, p=[0.6, 0.2, 0.2]),
        rng.choice(["Completed", "Pending"], n, p=[0.97, 0.03]),
    )

    return pd.DataFrame({
        "transaction_id": [f"TXN{i:08d}" for i in range(n)],
        "customer_id": rng.randint(100000, 100200, n),
        "transaction_date": pd.to_datetime(
            rng.randint(0, 365, n), unit="D", origin="2025-01-01"
        ).strftime("%Y-%m-%d"),
        "transaction_time": [f"{h:02d}:{rng.randint(0, 60):02d}:00" for h in hour],
        "transaction_type": rng.choice(["UPI", "Card Payment", "ATM Withdrawal"], n),
        "account_type": rng.choice(["Savings", "Current", "Salary"], n),
        "amount": amount,
        "balance_before": balance_before,
        "balance_after": np.round(balance_after, 2),
        "merchant": rng.choice(["Amazon", "Flipkart", "Swiggy", "IRCTC"], n),
        "location": loc,
        "payment_method": rng.choice(["UPI", "Credit Card", "Debit Card"], n),
        "device_type": rng.choice(["Android", "iOS", "Windows"], n),
        "transaction_status": status,
        "is_fraud": is_fraud,
    })


@pytest.fixture(scope="session")
def customer_stats(synthetic_transactions):
    from src.fraud_detection.train_models import fit_customer_baselines
    return fit_customer_baselines(synthetic_transactions)
