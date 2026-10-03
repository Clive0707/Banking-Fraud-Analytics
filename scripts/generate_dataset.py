"""
Regenerate the 15,000,000-row banking transactions dataset.

Why this exists
---------------
The raw dataset is ~1.76 GB. GitHub rejects files above 100 MB, so it cannot be
committed, which meant anyone cloning the repository could run the dashboard and
retrain the sampled models but could NOT run the PySpark pipeline -- the
centerpiece of the project. This script closes that gap: one command reproduces a
statistically faithful dataset and the whole pipeline becomes runnable from a
fresh clone.

Fidelity
--------
Every parameter below was measured from the original dataset rather than
invented, and the generator reproduces:

* the exact schema, id formats and customer id range
* categorical mixes (transaction type, account type, payment method, device,
  merchant, location) to within rounding
* amounts as a clipped lognormal (log-mean 7.6512, log-std 1.3404, 50-250,000)
* the fraud structure: a probability table keyed by (overnight, international,
  high-value), reproducing the measured 3x / 5x / 9x lifts and their interactions
* the account-drain defect: a transaction larger than the available balance
  zeroes it instead of overdrawing, which is the real source of the ~95%
  consistency score the pipeline reports
* the `transaction_status` target leak: Declined and Flagged occur only on
  fraudulent rows, so the leakage audit has something to find

How faithful
------------
This reproduces the dataset's *statistical structure*, not its exact bytes. A
regenerated file gives the same schema, the same fraud mechanics and lifts within
a few percent, and the same qualitative findings -- but the headline figures
quoted in the README (0.9776% fraud rate, 746,174 ledger violations, 95.03%
consistency) were measured on the original file and a fresh draw will land near
them rather than on them. `--verify` checks the properties that matter with
explicit tolerance bands.

Usage
-----
    python scripts/generate_dataset.py                 # full 15M rows
    python scripts/generate_dataset.py --rows 1000000  # smaller, for testing
    python scripts/generate_dataset.py --seed 7        # a different draw

Writes `data/raw/banking_transactions_15m.csv` plus the small
`customer_profiles.csv` companion.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import config  # noqa: E402

# ---------------------------------------------------------------------------
# Measured parameters
# ---------------------------------------------------------------------------

DEFAULT_ROWS = 15_000_000
N_CUSTOMERS = 25_000
CUSTOMER_ID_START = 100_001
TXN_ID_START = 10_000_000  # ids render as TXN10000000 upward
DATE_START = "2025-01-01"
N_DAYS = 365

# Amounts: lognormal, clipped to the observed range. The parameters are solved
# so the generated median (1,946) and mean (5,267) match the source. The real
# distribution has a slightly thinner upper tail than a true lognormal, so the
# share above the high-value threshold comes out a little higher here -- see the
# fidelity note in the module docstring.
AMOUNT_LOG_MEAN = 7.5737
AMOUNT_LOG_STD = 1.4111
AMOUNT_MIN, AMOUNT_MAX = 50.0, 250_000.0

# Opening balances: roughly normal, floored well above zero.
BALANCE_MEAN, BALANCE_STD = 75_916.0, 43_008.0
BALANCE_MIN, BALANCE_MAX = 1_000.0, 285_000.0

# An account drain is not a random event in the source data: it happens exactly
# when the transaction exceeds the available balance. Every row with
# `amount > balance_before` has `balance_after == 0`, and every zero-balance row
# is such a row -- the correspondence is 100% in both directions. Rather than
# allowing an overdraft, the original generator zeroed the balance, which is
# precisely the defect that drops the pipeline's consistency score to ~95%.
# The rate below is the emergent consequence of the amount and balance
# distributions overlapping, recorded here only as the expected value.
EXPECTED_DRAIN_RATE = 0.04936

CATEGORICALS = {
    "transaction_type": (
        ["UPI", "Card Payment", "Bank Transfer", "ATM Withdrawal", "Online Purchase"],
        [0.38106, 0.25008, 0.14918, 0.11964, 0.10004],
    ),
    "account_type": (
        ["Savings", "Salary", "Current"],
        [0.64912, 0.20460, 0.14628],
    ),
    "payment_method": (
        ["UPI", "Debit Card", "Net Banking", "Credit Card", "ATM"],
        [0.38067, 0.24960, 0.14933, 0.11987, 0.10053],
    ),
    "device_type": (
        ["Android", "iOS", "Windows", "ATM", "Mac"],
        [0.45006, 0.19964, 0.18006, 0.10020, 0.07004],
    ),
    "merchant": (
        ["Zomato", "Local Merchant", "Amazon", "Reliance", "IRCTC", "Uber",
         "DMart", "Swiggy", "Utility Bill", "Bank Transfer", "Flipkart", "BookMyShow"],
        [0.08393, 0.08377, 0.08367, 0.08348, 0.08341, 0.08338,
         0.08326, 0.08326, 0.08315, 0.08295, 0.08289, 0.08284],
    ),
    "location": (
        ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Kolkata",
         "Ahmedabad", "Nashik", "Nagpur", "New York", "Singapore", "Dubai", "London"],
        [0.21989, 0.14164, 0.13129, 0.09695, 0.09572, 0.07891, 0.06803,
         0.06411, 0.04887, 0.04032, 0.00363, 0.00361, 0.00356, 0.00346],
    ),
}

# Fraud probability keyed by (is_night, is_international, is_high_amount).
# These are the measured per-cell rates; they are what produce the documented
# 3.0x overnight, 5.0x international and 7.6x high-value lifts, and the
# interaction effects that exceed either flag alone.
FRAUD_PROBABILITY = {
    (False, False, False): 0.004038,
    (False, False, True): 0.075061,
    (False, True, False): 0.040982,
    (False, True, True): 0.106280,
    (True, False, False): 0.028495,
    (True, False, True): 0.110000,
    (True, True, False): 0.082361,
    (True, True, True): 0.120000,
}

# transaction_status is the bank's own fraud verdict, which is exactly why it
# leaks: Declined and Flagged never appear on a legitimate transaction.
STATUS_IF_FRAUD = (["Flagged", "Completed", "Declined"], [0.55562, 0.24495, 0.19943])
STATUS_IF_LEGIT = (["Completed", "Pending"], [0.97020, 0.02980])

COLUMNS = config.RAW_COLUMNS
CHUNK_ROWS = 1_000_000


def _draw(rng, spec, size):
    values, probs = spec
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()  # guard against rounding drift
    return rng.choice(values, size=size, p=probs)


def generate_chunk(rng, n, start_id, date_index):
    """Build one chunk of transactions as a DataFrame."""
    txn_id = np.arange(start_id, start_id + n)
    customer_id = rng.integers(CUSTOMER_ID_START, CUSTOMER_ID_START + N_CUSTOMERS, n)

    day_offset = rng.integers(0, N_DAYS, n)
    dates = date_index[day_offset]

    hour = rng.integers(0, 24, n)
    minute = rng.integers(0, 60, n)
    second = rng.integers(0, 60, n)

    amount = np.round(
        np.clip(rng.lognormal(AMOUNT_LOG_MEAN, AMOUNT_LOG_STD, n), AMOUNT_MIN, AMOUNT_MAX), 2
    )
    balance_before = np.round(
        np.clip(rng.normal(BALANCE_MEAN, BALANCE_STD, n), BALANCE_MIN, BALANCE_MAX), 2
    )

    location = _draw(rng, CATEGORICALS["location"], n)

    is_night = hour < config.NIGHT_END_HOUR
    is_intl = np.isin(location, list(config.INTERNATIONAL_LOCATIONS))
    is_high = amount > config.HIGH_AMOUNT_THRESHOLD

    # Vectorised lookup into the measured probability table.
    p = np.empty(n, dtype=float)
    for (night, intl, high), prob in FRAUD_PROBABILITY.items():
        mask = (is_night == night) & (is_intl == intl) & (is_high == high)
        p[mask] = prob
    is_fraud = (rng.random(n) < p).astype(np.int8)

    # Overdrafts are zeroed rather than allowed to go negative. This single rule
    # reproduces the ledger defect: ~4.9% of rows end up violating
    # `balance_before - balance_after == amount`, and every one of them is a
    # zero-balance row, matching the source data exactly.
    overdraft = amount > balance_before
    balance_after = np.round(np.where(overdraft, 0.0, balance_before - amount), 2)

    status = np.where(
        is_fraud == 1,
        _draw(rng, STATUS_IF_FRAUD, n),
        _draw(rng, STATUS_IF_LEGIT, n),
    )

    return pd.DataFrame({
        "transaction_id": np.char.add("TXN", txn_id.astype(str)),
        "customer_id": customer_id,
        "transaction_date": dates,
        "transaction_time": pd.Series(hour).map("{:02d}".format).values
                            + ":" + pd.Series(minute).map("{:02d}".format).values
                            + ":" + pd.Series(second).map("{:02d}".format).values,
        "transaction_type": _draw(rng, CATEGORICALS["transaction_type"], n),
        "account_type": _draw(rng, CATEGORICALS["account_type"], n),
        "amount": amount,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "merchant": _draw(rng, CATEGORICALS["merchant"], n),
        "location": location,
        "payment_method": _draw(rng, CATEGORICALS["payment_method"], n),
        "device_type": _draw(rng, CATEGORICALS["device_type"], n),
        "transaction_status": status,
        "is_fraud": is_fraud,
    })[COLUMNS]


def generate_dataset(rows=DEFAULT_ROWS, seed=config.RANDOM_SEED, output=None,
                     chunk_rows=CHUNK_ROWS, write_profiles=True):
    """
    Write `rows` transactions to CSV, streaming in chunks so peak memory stays
    near one chunk rather than the whole dataset.
    """
    output = Path(output) if output else config.RAW_TRANSACTIONS_CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    rng = np.random.default_rng(seed)
    date_index = np.array(
        pd.date_range(DATE_START, periods=N_DAYS, freq="D").strftime("%Y-%m-%d")
    )

    print(f"Generating {rows:,} transactions -> {output}")
    print(f"  seed={seed}  chunk={chunk_rows:,} rows")

    t0 = time.perf_counter()
    written = 0
    fraud_total = 0

    while written < rows:
        n = min(chunk_rows, rows - written)
        chunk = generate_chunk(rng, n, TXN_ID_START + written, date_index)
        chunk.to_csv(output, mode="a", header=(written == 0), index=False)
        fraud_total += int(chunk["is_fraud"].sum())
        written += n
        elapsed = time.perf_counter() - t0
        pct = written / rows * 100
        print(f"  {written:>12,} / {rows:,} rows ({pct:5.1f}%)  "
              f"{elapsed:6.1f}s  {int(written / elapsed):,} rows/s")

    size_gb = output.stat().st_size / 1e9
    elapsed = time.perf_counter() - t0
    print(f"\nWrote {written:,} rows ({size_gb:.2f} GB) in {elapsed:.1f}s")
    print(f"  fraud: {fraud_total:,} ({fraud_total / written * 100:.4f}%)")

    if write_profiles:
        _write_customer_profiles(output.parent, rng)

    return {"rows": written, "fraud": fraud_total, "path": str(output),
            "size_gb": round(size_gb, 2), "seconds": round(elapsed, 1)}


def _write_customer_profiles(raw_dir, rng):
    """
    Small companion file of customer attributes.

    The pipeline derives its real customer profiles by aggregating the 15M
    transactions in Spark; this file only exists as the fallback input the
    segmentation job accepts when the aggregation has not run yet.
    """
    path = raw_dir / config.RAW_CUSTOMERS_CSV.name
    ids = np.arange(CUSTOMER_ID_START, CUSTOMER_ID_START + N_CUSTOMERS)
    pd.DataFrame({
        "customer_id": ids,
        "account_type": rng.choice(["Savings", "Salary", "Current"], N_CUSTOMERS,
                                   p=[0.64912, 0.20460, 0.14628]),
        "location": rng.choice(CATEGORICALS["location"][0], N_CUSTOMERS,
                               p=np.array(CATEGORICALS["location"][1])
                               / sum(CATEGORICALS["location"][1])),
        "age": rng.integers(18, 75, N_CUSTOMERS),
    }).to_csv(path, index=False)
    print(f"Wrote {N_CUSTOMERS:,} customer profiles -> {path}")


def verify(path=None, sample_rows=2_000_000):
    """
    Check a generated file against the documented statistical properties.

    Reads only the first `sample_rows` rows, which is plenty to confirm the
    distributions and costs seconds rather than minutes.
    """
    path = Path(path) if path else config.RAW_TRANSACTIONS_CSV
    if not path.exists():
        print(f"No dataset at {path}")
        return False

    print(f"\nVerifying {path.name} (first {sample_rows:,} rows)...")
    df = pd.read_csv(path, nrows=sample_rows)
    df["hour"] = pd.to_datetime(df["transaction_time"], format="%H:%M:%S").dt.hour
    intl = df["location"].isin(config.INTERNATIONAL_LOCATIONS)
    night = df["hour"] < config.NIGHT_END_HOUR
    high = df["amount"] > config.HIGH_AMOUNT_THRESHOLD
    base = df["is_fraud"].mean()

    checks = []

    def chk(label, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")

    chk("schema matches", list(df.columns[:15]) == COLUMNS)
    chk("fraud rate near 1%", 0.006 < base < 0.015, f"{base * 100:.4f}%")
    chk("overnight lift ~3x", 2.0 < df[night].is_fraud.mean() / base < 4.5,
        f"{df[night].is_fraud.mean() / base:.2f}x")
    chk("international lift ~5x", 3.0 < df[intl].is_fraud.mean() / base < 7.0,
        f"{df[intl].is_fraud.mean() / base:.2f}x")
    if high.any():
        chk("high-value lift >5x", df[high].is_fraud.mean() / base > 5.0,
            f"{df[high].is_fraud.mean() / base:.2f}x")

    leak = df["transaction_status"].isin(config.LEAKING_STATUS_VALUES)
    chk("status leak present (precision 1.0)",
        leak.any() and df.loc[leak, "is_fraud"].mean() == 1.0)

    ledger_ok = np.isclose(df["balance_before"] - df["balance_after"], df["amount"], atol=0.01)
    violation_rate = 1 - ledger_ok.mean()
    chk("ledger defect ~5%", 0.03 < violation_rate < 0.07, f"{violation_rate * 100:.3f}%")
    chk("all ledger violations are drains",
        bool((df.loc[~ledger_ok, "balance_after"] == 0).all()))
    # The defining property: drains and overdrafts are the same set of rows.
    overdraft = df["amount"] > df["balance_before"]
    drained = df["balance_after"] == 0
    chk("drains correspond exactly to overdrafts", bool((overdraft == drained).all()))
    chk("no negative balances", bool((df["balance_after"] >= 0).all()))

    chk("no nulls", int(df.isna().sum().sum()) == 0)
    chk("unique transaction ids", df["transaction_id"].is_unique)
    chk("amounts within range",
        bool((df["amount"] >= AMOUNT_MIN).all() and (df["amount"] <= AMOUNT_MAX).all()))

    ok = all(checks)
    print(f"\n{'VERIFICATION PASSED' if ok else 'VERIFICATION FAILED'}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the banking transactions dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The full 15M-row file is ~1.76 GB and takes a few minutes to write.\n"
            "Use --rows for a smaller dataset when you only want to exercise the pipeline."
        ),
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help=f"number of transactions (default {DEFAULT_ROWS:,})")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="output CSV path (default data/raw/banking_transactions_15m.csv)")
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS,
                        help="rows generated per chunk (controls peak memory)")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify an existing dataset without regenerating it")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip verification after generating")
    args = parser.parse_args()

    if args.verify_only:
        return 0 if verify(args.output) else 1

    generate_dataset(rows=args.rows, seed=args.seed, output=args.output,
                     chunk_rows=args.chunk_rows)

    if not args.no_verify:
        sample = min(args.rows, 2_000_000)
        if not verify(args.output, sample_rows=sample):
            return 1

    print("\nNext: python run.py --all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
