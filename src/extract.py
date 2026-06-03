"""
EXTRACT
=======
Generates a realistic synthetic credit-card transaction dataset that mimics
the structure of real financial crime data (similar to Kaggle's credit-card
fraud dataset used in academic literature).

Real-world equivalent: this step would be replaced by an API call, S3 pull,
or database query — the rest of the pipeline is identical.
"""

import numpy as np
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

# Reproducible seed
RNG_SEED = 42

# Merchant categories with rough spend profiles (mean, std in USD)
MERCHANT_CATEGORIES = {
    "grocery":       {"mean": 65,   "std": 35,   "fraud_rate": 0.005},
    "gas_station":   {"mean": 45,   "std": 20,   "fraud_rate": 0.008},
    "restaurant":    {"mean": 38,   "std": 22,   "fraud_rate": 0.006},
    "online_retail": {"mean": 120,  "std": 95,   "fraud_rate": 0.025},
    "electronics":   {"mean": 380,  "std": 280,  "fraud_rate": 0.045},
    "travel":        {"mean": 540,  "std": 420,  "fraud_rate": 0.035},
    "atm":           {"mean": 180,  "std": 120,  "fraud_rate": 0.018},
    "pharmacy":      {"mean": 42,   "std": 30,   "fraud_rate": 0.003},
}

US_CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
]


def _assign_fraud(row: pd.Series, rng: np.random.Generator) -> int:
    """
    Probabilistic fraud label that respects merchant-category base rates
    and applies uplift for known fraud signals.
    """
    base_rate = MERCHANT_CATEGORIES[row["merchant_category"]]["fraud_rate"]

    # Uplift signals
    if row["hour"] in range(0, 5):       # late-night transactions
        base_rate *= 3.5
    if row["amount"] > 800:              # unusually large amounts
        base_rate *= 2.5
    if row["is_foreign"]:                # foreign transaction on a domestic card
        base_rate *= 4.0
    if row["transaction_count_1h"] > 4:  # velocity spike
        base_rate *= 5.0

    base_rate = min(base_rate, 0.95)
    return int(rng.random() < base_rate)


def extract(n_records: int = 10_000) -> pd.DataFrame:
    """
    Generate n_records synthetic transactions.

    Returns a raw DataFrame mimicking data as it would arrive from a source
    system — including intentional quality issues (nulls, duplicates, outliers)
    for the Transform step to handle.
    """
    rng = np.random.default_rng(RNG_SEED)
    logger.debug("Generating %d synthetic transactions with seed=%d", n_records, RNG_SEED)

    categories = list(MERCHANT_CATEGORIES.keys())
    cat_weights = [0.22, 0.15, 0.18, 0.16, 0.08, 0.07, 0.08, 0.06]

    chosen_cats = rng.choice(categories, size=n_records, p=cat_weights)

    # Amounts drawn from category-specific distributions
    amounts = np.array([
        max(0.01, rng.normal(MERCHANT_CATEGORIES[c]["mean"],
                             MERCHANT_CATEGORIES[c]["std"]))
        for c in chosen_cats
    ])

    # Transaction timestamps spread over 90 days
    base_ts = pd.Timestamp("2024-01-01")
    seconds_offset = rng.integers(0, 90 * 86_400, size=n_records)
    timestamps = [base_ts + pd.Timedelta(seconds=int(s)) for s in seconds_offset]

    hours = [t.hour for t in timestamps]

    # Cardholder IDs (200 unique cards → repeat transactions per card)
    n_cards = 200
    card_ids = [f"CARD_{rng.integers(1000, 9999)}" for _ in range(n_cards)]
    card_assignment = rng.choice(card_ids, size=n_records)

    # Merchant IDs
    n_merchants = 80
    merchant_ids = [f"MERCH_{rng.integers(100, 999)}" for _ in range(n_merchants)]
    merchant_assignment = rng.choice(merchant_ids, size=n_records)

    # Cities
    city_assignment = rng.choice(US_CITIES, size=n_records)

    # Foreign flag (5% of transactions)
    is_foreign = rng.random(n_records) < 0.05

    # Velocity: count of transactions per card in prior hour
    # (simplified: random draw from Poisson, elevated for some cards)
    velocity = rng.poisson(lam=1.2, size=n_records)
    high_velocity_idx = rng.choice(n_records, size=int(n_records * 0.03), replace=False)
    velocity[high_velocity_idx] = rng.integers(5, 12, size=len(high_velocity_idx))

    df = pd.DataFrame({
        "transaction_id":       [f"TXN_{i:07d}" for i in range(n_records)],
        "timestamp":            timestamps,
        "card_id":              card_assignment,
        "merchant_id":          merchant_assignment,
        "merchant_category":    chosen_cats,
        "amount":               amounts.round(2),
        "city":                 city_assignment,
        "hour":                 hours,
        "is_foreign":           is_foreign,
        "transaction_count_1h": velocity,
    })

    # ── Inject intentional data quality issues ──────────────────
    # 1. ~2% null amounts (missing values from source system)
    null_amount_idx = rng.choice(n_records, size=int(n_records * 0.02), replace=False)
    df.loc[null_amount_idx, "amount"] = np.nan

    # 2. ~1% duplicate rows (double-posted transactions)
    dup_idx = rng.choice(n_records, size=int(n_records * 0.01), replace=False)
    duplicates = df.iloc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # 3. ~0.5% obviously invalid amounts (negative or > $10k — data entry errors)
    bad_idx = rng.choice(len(df), size=int(len(df) * 0.005), replace=False)
    df.loc[bad_idx, "amount"] = rng.choice([-99.99, -1.0, 15000.0, 25000.0],
                                            size=len(bad_idx))

    # Assign fraud labels after quality issues are injected
    df["is_fraud"] = df.apply(lambda row: _assign_fraud(row, rng), axis=1)

    logger.debug("Raw data shape: %s | Fraud rate: %.2f%%",
                 df.shape, df["is_fraud"].mean() * 100)
    return df
