"""
TRANSFORM
=========
Two-stage transformation:

  Stage 1 — Data Quality
    • Drop exact duplicates
    • Drop rows with null amounts
    • Remove transactions with invalid amounts (< $0.01 or > $9,999)

  Stage 2 — Feature Engineering
    • amount_zscore      : standardised spend per merchant category
    • is_high_amount     : binary flag for amounts above 95th percentile
    • is_night           : binary flag for transactions between midnight–5 AM
    • velocity_flag      : binary flag for ≥ 4 transactions on same card in 1 hr
    • risk_score         : composite 0–100 score aggregating fraud signals
    • risk_tier          : categorical bucket (LOW / MEDIUM / HIGH)
"""

import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)


# ── Stage 1: Data Quality ─────────────────────────────────────────────────────

def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    logger.debug("  drop_duplicates: removed %d rows", dropped)
    return df


def _drop_null_amounts(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["amount"])
    dropped = before - len(df)
    logger.debug("  drop_null_amounts: removed %d rows", dropped)
    return df


def _remove_invalid_amounts(df: pd.DataFrame,
                             min_amount: float = 0.01,
                             max_amount: float = 9_999.99) -> pd.DataFrame:
    before = len(df)
    df = df[(df["amount"] >= min_amount) & (df["amount"] <= max_amount)]
    dropped = before - len(df)
    logger.debug("  remove_invalid_amounts: removed %d rows (range: $%.2f–$%.2f)",
                 dropped, min_amount, max_amount)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all data quality rules in sequence."""
    logger.debug("Running Stage 1 — Data Quality (%d rows in)", len(df))
    df = _drop_duplicates(df)
    df = _drop_null_amounts(df)
    df = _remove_invalid_amounts(df)
    logger.debug("Stage 1 complete — %d rows out", len(df))
    return df.reset_index(drop=True)


# ── Stage 2: Feature Engineering ─────────────────────────────────────────────

def _add_amount_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise amount within each merchant category.
    High z-scores flag unusual spend for that category's typical range.
    """
    stats = df.groupby("merchant_category")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "cat_mean", "std": "cat_std"}
    )
    df = df.merge(stats, on="merchant_category", how="left")
    df["amount_zscore"] = ((df["amount"] - df["cat_mean"]) / df["cat_std"]).round(4)
    df = df.drop(columns=["cat_mean", "cat_std"])
    return df


def _add_high_amount_flag(df: pd.DataFrame, percentile: float = 95) -> pd.DataFrame:
    """Flag amounts in the top percentile — unusually large transactions."""
    threshold = df["amount"].quantile(percentile / 100)
    df["is_high_amount"] = (df["amount"] >= threshold).astype(int)
    logger.debug("  high_amount threshold (p%d): $%.2f", percentile, threshold)
    return df


def _add_night_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag transactions between midnight and 5 AM (inclusive)."""
    df["is_night"] = df["hour"].between(0, 4).astype(int)
    return df


def _add_velocity_flag(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """Flag cards with ≥ threshold transactions in the prior hour."""
    df["velocity_flag"] = (df["transaction_count_1h"] >= threshold).astype(int)
    return df


def _add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite risk score (0–100) based on weighted fraud signals.

    Weights reflect empirical importance of each signal:
      - velocity_flag:    high-frequency abuse is the strongest indicator
      - is_foreign:       card-not-present / cross-border uplift
      - is_high_amount:   unusually large spend
      - amount_zscore:    spend anomalous for that merchant category
      - is_night:         off-hours transactions
    """
    score = (
          df["velocity_flag"]   * 35
        + df["is_foreign"]      * 25
        + df["is_high_amount"]  * 20
        + df["amount_zscore"].clip(0, 3) / 3 * 15
        + df["is_night"]        * 5
    )
    # Normalise to 0–100
    df["risk_score"] = score.clip(0, 100).round(1)
    return df


def _add_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket risk_score into human-readable tiers for analyst review."""
    df["risk_tier"] = pd.cut(
        df["risk_score"],
        bins=[-0.01, 20, 55, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    logger.debug("Running Stage 2 — Feature Engineering (%d rows in)", len(df))
    df = _add_amount_zscore(df)
    df = _add_high_amount_flag(df)
    df = _add_night_flag(df)
    df = _add_velocity_flag(df)
    df = _add_risk_score(df)
    df = _add_risk_tier(df)
    logger.debug("Stage 2 complete — %d columns out", len(df.columns))
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full transform step: data quality → feature engineering.
    Returns a clean, enriched DataFrame ready for loading.
    """
    df = clean(raw_df)
    df = engineer_features(df)

    # Final column ordering: IDs → transaction info → engineered features → label
    col_order = [
        "transaction_id", "timestamp", "card_id", "merchant_id",
        "merchant_category", "amount", "city", "hour",
        "is_foreign", "transaction_count_1h",
        # engineered
        "amount_zscore", "is_high_amount", "is_night",
        "velocity_flag", "risk_score", "risk_tier",
        # label (kept for evaluation / model training downstream)
        "is_fraud",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    return df
