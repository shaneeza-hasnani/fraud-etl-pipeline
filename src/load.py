"""
LOAD
====
Writes the transformed DataFrame to two destinations:

  1. SQLite database  — full enriched table for SQL querying
  2. CSV export       — flat file for dashboards, further analysis, or ML ingestion

Also computes and returns a run summary dict for the pipeline log.
"""

import os
import sqlite3
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

CSV_OUTPUT = "data/processed/transactions_clean.csv"
HIGH_RISK_CSV = "data/processed/high_risk_alerts.csv"


def _ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _write_sqlite(df: pd.DataFrame, db_path: str) -> None:
    """
    Write full dataset to SQLite.

    Two tables are created:
      transactions  — all cleaned & enriched records
      risk_summary  — aggregate counts by risk_tier and merchant_category
    """
    _ensure_dirs(db_path)
    conn = sqlite3.connect(db_path)

    df_write = df.copy()
    df_write["risk_tier"] = df_write["risk_tier"].astype(str)

    df_write.to_sql("transactions", conn, if_exists="replace", index=False)
    logger.debug("  SQLite: wrote %d rows to 'transactions' table", len(df_write))

    # Pre-built summary table
    summary = (
        df_write.groupby(["risk_tier", "merchant_category"])
        .agg(
            transaction_count=("transaction_id", "count"),
            avg_amount=("amount", "mean"),
            fraud_count=("is_fraud", "sum"),
        )
        .reset_index()
    )
    summary["avg_amount"] = summary["avg_amount"].round(2)
    summary.to_sql("risk_summary", conn, if_exists="replace", index=False)
    logger.debug("  SQLite: wrote risk_summary table (%d rows)", len(summary))

    conn.close()


def _write_csv(df: pd.DataFrame) -> None:
    """Write full cleaned dataset to CSV."""
    _ensure_dirs(CSV_OUTPUT)
    df_write = df.copy()
    df_write["risk_tier"] = df_write["risk_tier"].astype(str)
    df_write.to_csv(CSV_OUTPUT, index=False)
    logger.debug("  CSV: wrote %d rows to %s", len(df_write), CSV_OUTPUT)


def _write_high_risk_csv(df: pd.DataFrame) -> None:
    """Write HIGH-tier records to a separate alert file for analyst review."""
    _ensure_dirs(HIGH_RISK_CSV)
    high_risk = df[df["risk_tier"] == "HIGH"].copy()
    high_risk["risk_tier"] = high_risk["risk_tier"].astype(str)
    high_risk = high_risk.sort_values("risk_score", ascending=False)
    high_risk.to_csv(HIGH_RISK_CSV, index=False)
    logger.debug("  HIGH-RISK CSV: %d alerts written to %s",
                 len(high_risk), HIGH_RISK_CSV)


def load(df: pd.DataFrame, db_path: str = "data/processed/transactions.db") -> dict:
    """
    Load transformed data to all outputs and return a run summary.

    Args:
        df:      Transformed DataFrame from the transform step.
        db_path: Path to the SQLite database file.

    Returns:
        dict with load metadata for pipeline logging.
    """
    _write_sqlite(df, db_path)
    _write_csv(df)
    _write_high_risk_csv(df)

    tier_counts = df["risk_tier"].value_counts()
    high_risk_count = int(tier_counts.get("HIGH", 0))
    high_risk_pct   = high_risk_count / len(df) * 100

    return {
        "rows_loaded":       len(df),
        "db_path":           db_path,
        "csv_path":          CSV_OUTPUT,
        "high_risk_csv":     HIGH_RISK_CSV,
        "high_risk_count":   high_risk_count,
        "high_risk_pct":     round(high_risk_pct, 2),
        "tier_distribution": tier_counts.to_dict(),
    }
