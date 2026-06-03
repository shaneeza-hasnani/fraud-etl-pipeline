"""
Fraud Transaction ETL Pipeline
===============================
Extract → Transform → Load pipeline for synthetic fraud transaction data.

Author: Shaneeza Hasnani
"""

import argparse
import sys
from src.extract import extract
from src.transform import transform
from src.load import load
from src.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(n_records: int = 10_000, db_path: str = "data/processed/transactions.db"):
    """
    Orchestrate the full ETL pipeline.

    Args:
        n_records: Number of synthetic transactions to generate.
        db_path:   Path to the SQLite output database.
    """
    logger.info("=" * 60)
    logger.info("FRAUD ETL PIPELINE — starting run")
    logger.info("=" * 60)

    # ── EXTRACT ──────────────────────────────────────────────────
    logger.info("[1/3] EXTRACT — generating %s raw transactions", n_records)
    raw_df = extract(n_records=n_records)
    logger.info("      Extracted %d rows, %d columns", *raw_df.shape)

    # ── TRANSFORM ────────────────────────────────────────────────
    logger.info("[2/3] TRANSFORM — cleaning & engineering features")
    clean_df = transform(raw_df)
    logger.info("      Transformed to %d rows, %d columns", *clean_df.shape)

    # ── LOAD ─────────────────────────────────────────────────────
    logger.info("[3/3] LOAD — writing to SQLite + CSV")
    summary = load(clean_df, db_path=db_path)
    logger.info("      Rows loaded:    %d", summary["rows_loaded"])
    logger.info("      CSV written to: %s", summary["csv_path"])
    logger.info("      DB written to:  %s", summary["db_path"])

    logger.info("=" * 60)
    logger.info("Pipeline complete. High-risk transactions: %d (%.1f%%)",
                summary["high_risk_count"],
                summary["high_risk_pct"])
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Transaction ETL Pipeline")
    parser.add_argument("--records", type=int, default=10_000,
                        help="Number of synthetic transactions (default: 10000)")
    parser.add_argument("--db", type=str, default="data/processed/transactions.db",
                        help="Output SQLite path")
    args = parser.parse_args()

    try:
        run_pipeline(n_records=args.records, db_path=args.db)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)
