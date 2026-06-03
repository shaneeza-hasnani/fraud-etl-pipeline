# Fraud Transaction ETL Pipeline

A clean, modular Python ETL pipeline that extracts synthetic credit-card transaction data, applies data quality rules and fraud-signal feature engineering, and loads the enriched output to SQLite and CSV.

Built to demonstrate production ETL practices in a financial crime context — the domain where this kind of pipeline actually gets used.

---

## What It Does

```
EXTRACT → TRANSFORM → LOAD
```

| Stage | What happens |
|---|---|
| **Extract** | Generates 10,000+ synthetic transactions with realistic merchant distributions, cardholder velocity patterns, and intentional data quality issues (nulls, duplicates, invalid amounts) |
| **Transform** | Stage 1 cleans the raw data; Stage 2 engineers five fraud-signal features and produces a composite risk score with tier bucketing |
| **Load** | Writes to SQLite (two tables) + two CSV exports: full cleaned dataset and a high-risk alerts file |

---

## Feature Engineering

The transform step produces the following features on top of the raw transaction fields:

| Feature | Description |
|---|---|
| `amount_zscore` | Spend standardised within merchant category — flags unusual amounts relative to that category's typical range |
| `is_high_amount` | Binary flag for transactions in the 95th percentile of spend |
| `is_night` | Binary flag for transactions between midnight–5 AM |
| `velocity_flag` | Binary flag for ≥ 4 transactions on the same card in the prior hour |
| `risk_score` | Weighted composite score (0–100) aggregating all signals |
| `risk_tier` | Categorical bucket: LOW / MEDIUM / HIGH |

**Risk score weights** reflect the empirical importance of each signal in financial crime detection:

```
velocity_flag   × 35   ← high-frequency abuse is the strongest indicator
is_foreign      × 25   ← card-not-present / cross-border uplift  
is_high_amount  × 20   ← unusually large spend
amount_zscore   × 15   ← spend anomalous for that merchant category
is_night        ×  5   ← off-hours signal
```

---

## Project Structure

```
fraud-etl-pipeline/
├── pipeline.py          # Orchestrator — runs the full ETL
├── src/
│   ├── extract.py       # Data generation / extraction layer
│   ├── transform.py     # Cleaning + feature engineering
│   ├── load.py          # SQLite + CSV output
│   └── logger.py        # Centralised logging (console + file)
├── tests/
│   └── test_pipeline.py # 27 unit + integration tests (pytest)
├── data/
│   ├── raw/             # Would hold source files in a real pipeline
│   └── processed/
│       ├── transactions.db          # SQLite: transactions + risk_summary tables
│       ├── transactions_clean.csv   # Full enriched dataset
│       └── high_risk_alerts.csv     # HIGH-tier transactions for analyst review
├── logs/                # Timestamped run logs
└── requirements.txt
```

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/shaneeza-hasnani/fraud-etl-pipeline
cd fraud-etl-pipeline
pip install -r requirements.txt

# Run the pipeline (default: 10,000 transactions)
python pipeline.py

# Custom record count and output path
python pipeline.py --records 50000 --db data/processed/transactions.db

# Run tests
python -m pytest tests/ -v
```

**Sample output:**
```
2024-01-15 09:12:03  INFO  __main__ — ============================================================
2024-01-15 09:12:03  INFO  __main__ — FRAUD ETL PIPELINE — starting run
2024-01-15 09:12:03  INFO  __main__ — ============================================================
2024-01-15 09:12:03  INFO  __main__ — [1/3] EXTRACT — generating 10000 raw transactions
2024-01-15 09:12:03  INFO  __main__ —       Extracted 10100 rows, 11 columns
2024-01-15 09:12:03  INFO  __main__ — [2/3] TRANSFORM — cleaning & engineering features
2024-01-15 09:12:03  INFO  __main__ —       Transformed to 9753 rows, 17 columns
2024-01-15 09:12:03  INFO  __main__ — [3/3] LOAD — writing to SQLite + CSV
2024-01-15 09:12:03  INFO  __main__ —       Rows loaded:    9753
2024-01-15 09:12:04  INFO  __main__ — Pipeline complete. High-risk transactions: 51 (0.5%)
```

---

## Querying the Output

The SQLite database includes a pre-built `risk_summary` table:

```sql
-- Connect
sqlite3 data/processed/transactions.db

-- Fraud rate by risk tier
SELECT risk_tier,
       SUM(fraud_count) AS fraud_txns,
       SUM(transaction_count) AS total_txns,
       ROUND(100.0 * SUM(fraud_count) / SUM(transaction_count), 2) AS fraud_rate_pct
FROM risk_summary
GROUP BY risk_tier
ORDER BY fraud_rate_pct DESC;

-- Top merchant categories by high-risk volume
SELECT merchant_category,
       SUM(transaction_count) AS txns,
       ROUND(AVG(avg_amount), 2) AS avg_spend
FROM risk_summary
WHERE risk_tier = 'HIGH'
GROUP BY merchant_category
ORDER BY txns DESC;
```

---

## Design Decisions

**Why SQLite?**
Keeps the project self-contained and runnable without infrastructure. The `load.py` module's `_write_sqlite()` function maps directly to any SQLAlchemy-compatible database (Postgres, Snowflake) — swap the connection string and nothing else changes.

**Why synthetic data?**
Real transaction data carries PII and can't be committed to a public repo. The synthetic generator (`extract.py`) reproduces realistic statistical properties: category-specific spend distributions, cardholder velocity patterns, and a fraud rate (~3–8%) driven by the same signals a real fraud model would use.

**Why not use a workflow tool (Airflow, Prefect)?**
Intentional simplicity. The three-module structure (`extract` → `transform` → `load`) maps directly onto how these tools decompose pipelines. Wrapping `pipeline.py` in an Airflow DAG would take under 20 lines and is the obvious next step.

---

## Next Steps

This pipeline produces a risk-scored dataset ready for:

- **ML model training** — `is_fraud` is the label; the five engineered features are strong predictors
- **Dashboard integration** — `high_risk_alerts.csv` is formatted for direct Power BI / Tableau ingestion
- **Scheduling** — orchestrate with Airflow, Prefect, or a simple cron job
- **Real data sources** — replace `extract.py` with an API call, S3 pull, or database query; the rest is unchanged

---

## Stack

Python 3.11+ · pandas · NumPy · SQLite · pytest

---

*Shaneeza Hasnani — CFE · Data Scientist · [linkedin.com/in/shasnani](https://linkedin.com/in/shasnani)*
