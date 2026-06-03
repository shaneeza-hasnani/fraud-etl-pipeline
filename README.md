# Fraud Transaction ETL Pipeline

**Python · pandas · NumPy · SQLite · pytest**

---

I built this to show the full data engineering side of fraud analytics - not just the model, but the pipeline that makes the model possible.

Most fraud detection projects start with a clean dataset. This one builds the infrastructure that produces it: extract raw transaction data, handle the mess that comes with it, engineer the features that actually matter for fraud detection, and load the result somewhere useful.

---

## What it does

```
EXTRACT → TRANSFORM → LOAD
```

**Extract** - generates 10,000 synthetic credit card transactions with realistic merchant distributions, cardholder velocity patterns, and intentional data quality issues (nulls, duplicates, invalid amounts). In production this step would be replaced by an API call or database query - everything downstream stays the same.

**Transform** - two stages:
- *Clean:* drops duplicates, null amounts, and transactions with invalid values
- *Feature engineering:* adds five fraud-signal columns and a composite risk score

**Load** - writes to SQLite (queryable with SQL) and two CSVs: the full enriched dataset and a separate high-risk alerts file for analyst review.

---

## Features engineered

| Feature | What it captures |
|---|---|
| `amount_zscore` | Spend standardised within merchant category - flags unusual amounts relative to what's normal for that type of merchant |
| `is_high_amount` | Transactions in the top 5% of spend |
| `is_night` | Transactions between midnight and 5 AM |
| `velocity_flag` | Cards with 4+ transactions in the prior hour |
| `risk_score` | Weighted composite (0–100) combining all signals |
| `risk_tier` | LOW / MEDIUM / HIGH bucket for analyst triage |

The risk score weights reflect actual fraud signal importance - velocity is weighted highest because automated abuse is the strongest indicator; time-of-day is weighted lowest because it's correlated but not deterministic.

---

## Running it

```bash
git clone https://github.com/shaneeza-hasnani/fraud-etl-pipeline
cd fraud-etl-pipeline
pip install -r requirements.txt

# Run the pipeline
python pipeline.py

# Run tests
python -m pytest tests/ -v
```

Output:
```
[1/3] EXTRACT - generating 10000 raw transactions
      Extracted 10100 rows, 11 columns
[2/3] TRANSFORM - cleaning & engineering features
      Transformed to 9753 rows, 17 columns
[3/3] LOAD - writing to SQLite + CSV
      Rows loaded: 9753
Pipeline complete. High-risk transactions: 51 (0.5%)
```

---

## Project structure

```
fraud-etl-pipeline/
├── pipeline.py          # Orchestrator
├── src/
│   ├── extract.py       # Data generation / extraction
│   ├── transform.py     # Cleaning + feature engineering
│   ├── load.py          # SQLite + CSV output
│   └── logger.py        # Timestamped logging
├── tests/
│   └── test_pipeline.py # 27 unit and integration tests
└── data/processed/
    ├── transactions.db          # SQLite output
    ├── transactions_clean.csv   # Full enriched dataset
    └── high_risk_alerts.csv     # HIGH-tier records only
```

---

## Why I built it this way

Keeping the three stages as separate modules means swapping any one of them out doesn't touch the others. Replace `extract.py` with a real API call, change `load.py` to write to Postgres or Snowflake - the transform logic stays exactly the same. That's the design decision that matters most in production pipelines.

The synthetic data generator injects real data quality problems on purpose: nulls from missing source fields, duplicates from double-posted transactions, out-of-range values from upstream data entry errors. Handling these isn't a footnote - it's most of the actual work.

---

## What's next

The output of this pipeline feeds directly into a fraud classification model. `is_fraud` is the label; the five engineered features are strong predictors. That's the natural next step.

---

*Shaneeza Hasnani - CFE · MS Business Analytics & AI · [linkedin.com/in/shasnani](https://linkedin.com/in/shasnani)*
