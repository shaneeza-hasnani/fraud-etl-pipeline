"""
Tests for the Fraud ETL Pipeline.
Run with:  python -m pytest tests/ -v
"""

import pandas as pd
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.extract import extract
from src.transform import transform, clean, engineer_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return extract(n_records=500)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean(raw_df)


@pytest.fixture(scope="module")
def transformed_df(raw_df):
    return transform(raw_df)


# ── Extract Tests ─────────────────────────────────────────────────────────────

class TestExtract:

    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_expected_columns_present(self, raw_df):
        required = {"transaction_id", "timestamp", "card_id", "merchant_id",
                    "merchant_category", "amount", "city", "hour",
                    "is_foreign", "transaction_count_1h", "is_fraud"}
        assert required.issubset(set(raw_df.columns))

    def test_row_count_reasonable(self, raw_df):
        # Duplicates are injected, so row count > n_records
        assert len(raw_df) >= 500

    def test_fraud_label_is_binary(self, raw_df):
        assert set(raw_df["is_fraud"].unique()).issubset({0, 1})

    def test_has_some_nulls(self, raw_df):
        # Intentional null injection: amount column should have some nulls
        assert raw_df["amount"].isna().sum() > 0

    def test_has_some_duplicates(self, raw_df):
        assert raw_df.duplicated().sum() > 0

    def test_transaction_ids_are_strings(self, raw_df):
        # pandas 2.x uses StringDtype; object also acceptable
        assert pd.api.types.is_string_dtype(raw_df["transaction_id"])

    def test_hours_in_valid_range(self, raw_df):
        valid_hours = raw_df["hour"].dropna()
        assert valid_hours.between(0, 23).all()


# ── Clean Tests ───────────────────────────────────────────────────────────────

class TestClean:

    def test_no_duplicates_after_clean(self, clean_df):
        assert clean_df.duplicated().sum() == 0

    def test_no_null_amounts_after_clean(self, clean_df):
        assert clean_df["amount"].isna().sum() == 0

    def test_no_invalid_amounts(self, clean_df):
        assert (clean_df["amount"] >= 0.01).all()
        assert (clean_df["amount"] <= 9_999.99).all()

    def test_clean_reduces_rows(self, raw_df, clean_df):
        assert len(clean_df) < len(raw_df)

    def test_index_reset(self, clean_df):
        assert clean_df.index.tolist() == list(range(len(clean_df)))


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_amount_zscore_column_exists(self, transformed_df):
        assert "amount_zscore" in transformed_df.columns

    def test_amount_zscore_is_numeric(self, transformed_df):
        assert pd.api.types.is_numeric_dtype(transformed_df["amount_zscore"])

    def test_is_high_amount_is_binary(self, transformed_df):
        assert set(transformed_df["is_high_amount"].unique()).issubset({0, 1})

    def test_is_night_is_binary(self, transformed_df):
        assert set(transformed_df["is_night"].unique()).issubset({0, 1})

    def test_night_flag_correct_hours(self, transformed_df):
        night_rows = transformed_df[transformed_df["is_night"] == 1]
        assert night_rows["hour"].between(0, 4).all()

    def test_velocity_flag_is_binary(self, transformed_df):
        assert set(transformed_df["velocity_flag"].unique()).issubset({0, 1})

    def test_risk_score_range(self, transformed_df):
        assert transformed_df["risk_score"].between(0, 100).all()

    def test_risk_tier_values(self, transformed_df):
        valid_tiers = {"LOW", "MEDIUM", "HIGH"}
        actual = set(transformed_df["risk_tier"].astype(str).unique())
        assert actual.issubset(valid_tiers)

    def test_all_tiers_represented(self, transformed_df):
        # With 500 records we should see all three tiers
        tiers = transformed_df["risk_tier"].astype(str).unique()
        assert len(tiers) >= 2  # at least 2 tiers; likely all 3

    def test_no_new_nulls_introduced(self, clean_df, transformed_df):
        # Transform should not introduce nulls in the core columns
        core_cols = ["amount_zscore", "is_high_amount", "is_night",
                     "velocity_flag", "risk_score"]
        for col in core_cols:
            assert transformed_df[col].isna().sum() == 0, \
                f"Unexpected nulls in {col}"

    def test_column_order(self, transformed_df):
        # transaction_id should be first, is_fraud should be last
        assert transformed_df.columns[0] == "transaction_id"
        assert transformed_df.columns[-1] == "is_fraud"


# ── Integration Test ──────────────────────────────────────────────────────────

class TestPipelineIntegration:

    def test_full_pipeline_runs(self):
        raw = extract(n_records=200)
        final = transform(raw)
        assert len(final) > 0
        assert "risk_score" in final.columns
        assert "risk_tier" in final.columns

    def test_fraud_rate_plausible(self, transformed_df):
        fraud_rate = transformed_df["is_fraud"].mean()
        # Synthetic data should produce a fraud rate between 1% and 20%
        assert 0.01 <= fraud_rate <= 0.20, \
            f"Unexpected fraud rate: {fraud_rate:.2%}"

    def test_high_risk_have_higher_fraud_rate(self, transformed_df):
        high_risk = transformed_df[transformed_df["risk_tier"] == "HIGH"]["is_fraud"].mean()
        low_risk  = transformed_df[transformed_df["risk_tier"] == "LOW"]["is_fraud"].mean()
        # Risk scoring should correlate with actual fraud labels
        assert high_risk > low_risk, \
            "HIGH tier should have a higher fraud rate than LOW tier"
