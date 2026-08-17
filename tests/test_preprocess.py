from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_predictor.preprocess import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    clean_data,
    engineer_features,
    get_feature_target,
    prepare_features,
)


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["CUST00001", "CUST00002", "CUST00003"],
            "tenure_months": [0, 12, 24],
            "monthly_charges": [50.0, 70.0, 90.0],
            "total_charges": [np.nan, 840.0, 2160.0],
            "contract_type": ["month-to-month", "one-year", "two-year"],
            "payment_method": ["electronic check", "credit card", "bank transfer"],
            "has_support_calls": [3, 1, 0],
            "internet_service": ["fiber optic", "DSL", "none"],
            "num_add_on_services": [0, 2, 4],
            "is_senior": [0, 1, 0],
            "churned": [1, 0, 0],
        }
    )


class TestCleanData:
    def test_fills_missing_total_charges_from_tenure_and_monthly(self):
        df = _sample_raw_df()
        cleaned = clean_data(df)

        # row 0 had NaN total_charges, tenure_months=0 -> imputed as 0 * 50 = 0
        assert cleaned.loc[0, "total_charges"] == pytest.approx(0.0)
        # untouched rows keep their original values
        assert cleaned.loc[1, "total_charges"] == pytest.approx(840.0)
        assert cleaned.loc[2, "total_charges"] == pytest.approx(2160.0)

    def test_clips_negative_values_to_zero(self):
        df = _sample_raw_df()
        df.loc[0, "has_support_calls"] = -5
        cleaned = clean_data(df)
        assert cleaned.loc[0, "has_support_calls"] == 0

    def test_does_not_mutate_input(self):
        df = _sample_raw_df()
        original = df.copy(deep=True)
        clean_data(df)
        pd.testing.assert_frame_equal(df, original)


class TestEngineerFeatures:
    def test_adds_avg_charge_per_month_column(self):
        df = clean_data(_sample_raw_df())
        engineered = engineer_features(df)
        assert "avg_charge_per_month" in engineered.columns

    def test_avg_charge_per_month_is_correct_for_normal_tenure(self):
        df = clean_data(_sample_raw_df())
        engineered = engineer_features(df)
        # row 1: total_charges=840, tenure=12 -> 70.0
        assert engineered.loc[1, "avg_charge_per_month"] == pytest.approx(70.0)
        # row 2: total_charges=2160, tenure=24 -> 90.0
        assert engineered.loc[2, "avg_charge_per_month"] == pytest.approx(90.0)

    def test_zero_tenure_falls_back_to_monthly_charges(self):
        df = clean_data(_sample_raw_df())
        engineered = engineer_features(df)
        # row 0: tenure_months=0 -> avg falls back to monthly_charges (50.0)
        assert engineered.loc[0, "avg_charge_per_month"] == pytest.approx(50.0)

    def test_no_nan_or_inf_produced(self):
        df = clean_data(_sample_raw_df())
        engineered = engineer_features(df)
        col = engineered["avg_charge_per_month"]
        assert not col.isna().any()
        assert np.isfinite(col).all()


class TestGetFeatureTarget:
    def test_target_and_id_dropped_from_features(self):
        df = prepare_features(_sample_raw_df())
        X, y = get_feature_target(df)
        assert "churned" not in X.columns
        assert "customer_id" not in X.columns

    def test_target_series_matches_original(self):
        df = prepare_features(_sample_raw_df())
        _, y = get_feature_target(df)
        assert list(y) == [1, 0, 0]

    def test_all_expected_feature_columns_present(self):
        df = prepare_features(_sample_raw_df())
        X, _ = get_feature_target(df)
        for col in CATEGORICAL_COLS + NUMERIC_COLS:
            assert col in X.columns

    def test_no_target_leakage_even_without_id_column(self):
        df = prepare_features(_sample_raw_df()).drop(columns=["customer_id"])
        X, y = get_feature_target(df)
        assert "churned" not in X.columns
        assert y is not None
