"""Cleaning and feature engineering for the churn dataset.

Every function here is pure (takes a DataFrame, returns a new DataFrame or
array — no I/O, no global state) so they're each independently testable with
small hand-built DataFrames.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "churned"
ID_COL = "customer_id"

CATEGORICAL_COLS = ["contract_type", "payment_method", "internet_service"]
BASE_NUMERIC_COLS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "has_support_calls",
    "num_add_on_services",
    "is_senior",
]
ENGINEERED_NUMERIC_COLS = ["avg_charge_per_month"]
NUMERIC_COLS = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and obviously-invalid values.

    - ``total_charges`` missing (e.g. a brand-new customer with no billing
      cycle yet) is imputed as ``monthly_charges * tenure_months``.
    - Any negative numeric values (shouldn't occur, but guard anyway) are
      clipped to 0.
    """
    df = df.copy()

    if "total_charges" in df.columns:
        missing = df["total_charges"].isna()
        if missing.any():
            df.loc[missing, "total_charges"] = (
                df.loc[missing, "monthly_charges"] * df.loc[missing, "tenure_months"]
            )

    clip_cols = (
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "has_support_calls",
        "num_add_on_services",
    )
    for col in clip_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional features from the cleaned raw columns.

    Adds ``avg_charge_per_month``: ``total_charges / tenure_months``, with
    ``tenure_months == 0`` treated as ``monthly_charges`` (a brand-new
    customer's "average" is just their current monthly rate) so the division
    never produces inf/NaN.
    """
    df = df.copy()

    safe_tenure = df["tenure_months"].replace(0, pd.NA)
    avg_charge = df["total_charges"] / safe_tenure
    df["avg_charge_per_month"] = avg_charge.fillna(df["monthly_charges"]).astype(float)

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full clean -> engineer pipeline and return the feature-ready frame."""
    return engineer_features(clean_data(df))


def get_feature_target(
    df: pd.DataFrame, target_col: str = TARGET_COL, id_col: str = ID_COL
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a prepared DataFrame into feature matrix ``X`` and target ``y``.

    Drops the identifier and target columns from ``X`` so the target can
    never leak into the features used for training/inference.
    """
    drop_cols = [c for c in (target_col, id_col) if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col] if target_col in df.columns else None
    assert target_col not in X.columns, "target column leaked into feature matrix"
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the sklearn ColumnTransformer used inside the model Pipeline.

    One-hot encodes the categorical columns and standard-scales the numeric
    columns (harmless for tree models, needed for logistic regression, so one
    shared preprocessor works for both).
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )


def build_pipeline(model) -> Pipeline:
    """Wrap a fitted-or-unfitted estimator with the shared preprocessing step."""
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
