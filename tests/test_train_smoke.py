from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from churn_predictor.generate_data import generate_customers
from churn_predictor.preprocess import build_pipeline, get_feature_target, prepare_features
from churn_predictor.train import evaluate, run_training, select_best_model


@pytest.fixture
def tiny_dataset() -> pd.DataFrame:
    # Small but big enough for a stratified split + 5-fold CV to see both
    # classes in every fold (avoids flaky single-class-in-fold warnings).
    return generate_customers(n_rows=200, seed=7)


def test_generate_customers_has_expected_schema_and_signal(tiny_dataset):
    expected_cols = {
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "payment_method",
        "has_support_calls",
        "internet_service",
        "num_add_on_services",
        "is_senior",
        "churned",
    }
    assert expected_cols.issubset(set(tiny_dataset.columns))
    assert set(tiny_dataset["churned"].unique()).issubset({0, 1})
    # both classes should show up so training/CV is possible
    assert tiny_dataset["churned"].nunique() == 2


def test_pipeline_fits_and_predicts_without_error(tiny_dataset):
    prepared = prepare_features(tiny_dataset)
    X, y = get_feature_target(prepared)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )

    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.all((probs >= 0) & (probs <= 1))


def test_evaluate_returns_all_expected_metrics(tiny_dataset):
    prepared = prepare_features(tiny_dataset)
    X, y = get_feature_target(prepared)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_select_best_model_picks_a_known_candidate(tiny_dataset):
    prepared = prepare_features(tiny_dataset)
    X, y = get_feature_target(prepared)
    best_name, cv_results = select_best_model(X, y, cv_folds=3, random_state=0)
    assert best_name in {"logistic_regression", "random_forest"}
    assert set(cv_results.keys()) == {"logistic_regression", "random_forest"}


def test_run_training_end_to_end_on_tiny_sample(tmp_path, tiny_dataset):
    data_path = tmp_path / "tiny_customers.csv"
    tiny_dataset.to_csv(data_path, index=False)

    summary = run_training(
        data_path=data_path,
        model_path=tmp_path / "model.joblib",
        reports_dir=tmp_path / "reports",
        test_size=0.3,
        random_state=0,
    )

    assert summary["best_model"] in {"logistic_regression", "random_forest"}
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "reports" / "confusion_matrix.png").exists()
    assert (tmp_path / "reports" / "roc_curve.png").exists()
    assert 0.0 <= summary["test_metrics"]["roc_auc"] <= 1.0
