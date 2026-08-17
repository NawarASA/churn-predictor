"""Train, compare, and evaluate churn models.

Trains Logistic Regression and Random Forest classifiers (each wrapped in the
shared preprocessing Pipeline from :mod:`churn_predictor.preprocess`), picks
the better of the two by cross-validated ROC-AUC on the training split, then
reports held-out test metrics and saves the winning fitted pipeline.

Run directly:

    python -m churn_predictor.train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from churn_predictor.preprocess import build_pipeline, get_feature_target, prepare_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

CANDIDATE_MODELS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


def select_best_model(
    X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5, random_state: int = 42
) -> tuple[str, dict]:
    """Cross-validate each candidate model on ROC-AUC and return the winner's name + CV scores."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = {}
    for name, make_model in CANDIDATE_MODELS.items():
        pipeline = build_pipeline(make_model())
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_results[name] = {
            "mean_roc_auc": float(scores.mean()),
            "std_roc_auc": float(scores.std()),
        }
    best_name = max(cv_results, key=lambda n: cv_results[n]["mean_roc_auc"])
    return best_name, cv_results


def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute held-out test metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def save_confusion_matrix(pipeline, X_test, y_test, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, pipeline.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["stayed", "churned"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix (held-out test set)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(pipeline, X_test, y_test, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_title("ROC Curve (held-out test set)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_training(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> dict:
    """Full training pipeline: load -> prepare -> split -> select -> fit -> evaluate -> save.

    Returns a summary dict (best model name, CV results, test metrics) so
    callers (CLI, tests) can inspect results without parsing stdout.
    """
    raw = pd.read_csv(data_path)
    prepared = prepare_features(raw)
    X, y = get_feature_target(prepared)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    best_name, cv_results = select_best_model(X_train, y_train, random_state=random_state)
    print("Cross-validated ROC-AUC by model:")
    for name, result in cv_results.items():
        marker = " <- selected" if name == best_name else ""
        print(f"  {name}: {result['mean_roc_auc']:.4f} (+/- {result['std_roc_auc']:.4f}){marker}")

    best_pipeline = build_pipeline(CANDIDATE_MODELS[best_name]())
    best_pipeline.fit(X_train, y_train)

    metrics = evaluate(best_pipeline, X_test, y_test)
    print(f"\nHeld-out test metrics ({best_name}):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    if save_artifacts:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, model_path)
        print(f"\nSaved fitted pipeline to {model_path}")

        save_confusion_matrix(best_pipeline, X_test, y_test, reports_dir / "confusion_matrix.png")
        save_roc_curve(best_pipeline, X_test, y_test, reports_dir / "roc_curve.png")
        print(f"Saved confusion_matrix.png and roc_curve.png to {reports_dir}")

    return {
        "best_model": best_name,
        "cv_results": cv_results,
        "test_metrics": metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to training CSV."
    )
    parser.add_argument(
        "--model-out", type=Path, default=DEFAULT_MODEL_PATH, help="Where to save the fitted model."
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Where to save plots."
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to also dump the run summary as JSON.",
    )
    args = parser.parse_args()

    summary = run_training(
        data_path=args.data,
        model_path=args.model_out,
        reports_dir=args.reports_dir,
        test_size=args.test_size,
        random_state=args.seed,
    )

    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
