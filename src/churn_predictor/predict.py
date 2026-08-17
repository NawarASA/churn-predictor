"""CLI to score new customers with a saved churn model.

    python -m churn_predictor predict --input data/new_customers.csv
    python -m churn_predictor predict --input data/new_customers.csv --output scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from churn_predictor.preprocess import get_feature_target, prepare_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python -m churn_predictor train` first."
        )
    return joblib.load(model_path)


def score_customers(model, df: pd.DataFrame) -> pd.DataFrame:
    """Score a raw customer DataFrame (same schema as training data, minus the target).

    Returns the original rows with a ``churn_probability`` column appended.
    """
    prepared = prepare_features(df)
    X, _ = get_feature_target(prepared)
    probabilities = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["churn_probability"] = probabilities
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV of customers to score.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to write the scored CSV."
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved model."
    )
    args = parser.parse_args()

    model = load_model(args.model)
    customers = pd.read_csv(args.input)
    scored = score_customers(model, customers)

    id_col = "customer_id" if "customer_id" in scored.columns else scored.index.name or "row"
    display_cols = [
        c for c in (id_col, "churn_probability") if c in scored.columns or c == "churn_probability"
    ]
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(scored[display_cols].to_string(index=(id_col not in scored.columns)))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False)
        print(f"\nWrote scored CSV to {args.output}")


if __name__ == "__main__":
    main()
