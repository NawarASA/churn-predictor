"""Top-level CLI entry point wiring the ``train`` and ``predict`` subcommands.

    python -m churn_predictor train
    python -m churn_predictor predict --input data/new_customers.csv
    python -m churn_predictor generate-data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churn_predictor import generate_data, predict, train


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="churn_predictor", description="Customer churn prediction pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate churn models.")
    train_parser.add_argument("--data", type=Path, default=train.DEFAULT_DATA_PATH)
    train_parser.add_argument("--model-out", type=Path, default=train.DEFAULT_MODEL_PATH)
    train_parser.add_argument("--reports-dir", type=Path, default=train.DEFAULT_REPORTS_DIR)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)

    predict_parser = subparsers.add_parser(
        "predict", help="Score new customers with a saved model."
    )
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, default=None)
    predict_parser.add_argument("--model", type=Path, default=predict.DEFAULT_MODEL_PATH)

    gen_parser = subparsers.add_parser("generate-data", help="Regenerate the synthetic dataset.")
    gen_parser.add_argument("--rows", type=int, default=generate_data.DEFAULT_N_ROWS)
    gen_parser.add_argument("--seed", type=int, default=generate_data.DEFAULT_SEED)
    gen_parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "customers.csv"
    )

    args = parser.parse_args()

    if args.command == "train":
        train.run_training(
            data_path=args.data,
            model_path=args.model_out,
            reports_dir=args.reports_dir,
            test_size=args.test_size,
            random_state=args.seed,
        )
    elif args.command == "predict":
        model = predict.load_model(args.model)
        import pandas as pd

        customers = pd.read_csv(args.input)
        scored = predict.score_customers(model, customers)
        print(scored[["customer_id", "churn_probability"]].to_string(index=False))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(args.output, index=False)
            print(f"\nWrote scored CSV to {args.output}")
    elif args.command == "generate-data":
        df = generate_data.generate_customers(n_rows=args.rows, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} rows to {args.out}")
    else:  # pragma: no cover - argparse enforces valid choices
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
