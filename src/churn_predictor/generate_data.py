"""Generate a synthetic customer-churn dataset.

There's no bundled real-world dataset for this project, so this module builds a
realistic stand-in: customer/account/usage features are drawn from sensible
distributions, and churn is simulated from a logistic model over those features
plus noise, so the relationship is real (and learnable) but not perfectly
separable. Seeded for reproducibility.

Run directly to (re)write ``data/customers.csv``:

    python -m churn_predictor.generate_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_N_ROWS = 4000
DEFAULT_SEED = 42

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
CONTRACT_WEIGHTS = [0.55, 0.25, 0.20]

PAYMENT_METHODS = ["electronic check", "mailed check", "bank transfer", "credit card"]
PAYMENT_WEIGHTS = [0.35, 0.20, 0.23, 0.22]

INTERNET_SERVICES = ["DSL", "fiber optic", "none"]
INTERNET_WEIGHTS = [0.35, 0.45, 0.20]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(n_rows: int = DEFAULT_N_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate ``n_rows`` synthetic customer records with a realistic churn signal.

    Churn probability is built from a logistic model of the underlying features
    (month-to-month contracts, low tenure, many support calls, and high charges
    push probability up; long contracts and add-on services push it down) plus
    Gaussian noise on the logit, so the resulting labels are correlated with the
    features but not a deterministic function of them.
    """
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST{i:05d}" for i in range(1, n_rows + 1)]

    tenure_months = rng.integers(0, 73, size=n_rows)

    contract_type = rng.choice(CONTRACT_TYPES, size=n_rows, p=CONTRACT_WEIGHTS)
    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows, p=PAYMENT_WEIGHTS)
    internet_service = rng.choice(INTERNET_SERVICES, size=n_rows, p=INTERNET_WEIGHTS)

    # Base monthly charge depends on internet service tier plus noise.
    base_charge = np.select(
        [
            internet_service == "fiber optic",
            internet_service == "DSL",
            internet_service == "none",
        ],
        [
            rng.normal(85, 12, size=n_rows),
            rng.normal(55, 10, size=n_rows),
            rng.normal(25, 6, size=n_rows),
        ],
    )
    num_add_on_services = rng.integers(0, 6, size=n_rows)
    add_on_charge = num_add_on_services * rng.normal(5, 1, size=n_rows)
    monthly_charges = np.clip(base_charge + add_on_charge, 15, None)
    monthly_charges = np.round(monthly_charges, 2)

    # total_charges accrues roughly monthly_charges * tenure, with some jitter
    # (billing adjustments, promos, partial months) and a floor at 0.
    jitter = rng.normal(1.0, 0.08, size=n_rows)
    total_charges = np.clip(monthly_charges * tenure_months * jitter, 0, None)
    total_charges = np.round(total_charges, 2)

    has_support_calls = rng.poisson(lam=np.where(tenure_months < 12, 1.8, 0.8), size=n_rows)
    has_support_calls = np.clip(has_support_calls, 0, 12)

    is_senior = rng.binomial(1, 0.16, size=n_rows)

    # --- churn simulation -------------------------------------------------
    # Logistic model: positive coefficients push churn probability up.
    contract_coef = np.select(
        [
            contract_type == "month-to-month",
            contract_type == "one-year",
            contract_type == "two-year",
        ],
        [1.1, -0.4, -1.3],
    )
    payment_coef = np.where(payment_method == "electronic check", 0.45, -0.1)
    internet_coef = np.select(
        [internet_service == "fiber optic", internet_service == "DSL", internet_service == "none"],
        [0.35, 0.0, -0.25],
    )

    logit = (
        -1.1
        + contract_coef
        + payment_coef
        + internet_coef
        - 0.045 * tenure_months
        + 0.28 * has_support_calls
        + 0.010 * (monthly_charges - 60)
        - 0.22 * num_add_on_services
        + 0.30 * is_senior
        + rng.normal(0, 0.75, size=n_rows)  # noise so it's not perfectly separable
    )
    churn_prob = _sigmoid(logit)
    churned = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "has_support_calls": has_support_calls,
            "internet_service": internet_service,
            "num_add_on_services": num_add_on_services,
            "is_senior": is_senior,
            "churned": churned,
        }
    )

    # Sprinkle a small amount of realistic missingness into total_charges
    # (e.g. brand-new customers with no billing cycle yet) for preprocessing
    # to have something real to handle.
    missing_mask = rng.random(n_rows) < 0.01
    df.loc[missing_mask, "total_charges"] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_N_ROWS, help="Number of rows to generate."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "customers.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    df = generate_customers(n_rows=args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    churn_rate = df["churned"].mean()
    print(f"Wrote {len(df)} rows to {args.out} (churn rate: {churn_rate:.1%})")


if __name__ == "__main__":
    main()
