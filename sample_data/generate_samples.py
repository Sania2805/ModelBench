"""
Generate the two synthetic sample datasets shipped with ModelBench.

Run from the project root (with the backend virtual environment active):

    python sample_data/generate_samples.py

The data is fake but has real, learnable patterns, plus some deliberate
messiness (missing values, an ID column, text categories) so you can see how
ModelBench handles them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent
rng = np.random.default_rng(7)


def make_customer_churn(n: int = 600) -> pd.DataFrame:
    """Binary classification: will a customer cancel their subscription (churned = yes/no)?"""
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n).clip(15, 150).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    support_calls = rng.poisson(1.5, n)
    internet = rng.choice(["Fiber", "DSL", "No internet"], n, p=[0.45, 0.4, 0.15])
    age = rng.integers(18, 80, n)

    # Churn is more likely for short tenure, high bills, monthly contracts and many support calls.
    logit = (
        -0.04 * tenure + 0.02 * monthly + 0.45 * support_calls
        + np.where(contract == "Month-to-month", 1.2, np.where(contract == "One year", -0.2, -1.0))
        + np.where(internet == "Fiber", 0.4, 0.0) - 1.8
    )
    churned = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "age": age.astype(float),
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "contract_type": contract,
            "internet_service": internet,
            "support_calls": support_calls,
            "churned": churned,
        }
    )
    # Sprinkle in missing values so imputation has something to do.
    df.loc[rng.choice(n, 30, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "internet_service"] = np.nan
    return df


def make_house_prices(n: int = 500) -> pd.DataFrame:
    """Regression: predict a house's sale price (in thousands)."""
    area = rng.normal(1600, 450, n).clip(500, 4000).round()
    bedrooms = rng.integers(1, 6, n)
    age = rng.integers(0, 60, n)
    neighborhood = rng.choice(["Downtown", "Suburb", "Rural", "Lakeside"], n, p=[0.3, 0.4, 0.2, 0.1])
    garage = rng.choice(["yes", "no"], n, p=[0.6, 0.4])
    distance = rng.gamma(2.0, 4.0, n).round(1)

    base = {"Downtown": 120, "Suburb": 60, "Rural": 0, "Lakeside": 180}
    price = (
        0.15 * area + 12 * bedrooms - 1.1 * age - 2.5 * distance
        + np.vectorize(base.get)(neighborhood) + np.where(garage == "yes", 20, 0)
        + rng.normal(0, 25, n) + 40
    ).round(1)

    df = pd.DataFrame(
        {
            "area_sqft": area,
            "bedrooms": bedrooms,
            "house_age_years": age,
            "neighborhood": neighborhood,
            "has_garage": garage,
            "distance_to_center_km": distance,
            "price_thousands": price,
        }
    )
    df.loc[rng.choice(n, 25, replace=False), "distance_to_center_km"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "neighborhood"] = np.nan
    return df


if __name__ == "__main__":
    make_customer_churn().to_csv(OUTPUT_DIR / "customer_churn_classification.csv", index=False)
    make_house_prices().to_csv(OUTPUT_DIR / "house_prices_regression.csv", index=False)
    print(f"Sample datasets written to {OUTPUT_DIR}")
