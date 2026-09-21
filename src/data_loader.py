"""
Loads the four tables and joins them into the frame the evidence tools work on.

Kept in one place so the app, the scripts and the tests all see exactly the
same enriched table. `load_data` reads a folder of CSVs; `enrich` does the join
on tables already in memory, which is what the upload path uses after its own
validation (see validation.py).
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
SCENARIO_DIR = DEFAULT_DATA_DIR / "scenarios"


def enrich(orders, products, customers, marketing):
    """Return (orders, marketing) where `orders` carries category, region,
    customer_segment, unit_price and unit_cost alongside the raw order columns."""
    enriched = (
        orders
        .merge(
            products[["product_id", "category", "cost"]].rename(columns={"cost": "unit_cost"}),
            on="product_id", how="left",
        )
        .merge(
            customers[["customer_id", "region", "segment"]].rename(
                columns={"segment": "customer_segment"}
            ),
            on="customer_id", how="left",
        )
    )
    enriched["unit_price"] = enriched["revenue"] / enriched["quantity"]

    missing = enriched[["category", "region"]].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"{missing} orders could not be matched to a product or customer")

    return enriched, marketing


def load_data(data_dir=None):
    """Read orders, products, customers and marketing CSVs from a folder and
    return the enriched (orders, marketing) pair."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    orders = pd.read_csv(data_dir / "orders.csv", parse_dates=["order_date"])
    products = pd.read_csv(data_dir / "products.csv")
    customers = pd.read_csv(data_dir / "customers.csv")
    marketing = pd.read_csv(data_dir / "marketing.csv", parse_dates=["date"])
    return enrich(orders, products, customers, marketing)


def scenario_dir(name):
    """Folder holding a scenario's committed CSVs; the default scenario is the
    original data folder."""
    return DEFAULT_DATA_DIR if name == "default" else SCENARIO_DIR / name
