"""
Loads the four CSVs and joins them into the frame the evidence tools work on.

Kept in one place so the app, the scripts and the tests all see exactly the
same enriched table.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def load_data(data_dir=None):
    """Return (orders, marketing) where `orders` carries category, region,
    customer_segment and unit_price alongside the raw order columns."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    orders = pd.read_csv(data_dir / "orders.csv", parse_dates=["order_date"])
    products = pd.read_csv(data_dir / "products.csv")
    customers = pd.read_csv(data_dir / "customers.csv")
    marketing = pd.read_csv(data_dir / "marketing.csv", parse_dates=["date"])

    enriched = (
        orders
        .merge(products[["product_id", "category"]], on="product_id", how="left")
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
