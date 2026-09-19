"""
Synthetic dataset generator for KyuYaar (Layer 1).

Why synthetic: we need a dataset where the true root cause of the revenue
drop is known in advance, so baseline_trend() and segment_breakdown() can be
verified against ground truth instead of just producing plausible-looking
output. Real scraped data has no such ground truth.

Injected root cause (kept deliberately simple for Layer 1, two effects that
overlap in a way segment_breakdown() should be able to surface):

  1. Category-wide price hike: on ROOT_CAUSE_DATE, "Electronics" prices
     jump 25%. Demand has mild elasticity, so Electronics order frequency
     drops a bit everywhere (moderate, category-level effect).

  2. Region-specific marketing cut: on the same date, paid marketing spend
     in the North region is cut ~45%. This reduces order frequency for
     North customers across all categories (a stronger, region-level
     effect than the price hike alone).

  North + Electronics together should show the sharpest drop, since both
  effects stack there. Layer 1 doesn't need to detect the interaction --
  just recover "North" and "Electronics" as the two segments most
  responsible for the drop.

Output: data/customers.csv, data/products.csv, data/orders.csv,
data/marketing.csv

marketing.csv is generated now (so Layer 2 doesn't require regenerating
everything) but is not consumed by Layer 1's functions.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data"

SEED = 42
rng = np.random.default_rng(SEED)

START_DATE = date(2025, 9, 1)
END_DATE = date(2026, 8, 31)
ROOT_CAUSE_DATE = date(2026, 8, 1)  # final month of the period

REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["New", "Returning", "VIP"]
CATEGORIES = ["Electronics", "Apparel", "Home", "Beauty", "Sports"]
CHANNELS = ["Organic", "Paid", "Email", "Referral"]

N_CUSTOMERS = 800
N_PRODUCTS_PER_CATEGORY = 8
PRICE_HIKE_PCT = 0.10                # kept modest -- if this is too big it
                                      # offsets the quantity drop and revenue
                                      # can rise even though volume fell
NORTH_MARKETING_CUT_PCT = 0.45
ELECTRONICS_ELASTICITY_DROP = 0.35   # order-frequency drop from price hike
NORTH_DEMAND_DROP = 0.45             # order-frequency drop from marketing cut


def daterange(start, end):
    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(days)]


def gen_customers():
    signup_start = START_DATE - timedelta(days=730)
    rows = []
    # Segment mix isn't uniform -- Returning is most common, VIP is rare.
    segment_probs = [0.30, 0.55, 0.15]
    for i in range(N_CUSTOMERS):
        cust_id = f"C{i+1:04d}"
        signup_date = signup_start + timedelta(
            days=int(rng.integers(0, (END_DATE - signup_start).days))
        )
        region = rng.choice(REGIONS)
        segment = rng.choice(SEGMENTS, p=segment_probs)
        rows.append((cust_id, signup_date, region, segment))
    return pd.DataFrame(rows, columns=["customer_id", "signup_date", "region", "segment"])


def gen_products():
    rows = []
    pid = 1
    # Rough base price bands per category, cost as a category-typical margin.
    price_bands = {
        "Electronics": (40, 400),
        "Apparel": (15, 90),
        "Home": (20, 150),
        "Beauty": (10, 60),
        "Sports": (15, 120),
    }
    margin_pct = {
        "Electronics": 0.35,
        "Apparel": 0.55,
        "Home": 0.45,
        "Beauty": 0.60,
        "Sports": 0.50,
    }
    for category in CATEGORIES:
        low, high = price_bands[category]
        for _ in range(N_PRODUCTS_PER_CATEGORY):
            product_id = f"P{pid:04d}"
            price = round(float(rng.uniform(low, high)), 2)
            cost = round(price * (1 - margin_pct[category]), 2)
            rows.append((product_id, category, price, cost))
            pid += 1
    return pd.DataFrame(rows, columns=["product_id", "category", "price", "cost"])


def gen_orders(customers_df, products_df):
    dates = daterange(START_DATE, END_DATE)
    n_days = len(dates)

    # Baseline daily order volume: mild upward trend + weekly seasonality + noise.
    # Trend is kept small deliberately -- a strong secular growth trend can
    # mask the injected root-cause drop in the final month.
    base_level = 45
    trend = np.linspace(0, 4, n_days)                # gentle growth over the year
    day_of_week = np.array([d.weekday() for d in dates])
    weekend_bump = np.where(day_of_week >= 5, 12, 0)  # more orders on weekends
    noise = rng.normal(0, 4, n_days)
    daily_total = np.clip(base_level + trend + weekend_bump + noise, 10, None)

    # Orders are generated per (region, category) cell each day, with
    # expected volume split proportionally by region size and evenly by
    # category. This way the region/category demand cuts actually shrink
    # the number of orders drawn for that cell -- not just bias which
    # product gets picked within an order that would have happened anyway.
    region_counts = customers_df["region"].value_counts()
    region_weight = (region_counts / region_counts.sum()).to_dict()
    category_weight = {c: 1 / len(CATEGORIES) for c in CATEGORIES}

    customers_by_region = {
        r: customers_df.loc[customers_df.region == r]
                        .assign(signup_date=lambda x: pd.to_datetime(x.signup_date))
                        .sort_values("signup_date")[["customer_id", "signup_date"]]
                        .reset_index(drop=True)
        for r in REGIONS
    }
    products_by_category = {
        c: products_df.loc[products_df.category == c, ["product_id", "price"]]
        for c in CATEGORIES
    }

    rows = []
    order_num = 1
    for d, total in zip(dates, daily_total):
        post_root_cause = d >= ROOT_CAUSE_DATE
        d_ts = pd.Timestamp(d)

        for region in REGIONS:
            for category in CATEGORIES:
                expected = total * region_weight[region] * category_weight[category]
                if post_root_cause:
                    if region == "North":
                        expected *= (1 - NORTH_DEMAND_DROP)
                    if category == "Electronics":
                        expected *= (1 - ELECTRONICS_ELASTICITY_DROP)

                n_orders = rng.poisson(expected)
                if n_orders == 0:
                    continue

                # Only customers who had already signed up by this date are
                # eligible to place an order -- avoids the (obviously wrong)
                # possibility of an order predating a customer's signup.
                region_customers = customers_by_region[region]
                eligible_count = region_customers["signup_date"].searchsorted(d_ts, side="right")
                if eligible_count == 0:
                    continue
                eligible_ids = region_customers["customer_id"].to_numpy()[:eligible_count]

                cell_products = products_by_category[category]

                for _ in range(n_orders):
                    customer_id = rng.choice(eligible_ids)
                    prod_row = cell_products.iloc[rng.integers(0, len(cell_products))]
                    product_id = prod_row["product_id"]
                    unit_price = prod_row["price"]
                    if post_root_cause and category == "Electronics":
                        unit_price = round(unit_price * (1 + PRICE_HIKE_PCT), 2)

                    quantity = int(rng.integers(1, 4))
                    revenue = round(unit_price * quantity, 2)
                    channel = rng.choice(CHANNELS, p=[0.45, 0.30, 0.15, 0.10])

                    rows.append((
                        f"O{order_num:06d}", customer_id, product_id, d,
                        quantity, revenue, channel,
                    ))
                    order_num += 1

    return pd.DataFrame(rows, columns=[
        "order_id", "customer_id", "product_id", "order_date",
        "quantity", "revenue", "channel",
    ])


def gen_marketing():
    dates = daterange(START_DATE, END_DATE)
    rows = []
    base_spend = {
        "Organic": 0,       # organic has no spend by definition
        "Paid": 800,
        "Email": 150,
        "Referral": 100,
    }
    for d in dates:
        for region in REGIONS:
            for channel in CHANNELS:
                if channel == "Organic":
                    spend = 0.0
                else:
                    spend = base_spend[channel] * float(rng.uniform(0.85, 1.15))
                    if channel == "Paid" and region == "North" and d >= ROOT_CAUSE_DATE:
                        spend *= (1 - NORTH_MARKETING_CUT_PCT)
                impressions = int(spend * rng.uniform(8, 12)) if spend > 0 else int(rng.uniform(200, 500))
                rows.append((d, channel, region, round(spend, 2), impressions))
    return pd.DataFrame(rows, columns=["date", "channel", "region", "spend", "impressions"])


def main(out_dir=None):
    customers_df = gen_customers()
    products_df = gen_products()
    orders_df = gen_orders(customers_df, products_df)
    marketing_df = gen_marketing()

    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    customers_df.to_csv(out_dir / "customers.csv", index=False)
    products_df.to_csv(out_dir / "products.csv", index=False)
    orders_df.to_csv(out_dir / "orders.csv", index=False)
    marketing_df.to_csv(out_dir / "marketing.csv", index=False)

    print(f"customers: {len(customers_df)} rows")
    print(f"products:  {len(products_df)} rows")
    print(f"orders:    {len(orders_df)} rows")
    print(f"marketing: {len(marketing_df)} rows")
    print(f"date range: {START_DATE} to {END_DATE}")
    print(f"root cause date: {ROOT_CAUSE_DATE}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
