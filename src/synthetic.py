"""
Synthetic scenario generator.

The original dataset has one injected cause pair (a North paid-marketing cut and
an Electronics price rise). A system that only ever sees that one dataset can be
tuned to it without anyone noticing, so the generator now builds several
scenarios, each with a known ground truth:

  default        the original dataset, unchanged
  channel_loss   one region loses a paid channel; only that channel's orders fall
  demand_shock   demand falls everywhere and no driver in the data moved
  flat           nothing changes

`Scenario.expected_causes` lists the evidence ids a correct investigation should
call supported. An empty tuple means the correct answer is that the data does
not support any tested cause.

The default scenario must keep producing byte-identical CSVs (the committed
data and most of the test suite depend on them), so the random draws happen in
the same order as before and extra draws are made only when a scenario asks for
an effect the default does not have.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = date(2025, 9, 1)
END_DATE = date(2026, 8, 31)
CAUSE_DATE = date(2026, 8, 1)  # final month of the period

REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["New", "Returning", "VIP"]
CATEGORIES = ["Electronics", "Apparel", "Home", "Beauty", "Sports"]
CHANNELS = ["Organic", "Paid", "Email", "Referral"]
DEFAULT_CHANNEL_PROBS = (0.45, 0.30, 0.15, 0.10)

N_CUSTOMERS = 800
N_PRODUCTS_PER_CATEGORY = 8


@dataclass(frozen=True)
class MarketingCut:
    """Spend in one channel and region is cut from CAUSE_DATE.

    demand_drop is the share of orders lost. With channel_only False it applies
    to every order in the region (the original scenario); with channel_only True
    it applies only to orders recorded under `channel`.
    """
    region: str
    spend_cut_pct: float
    demand_drop: float
    channel: str = "Paid"
    channel_only: bool = False


@dataclass(frozen=True)
class PriceChange:
    """Prices in one category change from CAUSE_DATE (+0.10 is a 10% rise);
    demand_change is the resulting change in orders (-0.35 is 35% fewer)."""
    category: str
    pct: float
    demand_change: float


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    description: str
    seed: int = 42
    marketing_cuts: tuple = ()
    price_changes: tuple = ()
    uniform_demand_drop: float = 0.0       # demand falls everywhere, no driver in the data
    channel_probs: tuple = DEFAULT_CHANNEL_PROBS
    expected_causes: tuple = ()            # evidence ids that should come out supported


SCENARIOS = {
    s.name: s for s in (
        Scenario(
            name="default",
            title="North marketing cut and Electronics price rise",
            description=(
                "Paid marketing in North is cut 45% and Electronics prices rise 10% in the "
                "final month. Both cost orders."
            ),
            marketing_cuts=(MarketingCut("North", spend_cut_pct=0.45, demand_drop=0.45),),
            price_changes=(PriceChange("Electronics", pct=0.10, demand_change=-0.35),),
            expected_causes=("stat_marketing_North", "stat_price_Electronics"),
        ),
        Scenario(
            name="channel_loss",
            title="West loses its paid channel",
            description=(
                "A paid-acquisition-heavy business. Paid spend in West is cut 70% in the final "
                "month and West's Paid orders fall with it; the region's other channels are "
                "untouched, so the loss is concentrated in one channel."
            ),
            seed=7,
            marketing_cuts=(
                MarketingCut("West", spend_cut_pct=0.70, demand_drop=0.70, channel_only=True),
            ),
            channel_probs=(0.15, 0.70, 0.10, 0.05),
            expected_causes=("stat_marketing_West",),
        ),
        Scenario(
            name="demand_shock",
            title="Demand falls everywhere, nothing in the data explains it",
            description=(
                "Orders fall 25% in every region and category in the final month. Spend and "
                "prices are unchanged, so no candidate cause in the data moved. The correct "
                "answer is that the data cannot say why."
            ),
            seed=11,
            uniform_demand_drop=0.25,
        ),
        Scenario(
            name="flat",
            title="Nothing changes",
            description="No injected cause anywhere; the final month is an ordinary month.",
            seed=23,
        ),
    )
}


def get_scenario(name):
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    return SCENARIOS[name]


def _daterange(start, end):
    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(days)]


def _gen_customers(rng):
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


def _gen_products(rng):
    rows = []
    pid = 1
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


def _gen_orders(rng, sc, customers_df, products_df):
    dates = _daterange(START_DATE, END_DATE)
    n_days = len(dates)

    # Baseline daily volume: mild upward trend + weekly seasonality + noise. The
    # trend is kept small so it cannot mask the effect in the final month.
    base_level = 45
    trend = np.linspace(0, 4, n_days)
    day_of_week = np.array([d.weekday() for d in dates])
    weekend_bump = np.where(day_of_week >= 5, 12, 0)
    noise = rng.normal(0, 4, n_days)
    daily_total = np.clip(base_level + trend + weekend_bump + noise, 10, None)

    # Orders are drawn per (region, category) cell each day so a demand cut
    # shrinks the number of orders in that cell rather than only biasing which
    # product is picked.
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

    region_cuts = [c for c in sc.marketing_cuts if not c.channel_only]
    channel_cuts = [c for c in sc.marketing_cuts if c.channel_only]
    price_by_category = {p.category: p for p in sc.price_changes}

    rows = []
    order_num = 1
    for d, total in zip(dates, daily_total):
        post = d >= CAUSE_DATE
        d_ts = pd.Timestamp(d)

        for region in REGIONS:
            for category in CATEGORIES:
                expected = total * region_weight[region] * category_weight[category]
                if post:
                    for cut in region_cuts:
                        if region == cut.region:
                            expected *= (1 - cut.demand_drop)
                    change = price_by_category.get(category)
                    if change is not None:
                        expected *= (1 + change.demand_change)
                    if sc.uniform_demand_drop:
                        expected *= (1 - sc.uniform_demand_drop)

                n_orders = rng.poisson(expected)
                if n_orders == 0:
                    continue

                # Only customers who had already signed up by this date can order.
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
                    if post and category in price_by_category:
                        unit_price = round(unit_price * (1 + price_by_category[category].pct), 2)

                    quantity = int(rng.integers(1, 4))
                    revenue = round(unit_price * quantity, 2)
                    channel = rng.choice(CHANNELS, p=list(sc.channel_probs))

                    # A channel-specific loss removes only that channel's orders.
                    if post and any(
                        region == c.region and channel == c.channel and rng.random() < c.demand_drop
                        for c in channel_cuts
                    ):
                        continue

                    rows.append((
                        f"O{order_num:06d}", customer_id, product_id, d,
                        quantity, revenue, channel,
                    ))
                    order_num += 1

    return pd.DataFrame(rows, columns=[
        "order_id", "customer_id", "product_id", "order_date",
        "quantity", "revenue", "channel",
    ])


def _gen_marketing(rng, sc):
    dates = _daterange(START_DATE, END_DATE)
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
                    if d >= CAUSE_DATE:
                        for cut in sc.marketing_cuts:
                            if cut.channel == channel and cut.region == region:
                                spend *= (1 - cut.spend_cut_pct)
                impressions = int(spend * rng.uniform(8, 12)) if spend > 0 else int(rng.uniform(200, 500))
                rows.append((d, channel, region, round(spend, 2), impressions))
    return pd.DataFrame(rows, columns=["date", "channel", "region", "spend", "impressions"])


def generate(scenario="default"):
    """Build the four tables for a scenario. Returns a dict of DataFrames keyed
    customers / products / orders / marketing. Deterministic for a given scenario."""
    sc = get_scenario(scenario) if isinstance(scenario, str) else scenario
    rng = np.random.default_rng(sc.seed)
    customers = _gen_customers(rng)
    products = _gen_products(rng)
    orders = _gen_orders(rng, sc, customers, products)
    marketing = _gen_marketing(rng, sc)
    return {"customers": customers, "products": products, "orders": orders, "marketing": marketing}


def write_scenario(scenario, out_dir):
    """Generate a scenario and write its four CSVs to out_dir."""
    tables = generate(scenario)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return tables
