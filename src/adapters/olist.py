"""
Olist adapter -- Layer 9.

Maps the raw Olist Brazilian E-Commerce tables (Kaggle: olistbr/brazilian-ecommerce,
plus the separate "Marketing Funnel by Olist" dataset) onto KyuYaar's canonical
schema: orders.csv (order_id, customer_id, product_id, order_date, quantity,
revenue, channel), products.csv (product_id, category, cost[, price]),
customers.csv (customer_id, region[, segment, signup_date]) and marketing.csv
(date, channel, region, spend[, impressions]) -- see src/validation.py:SCHEMA
for the exact contract this adapter has to satisfy.

The analytical tools in src/ (baseline_trend, segment_breakdown,
aov_volume_decomposition, marketing_effect, marketing_channel_analysis,
price_effect) are not touched and never see a raw Olist column. This module's
only job is producing four clean canonical frames from Olist's raw tables;
everything downstream stays exactly as it is for the synthetic data.

A few of Olist's raw columns don't map cleanly onto a canonical field because
the concept simply isn't in the data (order-level marketing channel, product
cost, marketing spend in currency). Rather than invent a number, this module
fills those with an explicit, documented placeholder and records the gap in
MappingReport.notes so it surfaces in the Layer 9 write-up instead of being
silently absorbed. See MAPPING_NOTES below for the full list.
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- raw files ---

# Brazilian E-Commerce Public Dataset by Olist (core).
CORE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",   # optional
}

# Marketing Funnel by Olist -- a separate Kaggle dataset, joined via seller_id.
FUNNEL_FILES = {
    "sellers": "olist_sellers_dataset.csv",
    "leads": "olist_marketing_qualified_leads_dataset.csv",
    "deals": "olist_closed_deals_dataset.csv",
}

REQUIRED_CORE = ("orders", "order_items", "products", "customers")

# Brazil's 5 official macro-regions. Olist's real geography is customer_state
# (27 states), which is finer than this but heavily skewed -- Sao Paulo state
# alone carries roughly 40% of orders, and a dozen states have only a handful.
# Bucketing into macro-regions gives a small number of comparison groups with
# enough volume each, which is what segment_breakdown and the cause tests
# actually need. customer_state is still in the adapter's intermediate frame
# for anyone who wants the finer cut later.
STATE_TO_MACROREGION = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MS": "Centro-Oeste", "MT": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

# Deterministic fallback for bucketing the marketing funnel's free-text `origin`
# values into the same small channel vocabulary the synthetic data uses (Paid /
# Organic / Email / Referral), so marketing_channel_analysis compares like with
# like. See propose_channel_mapping() for how an LLM may refine this instead of
# the adapter guessing keyword-by-keyword.
DEFAULT_CHANNEL_MAP = {
    "organic_search": "Organic", "unknown": "Organic", "direct_traffic": "Organic",
    "social": "Paid", "paid_search": "Paid", "display": "Paid", "other_publicities": "Paid",
    "email": "Email",
    "referral": "Referral",
}
FALLBACK_CHANNEL = "Other"
CHANNEL_VOCAB = ("Paid", "Organic", "Email", "Referral", "Other")

# Canonical columns this adapter must fill, and why each one has no direct
# Olist source. Carried alongside MappingReport.notes so a caller that skips
# the docstring still gets the list.
MAPPING_NOTES = {
    "orders.grain": (
        "Real Olist orders often contain more than one product; KyuYaar's canonical "
        "orders.csv is one row per order (order_id is a de-facto unique key in "
        "validation.py). The adapter aggregates each order to one canonical row: "
        "quantity is the number of items, revenue is their total, and "
        "product_id/category comes from whichever item earned the order the most "
        "revenue. segment_breakdown and price_effect on category therefore reflect "
        "each multi-item order's single dominant product, not every item it "
        "contained -- a real simplification the synthetic single-item orders never "
        "had to make."
    ),
    "orders.channel": (
        "Olist records no order-level marketing channel -- there is nothing in "
        "olist_orders_dataset.csv or olist_order_items_dataset.csv that says how a "
        "customer reached the store. Every canonical order row is filled with the "
        "constant 'unknown'. marketing_channel_analysis attributes ORDERS to a "
        "channel via this column, so on Olist that test has nothing to attribute "
        "and will not find channel-level evidence -- that is the adapter being "
        "honest about a real gap, not a bug."
    ),
    "products.cost": (
        "Olist has no cost/COGS field anywhere -- only what the customer paid. "
        "Every canonical product row is filled with the constant 0.0. This is read "
        "only by the Layer 5 scenario engine (gross-margin projections), which Layer "
        "9 does not validate; baseline_trend, segment_breakdown, "
        "aov_volume_decomposition, marketing_effect, marketing_channel_analysis and "
        "price_effect never read this column."
    ),
    "marketing.spend": (
        "Marketing Funnel by Olist has no monetary spend field -- it tracks leads "
        "and closed deals for sellers joining Olist, not ad spend. Every canonical "
        "marketing row is filled with the constant 0.0. marketing_effect and "
        "marketing_channel_analysis will correctly report no support for a "
        "spend-linked cause here; that is the honest result of missing data, not a "
        "threshold to relax."
    ),
    "marketing.region_coverage": (
        "Region can only be assigned to leads that became a closed deal (a seller "
        "record with a state), which is a small fraction of all leads in the "
        "funnel -- see MappingReport.notes for the exact share on this run. Leads "
        "that never closed have no seller and are dropped rather than assigned a "
        "guessed region."
    ),
    "customers.segment": (
        "Olist has no customer segment/tier field at all -- no repeat-buyer or VIP "
        "label anywhere in the raw data. Every canonical customer row is filled "
        "with the constant 'Unknown', the same placeholder validation.py's own "
        "upload path uses when a real person's file omits this optional column."
    ),
    "customers.customer_id": (
        "Olist's own customer_id is per-order, not per-person -- a returning buyer "
        "gets a new customer_id each order. Canonical customer_id uses "
        "customer_unique_id instead, Olist's stable per-person identifier, so "
        "repeat-customer segments mean what they should."
    ),
    "customers.region_granularity": (
        "Region is customer_state (27 values) bucketed into Brazil's 5 official "
        "macro-regions, not used as-is. See STATE_TO_MACROREGION for the mapping "
        "and the module docstring for why."
    ),
    "products.category": (
        "product_category_name is in Portuguese with a small number of rows that "
        "have no entry in product_category_name_translation.csv or no category at "
        "all. Translated rows use the English name; untranslated rows keep the "
        "Portuguese name rather than being dropped; rows with no category at all "
        "become 'unknown'."
    ),
}


@dataclass
class MappingReport:
    """What the adapter actually did on one run, for the Layer 9 write-up."""

    notes: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def add(self, note):
        self.notes.append(note)


# ---------------------------------------------------------------- reading -----

def read_raw(data_dir, require_funnel=True):
    """Read the raw Olist CSVs from `data_dir`. Returns a dict keyed by the
    names in CORE_FILES/FUNNEL_FILES. Raises FileNotFoundError, naming the
    exact missing file, for anything in REQUIRED_CORE (and the funnel files
    too when require_funnel is True) -- there is no sensible partial mapping
    without them, so this fails loudly rather than adapting a half-empty table."""
    data_dir = Path(data_dir)
    names = dict(CORE_FILES)
    if require_funnel:
        names.update(FUNNEL_FILES)

    tables = {}
    missing = []
    for key, filename in names.items():
        path = data_dir / filename
        if not path.exists():
            required = key in REQUIRED_CORE or (require_funnel and key in FUNNEL_FILES)
            if required:
                missing.append(filename)
            tables[key] = None
            continue
        tables[key] = pd.read_csv(path, dtype=str)

    if missing:
        raise FileNotFoundError(
            f"Missing Olist file(s) in {data_dir}: {', '.join(missing)}. "
            f"Download both 'Brazilian E-Commerce Public Dataset by Olist' and "
            f"'Marketing Funnel by Olist' from Kaggle and place all CSVs in this folder."
        )
    return tables


# ---------------------------------------------------------------- orders -----

def adapt_orders(orders_raw, items_raw, customers_raw, report=None):
    """order_items is Olist's unit grain: one row per item, no quantity field,
    so a line's quantity is 1 -- not a placeholder, that is genuinely what the
    raw data records. A line's revenue is price + freight_value, matching
    Kaggle's own convention for "amount paid" on a line.

    KyuYaar's canonical orders.csv is one row per order (validation.py treats
    order_id as a de-facto unique key), but a real Olist order often has
    several different products in it -- there is no honest way to keep both
    "one row per order" and "one product_id per row" at once. This adapter
    keeps the canonical grain and aggregates: quantity is the number of items
    in the order, revenue is their total, and product_id/category is taken
    from whichever item earned the order the most revenue (the order's
    "dominant" item). segment_breakdown and price_effect on category will
    then reflect that dominant item, not every item a multi-item order
    contained -- see MAPPING_NOTES['orders.grain'] and the note this run adds
    with the actual multi-item share.

    Only orders with status 'delivered' are kept: canceled/unavailable orders
    have no real revenue to attribute, and Olist's own order_items table often
    still carries a price for them, which would double-count. This mirrors
    the usual treatment of this dataset in published analyses, not a Layer-9
    special case.

    customer_id is customer_unique_id (see MAPPING_NOTES['customers.customer_id'])."""
    report = report if report is not None else MappingReport()

    orders = orders_raw[["order_id", "customer_id", "order_status", "order_purchase_timestamp"]].copy()
    before = len(orders)
    orders = orders[orders["order_status"] == "delivered"].drop(columns="order_status")
    report.add(f"orders: kept {len(orders)}/{before} rows with order_status == 'delivered'.")

    id_map = customers_raw[["customer_id", "customer_unique_id"]].drop_duplicates()
    orders = orders.merge(id_map, on="customer_id", how="left")
    unmapped = orders["customer_unique_id"].isna().sum()
    if unmapped:
        report.add(f"orders: {unmapped} order(s) had no matching row in olist_customers_dataset.csv; dropped.")
        orders = orders.dropna(subset=["customer_unique_id"])

    items = items_raw[["order_id", "product_id", "price", "freight_value"]].copy()
    for col in ("price", "freight_value"):
        items[col] = pd.to_numeric(items[col], errors="coerce")
    before_items = len(items)
    items = items.dropna(subset=["price", "freight_value"])
    if len(items) < before_items:
        report.add(f"orders: {before_items - len(items)} order_items row(s) had a non-numeric price or "
                    f"freight_value; dropped.")

    merged = items.merge(orders, on="order_id", how="inner")
    dropped_no_order = before_items - len(merged) - (before_items - len(items))
    if dropped_no_order > 0:
        report.add(f"orders: {dropped_no_order} order_items row(s) belonged to a non-delivered or "
                    f"unmatched order; dropped.")

    merged["line_revenue"] = (merged["price"] + merged["freight_value"]).round(2)
    n_orders = merged["order_id"].nunique()
    multi_item = merged.groupby("order_id").size().gt(1).sum()
    if multi_item:
        report.add(f"orders: {multi_item}/{n_orders} orders ({multi_item / n_orders:.0%}) contained more "
                    f"than one item; aggregated to one canonical row each (see MAPPING_NOTES['orders.grain']).")
    report.add(MAPPING_NOTES["orders.grain"])

    agg = merged.sort_values("line_revenue", ascending=False).groupby("order_id", as_index=False).agg(
        customer_id=("customer_unique_id", "first"),
        product_id=("product_id", "first"),          # first row after the sort above = highest-revenue item
        order_date=("order_purchase_timestamp", "first"),
        quantity=("product_id", "size"),
        revenue=("line_revenue", "sum"),
    )

    canonical = pd.DataFrame({
        "order_id": agg["order_id"],
        "customer_id": agg["customer_id"],
        "product_id": agg["product_id"],
        "order_date": pd.to_datetime(agg["order_date"]).dt.floor("D"),
        "quantity": agg["quantity"],
        "revenue": agg["revenue"].round(2),
        "channel": "unknown",
    })
    report.add(MAPPING_NOTES["orders.channel"])
    report.counts["orders_rows"] = len(canonical)
    return canonical, report


# --------------------------------------------------------------- products ---

def adapt_products(products_raw, translation_raw=None, report=None):
    report = report if report is not None else MappingReport()

    products = products_raw[["product_id", "product_category_name"]].copy()
    if translation_raw is not None:
        lookup = dict(zip(translation_raw["product_category_name"], translation_raw["product_category_name_english"]))
    else:
        lookup = {}
        report.add("products: product_category_name_translation.csv not supplied; categories kept in Portuguese.")

    def translate(name):
        if pd.isna(name) or not str(name).strip():
            return "unknown"
        return lookup.get(name, name)

    category = products["product_category_name"].map(translate)
    no_category = (category == "unknown").sum()
    if no_category:
        report.add(f"products: {no_category} product(s) had no category at all; labeled 'unknown'.")
    untranslated = int(((category == products["product_category_name"]) & (category != "unknown")).sum())
    if translation_raw is not None and untranslated:
        report.add(f"products: {untranslated} product(s) had a category not present in the translation "
                    f"table; kept in Portuguese.")

    canonical = pd.DataFrame({
        "product_id": products["product_id"],
        "category": category,
        "cost": 0.0,
    })
    report.add(MAPPING_NOTES["products.cost"])
    report.add(MAPPING_NOTES["products.category"])
    report.counts["products_rows"] = len(canonical)
    return canonical, report


# -------------------------------------------------------------- customers ---

def adapt_customers(customers_raw, report=None):
    report = report if report is not None else MappingReport()

    customers = customers_raw[["customer_unique_id", "customer_state"]].drop_duplicates(subset="customer_unique_id")
    region = customers["customer_state"].map(STATE_TO_MACROREGION)
    unknown_state = region.isna().sum()
    if unknown_state:
        report.add(f"customers: {unknown_state} customer(s) had a state not in STATE_TO_MACROREGION; "
                    f"labeled 'unknown'.")
        region = region.fillna("unknown")

    canonical = pd.DataFrame({
        "customer_id": customers["customer_unique_id"],
        "region": region,
        "segment": "Unknown",
    })
    report.add(MAPPING_NOTES["customers.customer_id"])
    report.add(MAPPING_NOTES["customers.region_granularity"])
    report.add(MAPPING_NOTES["customers.segment"])
    report.counts["customers_rows"] = len(canonical)
    return canonical, report


# -------------------------------------------------------------- marketing ---

def propose_channel_mapping(origins, client=None, model=None):
    """Bucket Olist's free-text `origin` values into CHANNEL_VOCAB.

    The mapping is genuinely ambiguous -- "social" and "display" are plainly
    paid acquisition, but whether "direct_traffic" is closer to organic or to
    its own bucket is a judgment call, and Kaggle's own column glossary does
    not settle it. Rather than have the adapter guess silently, an LLM may
    propose the bucket for each value; the adapter applies whatever mapping
    comes back (default or proposed) as a plain dict lookup -- no numbers pass
    through the model, only these text labels.

    With no client (the default, and what every automated run and test in
    this repo uses -- matching orchestrator.py's offline-first convention),
    or if the call fails for any reason, DEFAULT_CHANNEL_MAP is used and the
    fallback is recorded, not raised."""
    if client is None:
        return dict(DEFAULT_CHANNEL_MAP), "default mapping (no LLM client supplied)"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return dict(DEFAULT_CHANNEL_MAP), "default mapping (ANTHROPIC_API_KEY not set)"

    prompt = (
        f"Bucket each of these marketing lead-origin values into exactly one of "
        f"{list(CHANNEL_VOCAB)}: {sorted(set(origins))}. "
        f"Reply with ONLY a JSON object mapping each input value to one bucket name, "
        f"nothing else."
    )
    try:
        body = json.dumps({
            "model": model or "claude-sonnet-5",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body, method="POST",
            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read())
        text = "".join(block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text")
        proposed = json.loads(text)
        if not isinstance(proposed, dict) or not all(v in CHANNEL_VOCAB for v in proposed.values()):
            raise ValueError("model returned a mapping outside CHANNEL_VOCAB")
        merged = dict(DEFAULT_CHANNEL_MAP)
        merged.update(proposed)
        return merged, "LLM-proposed mapping, applied deterministically"
    except (urllib.error.URLError, ValueError, json.JSONDecodeError, KeyError, TimeoutError) as exc:
        return dict(DEFAULT_CHANNEL_MAP), f"default mapping (LLM proposal failed: {exc})"


def adapt_marketing(leads_raw, deals_raw, sellers_raw, report=None, channel_client=None):
    """Region can only be assigned via a closed deal's seller (see
    MAPPING_NOTES['marketing.region_coverage']); leads that never closed are
    dropped rather than guessed at. spend is a documented 0.0 placeholder --
    see MAPPING_NOTES['marketing.spend']."""
    report = report if report is not None else MappingReport()

    leads = leads_raw[["mql_id", "first_contact_date", "origin"]].copy()
    deals = deals_raw[["mql_id", "seller_id"]].copy()
    sellers = sellers_raw[["seller_id", "seller_state"]].drop_duplicates()

    matched = leads.merge(deals, on="mql_id", how="inner").merge(sellers, on="seller_id", how="left")
    coverage = len(matched) / len(leads) if len(leads) else 0.0
    report.add(f"marketing: {len(matched)}/{len(leads)} leads ({coverage:.0%}) closed as a deal with a "
               f"known seller and could be assigned a region; the rest are dropped.")
    report.add(MAPPING_NOTES["marketing.region_coverage"])

    region = matched["seller_state"].map(STATE_TO_MACROREGION)
    unknown_state = region.isna().sum()
    if unknown_state:
        report.add(f"marketing: {unknown_state} matched lead(s) had a seller state not in "
                    f"STATE_TO_MACROREGION; dropped.")
    matched = matched.assign(region=region).dropna(subset=["region"])

    origins = matched["origin"].dropna().unique().tolist()
    mapping, mapping_note = propose_channel_mapping(origins, client=channel_client)
    report.add(f"marketing: channel bucketing used {mapping_note}.")
    matched["channel"] = matched["origin"].map(lambda o: mapping.get(o, FALLBACK_CHANNEL))

    canonical = pd.DataFrame({
        "date": pd.to_datetime(matched["first_contact_date"]).dt.floor("D"),
        "channel": matched["channel"],
        "region": matched["region"],
        "spend": 0.0,
    })
    report.add(MAPPING_NOTES["marketing.spend"])
    report.counts["marketing_rows"] = len(canonical)
    return canonical, report


# -------------------------------------------------------------------- all ---

def adapt_all(data_dir, channel_client=None, require_funnel=True):
    """Read every raw Olist table under `data_dir` and return
    (tables, report) where `tables` is a dict with the same keys as
    validation.TABLES (orders/products/customers/marketing), each a canonical
    DataFrame, and `report` is a MappingReport covering the whole run."""
    raw = read_raw(data_dir, require_funnel=require_funnel)
    report = MappingReport()

    products, report = adapt_products(raw["products"], raw.get("category_translation"), report)
    customers, report = adapt_customers(raw["customers"], report)
    orders, report = adapt_orders(raw["orders"], raw["order_items"], raw["customers"], report)

    if require_funnel and raw.get("leads") is not None:
        marketing, report = adapt_marketing(raw["leads"], raw["deals"], raw["sellers"], report,
                                             channel_client=channel_client)
    else:
        marketing = pd.DataFrame(columns=["date", "channel", "region", "spend"])
        report.add("marketing: funnel files not supplied; marketing.csv is empty.")

    return {"orders": orders, "products": products, "customers": customers, "marketing": marketing}, report


def write_canonical_csvs(tables, out_dir):
    """Write the four canonical frames to out_dir as orders.csv, products.csv,
    customers.csv and marketing.csv, in the shape validation.load_upload expects."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    return out_dir
