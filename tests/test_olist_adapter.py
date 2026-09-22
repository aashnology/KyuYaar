"""
Olist adapter -- Layer 9.

These fixtures are small hand-built frames shaped like Olist's real raw
columns, not the 100k-row Kaggle dataset itself (that is exercised
separately by scripts/verify_layer9.py once the CSVs are downloaded). The
point here is the mapping logic: does each canonical column come out with
the right values, types and placeholders, on inputs small enough to reason
about by eye.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters.olist import (  # noqa: E402
    CHANNEL_VOCAB, DEFAULT_CHANNEL_MAP, STATE_TO_MACROREGION,
    adapt_customers, adapt_marketing, adapt_orders, adapt_products,
    propose_channel_mapping,
)
from validation import TABLES, load_upload  # noqa: E402


def _orders_raw():
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_id": ["c1", "c2", "c3"],
        "order_status": ["delivered", "delivered", "canceled"],
        "order_purchase_timestamp": ["2018-01-05 10:00:00", "2018-02-10 12:00:00", "2018-02-11 09:00:00"],
    })


def _items_raw():
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "product_id": ["p1", "p2", "p3"],
        "price": ["100.00", "50.00", "999.00"],
        "freight_value": ["10.00", "5.00", "1.00"],
    })


def _customers_raw():
    return pd.DataFrame({
        "customer_id": ["c1", "c2", "c3"],
        "customer_unique_id": ["u1", "u2", "u3"],
        "customer_state": ["SP", "PR", "SP"],
    })


# --------------------------------------------------------------- orders ----

def test_orders_quantity_is_always_one():
    canonical, _ = adapt_orders(_orders_raw(), _items_raw(), _customers_raw())
    assert (canonical["quantity"] == 1).all()


def test_orders_revenue_is_price_plus_freight():
    canonical, _ = adapt_orders(_orders_raw(), _items_raw(), _customers_raw())
    row = canonical[canonical["product_id"] == "p1"].iloc[0]
    assert row["revenue"] == pytest.approx(110.00)


def test_orders_channel_is_the_documented_placeholder():
    canonical, report = adapt_orders(_orders_raw(), _items_raw(), _customers_raw())
    assert (canonical["channel"] == "unknown").all()
    assert any("no order-level marketing channel" in n for n in report.notes)


def test_orders_drops_non_delivered():
    canonical, report = adapt_orders(_orders_raw(), _items_raw(), _customers_raw())
    assert "p3" not in canonical["product_id"].tolist()
    assert len(canonical) == 2
    assert any("delivered" in n for n in report.notes)


def test_orders_customer_id_is_the_stable_person_id_not_the_per_order_one():
    canonical, _ = adapt_orders(_orders_raw(), _items_raw(), _customers_raw())
    row = canonical[canonical["product_id"] == "p1"].iloc[0]
    assert row["customer_id"] == "u1"          # customer_unique_id, not "c1"


def test_orders_multi_item_order_is_aggregated_to_one_canonical_row():
    orders_raw = pd.DataFrame({
        "order_id": ["o1"], "customer_id": ["c1"], "order_status": ["delivered"],
        "order_purchase_timestamp": ["2018-01-05 10:00:00"],
    })
    items_raw = pd.DataFrame({
        "order_id": ["o1", "o1"], "product_id": ["p_cheap", "p_pricey"],
        "price": ["10.00", "90.00"], "freight_value": ["1.00", "9.00"],
    })
    customers_raw = pd.DataFrame({"customer_id": ["c1"], "customer_unique_id": ["u1"], "customer_state": ["SP"]})

    canonical, report = adapt_orders(orders_raw, items_raw, customers_raw)
    assert len(canonical) == 1                                    # one order in, one canonical row out
    row = canonical.iloc[0]
    assert row["quantity"] == 2                                   # two items
    assert row["revenue"] == pytest.approx(110.00)                # (10+1) + (90+9)
    assert row["product_id"] == "p_pricey"                        # the higher-revenue item, not the first row
    assert any("contained more than one item" in n for n in report.notes)


# ------------------------------------------------------------- products ----

def test_products_category_is_translated_when_a_translation_exists():
    products_raw = pd.DataFrame({"product_id": ["p1", "p2"],
                                  "product_category_name": ["cama_mesa_banho", "sem_categoria"]})
    translation = pd.DataFrame({"product_category_name": ["cama_mesa_banho"],
                                 "product_category_name_english": ["bed_bath_table"]})
    canonical, report = adapt_products(products_raw, translation)
    got = dict(zip(canonical["product_id"], canonical["category"]))
    assert got["p1"] == "bed_bath_table"
    assert got["p2"] == "sem_categoria"        # no translation entry -> kept as-is, not dropped
    assert any("not present in the translation table" in n for n in report.notes)


def test_products_blank_category_becomes_unknown():
    products_raw = pd.DataFrame({"product_id": ["p1"], "product_category_name": [None]})
    canonical, report = adapt_products(products_raw)
    assert canonical.iloc[0]["category"] == "unknown"
    assert any("no category at all" in n for n in report.notes)


def test_products_cost_is_the_documented_zero_placeholder():
    products_raw = pd.DataFrame({"product_id": ["p1"], "product_category_name": ["x"]})
    canonical, report = adapt_products(products_raw)
    assert canonical.iloc[0]["cost"] == 0.0
    assert any("no cost/COGS field" in n for n in report.notes)


# ------------------------------------------------------------ customers ----

def test_customers_region_is_bucketed_into_macroregions():
    canonical, _ = adapt_customers(_customers_raw())
    got = dict(zip(canonical["customer_id"], canonical["region"]))
    assert got["u1"] == STATE_TO_MACROREGION["SP"] == "Sudeste"
    assert got["u2"] == STATE_TO_MACROREGION["PR"] == "Sul"


def test_customers_unknown_state_does_not_crash_and_is_labeled():
    raw = pd.DataFrame({"customer_id": ["c1"], "customer_unique_id": ["u1"], "customer_state": ["ZZ"]})
    canonical, report = adapt_customers(raw)
    assert canonical.iloc[0]["region"] == "unknown"
    assert any("not in STATE_TO_MACROREGION" in n for n in report.notes)


def test_customers_segment_is_the_documented_unknown_placeholder():
    canonical, report = adapt_customers(_customers_raw())
    assert (canonical["segment"] == "Unknown").all()
    assert any("no customer segment/tier field" in n for n in report.notes)


# ------------------------------------------------------------ marketing ----

def _leads_raw():
    return pd.DataFrame({
        "mql_id": ["m1", "m2", "m3"],
        "first_contact_date": ["2018-01-01", "2018-01-02", "2018-01-03"],
        "origin": ["organic_search", "paid_search", "organic_search"],
    })


def _deals_raw():
    return pd.DataFrame({"mql_id": ["m1", "m2"], "seller_id": ["s1", "s2"]})   # m3 never closed


def _sellers_raw():
    return pd.DataFrame({"seller_id": ["s1", "s2"], "seller_state": ["SP", "RS"]})


def test_marketing_drops_leads_that_never_closed():
    canonical, report = adapt_marketing(_leads_raw(), _deals_raw(), _sellers_raw())
    assert len(canonical) == 2                 # m3 dropped, not guessed at
    assert any("2/3 leads" in n for n in report.notes)


def test_marketing_spend_is_the_documented_zero_placeholder():
    canonical, report = adapt_marketing(_leads_raw(), _deals_raw(), _sellers_raw())
    assert (canonical["spend"] == 0.0).all()
    assert any("no monetary spend field" in n for n in report.notes)


def test_marketing_channel_uses_default_mapping_with_no_client():
    canonical, report = adapt_marketing(_leads_raw(), _deals_raw(), _sellers_raw())
    assert set(canonical["channel"]).issubset(set(CHANNEL_VOCAB))
    assert canonical[canonical["region"] == "Sudeste"].iloc[0]["channel"] == DEFAULT_CHANNEL_MAP["organic_search"]
    assert any("default mapping" in n for n in report.notes)


def test_propose_channel_mapping_falls_back_without_a_client():
    mapping, note = propose_channel_mapping(["organic_search", "paid_search"], client=None)
    assert mapping == DEFAULT_CHANNEL_MAP
    assert "default mapping" in note


# --------------------------------------------------------- schema contract --

def test_adapted_tables_satisfy_the_canonical_schema():
    """Same test style as test_validation.py: feed the adapter's output through
    the real validator rather than re-checking column names by hand, so a
    change to SCHEMA is caught here too. A small fixture like this is well
    under validation's sufficiency bar (months, regions, categories), so the
    contract check runs with smoke_test off, same as test_validation.py's own
    `run()` helper does for hand-built frames."""
    orders_raw = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(4)],
        "customer_id": [f"c{i}" for i in range(4)],
        "order_status": ["delivered"] * 4,
        "order_purchase_timestamp": ["2018-01-05", "2018-02-05", "2018-03-05", "2018-04-05"],
    })
    items_raw = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(4)],
        "product_id": ["p1", "p2", "p1", "p2"],
        "price": ["100.00"] * 4,
        "freight_value": ["10.00"] * 4,
    })
    customers_raw = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(4)],
        "customer_unique_id": [f"u{i}" for i in range(4)],
        "customer_state": ["SP", "PR", "SP", "PR"],
    })
    products_raw = pd.DataFrame({"product_id": ["p1", "p2"], "product_category_name": ["cat_a", "cat_b"]})
    leads_raw = pd.DataFrame({"mql_id": ["m1"], "first_contact_date": ["2018-04-01"], "origin": ["organic_search"]})
    deals_raw = pd.DataFrame({"mql_id": ["m1"], "seller_id": ["s1"]})
    sellers_raw = pd.DataFrame({"seller_id": ["s1"], "seller_state": ["SP"]})

    orders, report = adapt_orders(orders_raw, items_raw, customers_raw)
    products, report = adapt_products(products_raw, report=report)
    customers, report = adapt_customers(customers_raw, report=report)
    marketing, report = adapt_marketing(leads_raw, deals_raw, sellers_raw, report=report)

    sources = {name: frame.to_csv(index=False).encode() for name, frame in
               {"orders": orders, "products": products, "customers": customers, "marketing": marketing}.items()}
    assert set(sources) == set(TABLES)
    result = load_upload(sources, smoke_test=False)
    assert result.ok, result.errors
