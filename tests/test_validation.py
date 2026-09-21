"""Bring-your-own-CSV validation: clear errors, counted warnings, no silent changes."""

import io
from pathlib import Path

import pandas as pd
import pytest

from data_loader import DEFAULT_DATA_DIR, load_data
from validation import TABLES, DataValidationError, load_upload, load_validated

RAW = {t: pd.read_csv(DEFAULT_DATA_DIR / f"{t}.csv", dtype=str) for t in TABLES}


def csv(frame):
    return frame.to_csv(index=False).encode()


def sources(**overrides):
    """Bytes for the four tables, with any table replaced by a frame, bytes or None."""
    out = {}
    for t in TABLES:
        v = overrides.get(t, RAW[t])
        out[t] = v if v is None or isinstance(v, bytes) else csv(v)
    return out


def run(**overrides):
    return load_upload(sources(**overrides), smoke_test=False)


def test_the_committed_dataset_passes_untouched():
    result = load_upload({t: DEFAULT_DATA_DIR / f"{t}.csv" for t in TABLES})
    assert result.ok and not result.errors and not result.warnings
    orders, marketing = result.data
    ref_orders, ref_marketing = load_data()
    assert len(orders) == len(ref_orders)
    assert orders["revenue"].sum() == pytest.approx(ref_orders["revenue"].sum())
    assert marketing["spend"].sum() == pytest.approx(ref_marketing["spend"].sum())
    assert result.summary["months"] == 12


def test_upload_and_folder_paths_give_the_same_investigation_inputs():
    up_orders, _ = load_upload({t: DEFAULT_DATA_DIR / f"{t}.csv" for t in TABLES}).data
    ref_orders, _ = load_data()
    for col in ["region", "category", "channel", "unit_cost"]:
        assert up_orders[col].tolist() == ref_orders[col].tolist()


def test_missing_table_is_named():
    result = run(marketing=None)
    assert not result.ok
    assert "marketing.csv" in result.errors[0] and "Missing" in result.errors[0]


def test_missing_column_error_lists_it_and_suggests_a_rename():
    orders = RAW["orders"].rename(columns={"revenue": "revenues", "order_date": "date"})
    result = run(orders=orders)
    assert not result.ok
    msg = " ".join(result.errors)
    assert "order_date" in msg and "revenue" in msg and "Columns found" in msg
    assert "found 'revenues'" in msg
    assert "found 'order_id'" not in msg          # a real column of the table is never offered as a rename


def test_headers_are_matched_loosely():
    orders = RAW["orders"].rename(columns={"order_date": " Order Date ", "revenue": "REVENUE"})
    assert run(orders=orders).ok


def test_empty_and_header_only_files_are_rejected():
    assert "empty" in run(orders=b"").errors[0]
    header_only = RAW["orders"].iloc[0:0]
    assert "no rows" in run(orders=header_only).errors[0]


def test_semicolon_separated_export_is_read():
    text = RAW["products"].to_csv(index=False, sep=";").encode()
    assert run(products=text).ok


def test_latin1_file_is_read_with_a_note():
    customers = RAW["customers"].copy()
    customers.loc[0, "region"] = "Nörth"
    raw = customers.to_csv(index=False).encode("latin-1")
    result = run(customers=raw)
    assert any("Latin-1" in w for w in result.warnings)


def test_currency_text_is_read_as_numbers_and_reported():
    orders = RAW["orders"].copy()
    orders["revenue"] = orders["revenue"].map(lambda v: f"${float(v):,.2f}")
    result = run(orders=orders)
    assert result.ok
    assert any("currency symbols" in w for w in result.warnings)
    ref, _ = load_data()
    assert result.data[0]["revenue"].sum() == pytest.approx(ref["revenue"].sum())


def test_a_few_unusable_rows_are_dropped_and_counted():
    orders = RAW["orders"].copy()
    orders.loc[0, "revenue"] = "abc"
    orders.loc[1, "quantity"] = "0"
    orders.loc[2, "order_date"] = "not a date"
    orders.loc[3, "revenue"] = "-20"
    result = run(orders=orders)
    assert result.ok
    warning = next(w for w in result.warnings if "excluded 4 of" in w)
    assert "revenue is not a number" in warning and "refunds" in warning
    assert len(result.data[0]) == len(RAW["orders"]) - 4


def test_many_unusable_rows_reject_the_file():
    orders = RAW["orders"].copy()
    orders.loc[: len(orders) // 5, "revenue"] = "oops"
    result = run(orders=orders)
    assert not result.ok
    assert "wrong file" in result.errors[0]


def test_repeated_keys_in_a_reference_table_are_an_error():
    products = pd.concat([RAW["products"], RAW["products"].iloc[:2]])
    result = run(products=products)
    assert not result.ok and "must be unique" in result.errors[0]


def test_orders_that_match_no_product_are_dropped_when_few_and_rejected_when_many():
    few = RAW["orders"].copy()
    few.loc[:4, "product_id"] = "P9999"
    result = run(orders=few)
    assert result.ok and any("do not match a product or customer" in w for w in result.warnings)

    many = RAW["orders"].copy()
    many["product_id"] = "X" + many["product_id"]
    result = run(orders=many)
    assert not result.ok and "same ids" in result.errors[0]


def test_duplicate_order_ids_are_dropped_with_a_warning():
    orders = pd.concat([RAW["orders"], RAW["orders"].iloc[:10]])
    result = run(orders=orders)
    assert result.ok and any("10 repeated order_id" in w for w in result.warnings)
    assert len(result.data[0]) == len(RAW["orders"])


def test_case_variants_are_merged_and_reported():
    customers = RAW["customers"].copy()
    customers.loc[customers.index[:5], "region"] = customers.loc[customers.index[:5], "region"].str.lower()
    result = run(customers=customers)
    assert result.ok and any("capitalisation" in w for w in result.warnings)
    assert set(result.data[0]["region"]) == {"North", "South", "East", "West"}


def test_too_little_history_is_an_error_and_short_history_a_warning():
    dates = pd.to_datetime(RAW["orders"]["order_date"])
    two_months = RAW["orders"][dates < "2025-11-01"]
    result = run(orders=two_months, marketing=RAW["marketing"][pd.to_datetime(RAW["marketing"]["date"]) < "2025-11-01"])
    assert not result.ok and "at least 3" in " ".join(result.errors)

    five = RAW["orders"][dates < "2026-02-01"]
    mk = RAW["marketing"][pd.to_datetime(RAW["marketing"]["date"]) < "2026-02-01"]
    result = run(orders=five, marketing=mk)
    assert result.ok and any("Only 5 months" in w for w in result.warnings)


def test_partial_latest_month_is_flagged_and_can_be_left_out():
    dates = pd.to_datetime(RAW["orders"]["order_date"])
    cut = RAW["orders"][dates <= "2026-08-14"]
    result = run(orders=cut)
    assert result.ok and result.partial_last_month
    assert any("before the month ends" in w for w in result.warnings)

    trimmed = load_upload(sources(orders=cut), drop_partial_last_month=True, smoke_test=False)
    assert trimmed.ok and not trimmed.partial_last_month
    assert trimmed.data[0]["order_date"].max() < pd.Timestamp("2026-08-01")
    assert trimmed.summary["months"] == 11
    assert any("Left out the partial latest month" in w for w in trimmed.warnings)


def test_marketing_that_stops_early_is_an_error():
    mk = RAW["marketing"][pd.to_datetime(RAW["marketing"]["date"]) < "2026-07-01"]
    result = run(marketing=mk)
    assert not result.ok and "marketing.csv ends" in result.errors[0]


def test_marketing_regions_that_never_match_are_an_error():
    mk = RAW["marketing"].copy()
    mk["region"] = "Zone " + mk["region"]
    result = run(marketing=mk)
    assert not result.ok and "spell them identically" in result.errors[0]


def test_a_single_region_cannot_be_compared():
    customers = RAW["customers"].copy()
    customers["region"] = "North"
    result = run(customers=customers)
    assert not result.ok and "at least 2" in " ".join(result.errors)


def test_smoke_test_turns_a_tool_failure_into_a_readable_error(monkeypatch):
    import toolkit

    def boom(self, name, args=None):
        raise RuntimeError("something odd")

    monkeypatch.setattr(toolkit.Toolkit, "run", boom)
    result = load_upload(sources())
    assert not result.ok and "could not run on this data" in result.errors[0]


def test_load_validated_raises_with_the_messages():
    with pytest.raises(DataValidationError) as exc:
        load_validated(sources(marketing=None))
    assert "marketing.csv" in str(exc.value)
    orders, marketing = load_validated(sources())
    assert len(orders) > 0 and len(marketing) > 0
