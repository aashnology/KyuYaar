"""Layer 4: marketing by channel."""

import itertools

import pandas as pd
import pytest

from channels import channel_check, marketing_channel_analysis
from decisions import build_options
from effects import marketing_effect, price_effect
from guardrail import check_text
from narration import build_summary, label, narrate_step
from toolkit import TOOL_SPECS, Toolkit

REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Organic", "Paid", "Email"]
CATEGORIES = ["A", "B"]
BASE_SPEND = {"Paid": 1000.0, "Email": 200.0, "Organic": 0.0}


def world(base=60, order_factor=None, spend_factor=None):
    """A noise-free two-month world. Every (region, channel, category) cell has
    `base` orders in the prior month. `order_factor[(region, channel)]` scales the
    latest month's orders and `spend_factor[(region, channel)]` scales its spend."""
    order_factor = order_factor or {}
    spend_factor = spend_factor or {}
    rows, spend_rows = [], []
    for region, channel, category in itertools.product(REGIONS, CHANNELS, CATEGORIES):
        last_n = round(base * order_factor.get((region, channel), 1.0))
        rows += [("2026-01-15", region, category, channel, 10.0)] * base
        rows += [("2026-02-15", region, category, channel, 10.0)] * last_n
    for region, channel in itertools.product(REGIONS, CHANNELS):
        s = BASE_SPEND[channel]
        spend_rows.append(("2026-01-15", channel, region, s, 1000))
        spend_rows.append(("2026-02-15", channel, region, s * spend_factor.get((region, channel), 1.0), 1000))
    orders = pd.DataFrame(rows, columns=["order_date", "region", "category", "channel", "revenue"])
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    marketing = pd.DataFrame(spend_rows, columns=["date", "channel", "region", "spend", "impressions"])
    marketing["date"] = pd.to_datetime(marketing["date"])
    return orders, marketing


def cell(evs, region, channel):
    return next(e for e in evs if e.id == f"chan_{region}_{channel}")


PAID_CUT = {("North", "Paid"): 0.55}


# ------------------------------------------------------------ specificity

def test_loss_confined_to_the_cut_channel_is_reported_as_concentrated():
    orders, marketing = world(order_factor={("North", "Paid"): 0.5}, spend_factor=PAID_CUT)
    evs = marketing_channel_analysis(orders, marketing)
    north = cell(evs, "North", "Paid")
    assert north.strength == "strong"
    assert north.details["channel_pattern"]["status"] == "concentrated"
    assert north.details["channel_pattern"]["other_channels"] == ["Email", "Organic"]


def test_loss_shared_by_every_channel_is_reported_as_region_wide():
    orders, marketing = world(
        order_factor={("North", c): 0.5 for c in CHANNELS}, spend_factor=PAID_CUT,
    )
    north = cell(marketing_channel_analysis(orders, marketing), "North", "Paid")
    assert north.strength == "strong"                       # Paid orders did fall against comparison regions
    assert north.details["channel_pattern"]["status"] == "region_wide"
    assert "region-wide" in north.details["channel_pattern"]["text"]


def test_a_cell_with_no_clear_contrast_is_unclear_not_region_wide():
    # Small counts: Paid orders fall, the other channels stay flat, but there is too
    # little data for the gap to be significant, and the others show no move of their own.
    orders, marketing = world(base=3, order_factor={("North", "Paid"): 0.34}, spend_factor=PAID_CUT)
    north = cell(marketing_channel_analysis(orders, marketing), "North", "Paid")
    assert north.details["channel_pattern"]["status"] == "unclear"


# ------------------------------------------------------------- efficiency

def test_orders_falling_with_spend_leave_cost_per_order_unchanged():
    orders, marketing = world(order_factor={("North", "Paid"): 0.55}, spend_factor=PAID_CUT)
    eff = cell(marketing_channel_analysis(orders, marketing), "North", "Paid").details["efficiency"]
    assert eff["verdict"] == "unchanged"
    assert eff["cost_per_order_change_pct"] == pytest.approx(0.0, abs=1.0)
    assert "roughly in proportion" in eff["text"]


def test_orders_falling_faster_than_spend_is_reported_as_a_less_effective_channel():
    orders, marketing = world(base=100, order_factor={("North", "Paid"): 0.3}, spend_factor=PAID_CUT)
    eff = cell(marketing_channel_analysis(orders, marketing), "North", "Paid").details["efficiency"]
    assert eff["verdict"] == "worse"
    assert eff["cost_per_order_change_pct"] > 50
    assert "less effective" in eff["text"]


# ----------------------------------------------------- gates and guardrails

def test_a_channel_whose_spend_did_not_move_is_never_a_candidate():
    # Paid orders in North halve, but nobody touched Paid spend.
    orders, marketing = world(order_factor={("North", "Paid"): 0.5})
    north = cell(marketing_channel_analysis(orders, marketing), "North", "Paid")
    assert north.strength == "weak"
    assert north.details["driver_moved"] is False
    assert north.details["channel_pattern"] is None and north.details["efficiency"] is None
    assert "not a candidate" in north.caveats[0]


def test_orders_moving_the_wrong_way_do_not_support_a_spend_cut():
    orders, marketing = world(order_factor={("North", "Paid"): 1.5}, spend_factor=PAID_CUT)
    north = cell(marketing_channel_analysis(orders, marketing), "North", "Paid")
    assert north.strength == "weak"
    assert any("opposite direction" in c for c in north.caveats)


def test_no_comparison_region_means_weak_with_an_explanation():
    # Two regions cut Paid hard and two did not, so every region sits 25 points from typical.
    orders, marketing = world(spend_factor={("North", "Paid"): 0.5, ("South", "Paid"): 0.5})
    evs = marketing_channel_analysis(orders, marketing)
    north = cell(evs, "North", "Paid")
    assert north.strength == "weak" and north.value is None
    assert any("no comparison group" in c for c in north.caveats)


def test_small_cells_are_downgraded():
    orders, marketing = world(base=3, order_factor={("North", "Paid"): 0.0}, spend_factor=PAID_CUT)
    north = cell(marketing_channel_analysis(orders, marketing), "North", "Paid")
    assert north.sample_size < 30
    assert north.strength != "strong"
    assert any("below the 30-order minimum" in c for c in north.caveats)


def test_every_paid_channel_and_region_gets_one_evidence_object():
    orders, marketing = world()
    evs = marketing_channel_analysis(orders, marketing)
    assert len(evs) == len(REGIONS) * 2                      # Paid and Email; Organic has no spend
    assert all(e.evidence_type == "channel" for e in evs)
    assert len({e.id for e in evs}) == len(evs)
    assert all(e.strength == "weak" for e in evs)            # nothing changed anywhere


def test_no_paid_spend_gives_no_evidence():
    orders, marketing = world()
    assert marketing_channel_analysis(orders, marketing.assign(spend=0.0)) == []


def test_orders_without_a_channel_column_are_rejected():
    orders, marketing = world()
    with pytest.raises(ValueError, match="channel"):
        marketing_channel_analysis(orders.drop(columns="channel"), marketing)


# ----------------------------------------------------------- the real data

@pytest.fixture(scope="module")
def cells(orders, marketing):
    return marketing_channel_analysis(orders, marketing)


def test_only_north_paid_is_supported_and_it_is_region_wide(cells):
    supported = [e for e in cells if e.strength != "weak"]
    assert [e.id for e in supported] == ["chan_North_Paid"]
    north = supported[0]
    assert north.details["driver_change_pct"] == pytest.approx(-44.9, abs=0.1)
    assert north.details["channel_pattern"]["status"] == "region_wide"
    assert north.details["efficiency"]["verdict"] == "unchanged"


def test_the_spend_gate_is_what_keeps_chance_cells_out(cells):
    # Some cells whose spend did not move still show a significant order change by chance,
    # because orders are compared cell by cell. None of them may be flagged.
    quiet = [e for e in cells if not e.details["driver_moved"]]
    assert any(e.details["p_value"] < 0.05 for e in quiet)
    assert all(e.strength == "weak" for e in quiet)


def test_tool_is_registered_and_stamps_provenance(toolkit):
    assert "marketing_channel_analysis" in {s["name"] for s in TOOL_SPECS}
    evs = toolkit.run("marketing_channel_analysis")
    prov = evs[0].details["provenance"]
    assert prov == {"tool": "marketing_channel_analysis", "args": {}, "source": "src/channels.py"}


def test_channel_labels_read_naturally(cells):
    assert label(cell(cells, "North", "Paid")) == "Paid in North"


# -------------------------------------------------------- decisions and text

@pytest.fixture(scope="module")
def evidence(orders, marketing, cells):
    return marketing_effect(orders, marketing) + price_effect(orders) + cells


def test_channel_check_finds_the_moved_channel_for_a_regional_cause(evidence):
    north = next(e for e in evidence if e.id == "stat_marketing_North")
    chk = channel_check(north, evidence)
    assert chk["channel"] == "Paid" and chk["status"] == "region_wide"
    assert channel_check(next(e for e in evidence if e.id == "stat_marketing_East"), evidence) is None
    assert channel_check(north, [e for e in evidence if e.evidence_type != "channel"]) is None


def test_marketing_option_states_the_checked_channel_assumptions(evidence):
    ds = build_options(evidence)
    act = next(o for o in ds.options if o.id == "restore_marketing_North")
    joined = " ".join(act.assumptions)
    assert "Checked against the data" in joined and "region-wide" in joined
    assert "roughly in proportion" in joined                 # efficiency turned a stated assumption into a checked one
    assert any("Orders fell in every channel" in r for r in act.risks)
    assert any("Paid spend drives orders in the other channels" in u for u in ds.unresolved)
    assert any(x.startswith("Marketing by channel:") for x in ds.not_supported)


def test_channel_evidence_is_not_double_counted_as_a_cause(evidence):
    with_channels = build_options(evidence)
    without = build_options([e for e in evidence if e.evidence_type != "channel"])
    assert [o.id for o in with_channels.options] == [o.id for o in without.options]
    assert with_channels.options[-1].impact == without.options[-1].impact


def test_options_without_the_channel_tool_are_unchanged(evidence):
    ds = build_options([e for e in evidence if e.evidence_type != "channel"])
    act = next(o for o in ds.options if o.id == "restore_marketing_North")
    assert "region-wide" not in " ".join(act.assumptions)
    assert not any(x.startswith("Marketing by channel") for x in ds.not_supported)


def test_channel_narration_and_summary_use_only_figures_from_the_evidence(cells, evidence):
    step = narrate_step("marketing_channel_analysis", {}, cells)
    assert "region-wide" in step and "Paid" in step
    assert check_text(step, cells).ok
    summary = build_summary(evidence)
    assert "Which channel?" in summary
    assert check_text(summary, evidence).ok


def test_narration_when_no_channel_is_supported(orders, marketing):
    quiet = marketing_channel_analysis(*world())
    assert "not supported as an explanation" in narrate_step("marketing_channel_analysis", {}, quiet)
