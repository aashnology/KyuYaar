import pytest

from effects import marketing_effect, price_effect


def by_id(evidence):
    return {e.id: e for e in evidence}


def test_marketing_effect_finds_north_and_only_north(orders, marketing):
    ev = by_id(marketing_effect(orders, marketing))
    north = ev["stat_marketing_North"]
    assert north.evidence_type == "statistical"
    assert north.strength == "strong"
    assert -55 < north.value < -25
    assert north.details["driver_moved"] is True
    assert north.details["driver_change_pct"] < -25
    assert north.details["top_channel"] == "Paid"
    for region in ("East", "South", "West"):
        assert ev[f"stat_marketing_{region}"].strength == "weak"
        assert ev[f"stat_marketing_{region}"].details["driver_moved"] is False


def test_untouched_regions_are_not_flagged_by_the_treated_one(orders, marketing):
    # Regression: when every other region (North included) formed the control
    # group, East looked like a spend "increase" with a moderate order effect.
    ev = by_id(marketing_effect(orders, marketing))
    assert "North" not in ev["stat_marketing_East"].details["control_segments"]
    assert ev["stat_marketing_East"].strength == "weak"


def test_price_effect_finds_electronics_and_only_electronics(orders):
    ev = by_id(price_effect(orders))
    elec = ev["stat_price_Electronics"]
    assert elec.strength == "strong"
    assert -40 < elec.value < -15
    assert 8 < elec.details["driver_change_pct"] < 12      # the injected hike is 10%
    for cat in ("Apparel", "Beauty", "Home", "Sports"):
        assert ev[f"stat_price_{cat}"].strength == "weak"
        assert ev[f"stat_price_{cat}"].details["driver_moved"] is False


def test_only_moved_drivers_carry_a_revenue_estimate(orders, marketing):
    for e in marketing_effect(orders, marketing) + price_effect(orders):
        if e.strength == "weak":
            assert e.details["revenue_at_stake"] is None
        else:
            assert e.details["revenue_at_stake"] > 0


def test_no_false_positives_in_months_without_an_injected_cause(orders, marketing):
    per = orders["order_date"].dt.to_period("M")
    months = sorted(per.unique())
    tested = 0
    for cut in months[2:-1]:                         # the last month holds the real cause
        o = orders[per <= cut]
        m = marketing[marketing["date"].dt.to_period("M") <= cut]
        for e in marketing_effect(o, m) + price_effect(o):
            tested += 1
            assert e.strength == "weak", f"{e.id} flagged {e.strength} in {cut}"
    assert tested >= 60


def test_every_result_has_a_caveat_or_states_no_candidate(orders, marketing):
    for e in marketing_effect(orders, marketing) + price_effect(orders):
        assert e.caveats
