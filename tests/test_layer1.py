from evidence import Evidence, baseline_trend, segment_breakdown, MIN_SAMPLE_SIZE


def test_baseline_trend_flags_the_august_drop(orders):
    ev = baseline_trend(orders)
    assert ev.evidence_type == "observation"
    assert ev.strength == "strong"
    assert -26 < ev.value < -22
    assert ev.details["latest_period"] == "2026-08"


def test_region_breakdown_recovers_north(orders):
    ev = segment_breakdown(orders, "region")
    assert ev[0].segment == "region=North"
    assert ev[0].strength == "strong"
    assert all(e.strength == "weak" for e in ev[1:])


def test_category_breakdown_recovers_electronics(orders):
    ev = segment_breakdown(orders, "category")
    assert ev[0].segment == "category=Electronics"
    assert ev[0].strength in ("moderate", "strong")
    assert all(e.strength == "weak" for e in ev[1:])


def test_shares_are_not_ranked_when_the_total_change_is_tiny(orders):
    # Through 2026-05 the latest month moved only a few percent.
    early = orders[orders["order_date"] < "2026-06-01"]
    for dim in ("region", "category"):
        ev = segment_breakdown(early, dim)
        assert all(e.strength == "weak" for e in ev)
        assert any("unstable" in c for c in ev[0].caveats)


def test_small_sample_downgrades_strength():
    from evidence import _downgrade_if_small_sample

    strength, caveats = _downgrade_if_small_sample("strong", MIN_SAMPLE_SIZE - 1)
    assert strength == "moderate" and caveats
    strength, caveats = _downgrade_if_small_sample("weak", 5)
    assert strength == "weak"                       # cannot go below weak
    strength, caveats = _downgrade_if_small_sample("strong", MIN_SAMPLE_SIZE)
    assert strength == "strong" and not caveats
