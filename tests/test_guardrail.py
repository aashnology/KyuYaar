import pytest

from data_loader import load_data, scenario_dir
from evidence import Evidence
from guardrail import _allowed, check_text, is_signed_key
from orchestrator import investigate
from synthetic import SCENARIOS
from toolkit import Toolkit


def make(**kw):
    base = dict(
        id="x", hypothesis="North orders moved -39% (p<0.001) on 220 orders",
        evidence_type="statistical", metric="m", value=-38.6, baseline=0.0,
        segment="region=North", strength="strong", sample_size=220,
        caveats=[], details={"p_value": 1.59e-07, "p_value_text": "<0.001", "spend_prior": 32550.0},
    )
    base.update(kw)
    return Evidence(**base)


def test_rounded_figures_pass():
    ev = [make()]
    assert check_text("Orders fell 39% across 220 orders.", ev).ok
    assert check_text("Orders fell 38.6% in North.", ev).ok


def test_unsigned_figures_match_on_magnitude_and_a_wrong_magnitude_is_caught():
    ev = [make()]
    assert check_text("A decline of -39%.", ev).ok
    assert check_text("A decline of 39%.", ev).ok        # no sign written, nothing to contradict
    assert not check_text("A decline of 41%.", ev).ok


def test_invented_and_derived_numbers_are_caught():
    ev = [make()]
    res = check_text("Orders fell 39%, roughly 4x worse than the rest.", ev)
    assert not res.ok and "4" in res.unsupported
    assert not check_text("That is 250 orders.", ev).ok


def test_precision_matters():
    ev = [make()]
    assert not check_text("Orders fell 38.9%.", ev).ok


def test_thousands_separators_and_scientific_notation():
    ev = [make()]
    assert check_text("Spend was 32,550 before the cut.", ev).ok
    assert check_text("p = 1.59e-07", ev).ok


def test_text_without_numbers_passes_with_no_evidence():
    assert check_text("Let me check the overall trend first.", []).ok
    assert not check_text("Revenue is down 20%.", []).ok


# ---------------------------------------------------------- directionality ---

def test_a_narrated_gain_over_a_real_decline_is_blocked():
    ev = [make()]                                         # value -38.6, hypothesis "-39%"
    for text in ("North orders grew +38.6% in the period.",
                 "North orders rose +39% on 220 orders.",
                 "A gain of +38.6%."):
        res = check_text(text, ev)
        assert not res.ok, text
        assert res.unsupported and res.sign_flipped == res.unsupported


def test_a_narrated_decline_that_matches_the_evidence_passes():
    ev = [make()]
    assert check_text("North orders fell -38.6% in the period.", ev).ok
    assert check_text("Orders moved -39% on 220 orders (p<0.001).", ev).ok
    assert check_text("A decline of \u221238.6%.", ev).ok            # typographic minus


def test_a_genuine_gain_stated_as_a_gain_still_passes():
    ev = [make(value=12.5, hypothesis="Orders in South rose 12.5% on 180 orders", sample_size=180)]
    assert check_text("South orders rose +12.5% on 180 orders.", ev).ok
    assert check_text("South orders rose 12.5%.", ev).ok
    assert not check_text("South orders fell -12.5%.", ev).ok


def test_prose_that_repeats_the_magnitude_cannot_vouch_for_a_flipped_sign():
    # The hypothesis says "fell 12%" with no sign, so the only signed source is the value.
    ev = [make(hypothesis="Revenue fell 12% against its baseline", value=-12.0)]
    assert check_text("Revenue fell 12%.", ev).ok
    assert check_text("Revenue is -12%.", ev).ok
    assert not check_text("Revenue is +12%.", ev).ok


def test_the_baseline_carries_a_sign_too():
    ev = [make(value=-5.0, baseline=250.0, hypothesis="Revenue moved from the baseline", caveats=[])]
    assert check_text("The baseline was +250.", ev).ok
    assert not check_text("The baseline was -250.", ev).ok


def test_signed_detail_fields_are_checked_and_unsigned_ones_are_not():
    ev = [make(hypothesis="Control regions moved", value=1.0, details={
        "control_orders_change_pct": -7.4,       # a change: has a direction
        "ci_low_pct": -20.4, "ci_high_pct": 3.1,
        "spend_prior": 32550.0, "p_value": 0.02, "orders_last": 62,   # levels and counts: no direction
    })]
    assert check_text("Control regions moved -7.4%, CI -20.4% to +3.1%.", ev).ok
    assert not check_text("Control regions moved +7.4%.", ev).ok
    assert not check_text("The interval ran from +20.4% to 3.1%.", ev).ok
    # Counts, sample sizes and levels have no direction to flip, so a sign on them is not checked.
    assert check_text("Spend was +32,550 and there were +62 orders.", ev).ok
    assert check_text("Across +220 orders.", [make()]).ok


def test_a_zero_valued_field_has_no_direction_to_contradict():
    ev = [make(value=0.0, hypothesis="No change", details={"driver_change_pct": -0.0})]
    assert check_text("Spend moved +0.0% and then -0.0%.", ev).ok


def test_sign_flips_do_not_disturb_the_magnitude_checks():
    ev = [make()]
    res = check_text("A gain of +41%.", ev)
    assert not res.ok and res.unsupported == ["+41"] and res.sign_flipped == []   # wrong magnitude, not a flip
    assert not check_text("Orders grew +4x.", ev).ok


def test_hyphenated_ranges_and_dates_are_not_read_as_negative_numbers():
    ev = [make(hypothesis="Orders moved -39% between 2026-08 and 2026-09 across 220 orders")]
    assert check_text("Between 2026-08 and 2026-09 the change was -39%.", ev).ok


def test_is_signed_key_follows_the_naming_convention():
    for key in ("value", "baseline", "orders_change_pct", "segment_pct_change", "aov_effect",
                "orders_effect_pts", "ci_low_pct", "t", "z", "channel_pattern.other_ci_low_pct"):
        assert is_signed_key(key), key
    for key in ("p_value", "sample_size", "spend_prior", "orders_last", "revenue_at_stake",
                "latest_value", "baseline_periods", "days_last", "aov_prior"):
        assert not is_signed_key(key), key


def _numeric_detail_leaves(obj, key=""):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        yield key, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numeric_detail_leaves(v, f"{key}.{k}" if key else k)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numeric_detail_leaves(v, key)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_detail_field_that_goes_negative_in_real_evidence_is_treated_as_signed(scenario):
    """If a new tool stores a directional figure under a name the convention misses, the
    guardrail would silently stay sign-blind for it. This sweep is what would catch that."""
    orders, marketing = load_data(scenario_dir(scenario))
    events = list(investigate("Revenue dropped last month. Why?", Toolkit(orders, marketing)))
    evidence = events[-1].data["investigation"].evidence
    assert evidence
    missed = {key for ev in evidence
              for key, value in _numeric_detail_leaves({k: v for k, v in ev.details.items() if k != "provenance"})
              if value < 0 and not is_signed_key(key)}
    assert not missed, f"negative figures under unclassified detail keys: {sorted(missed)}"


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_real_evidence_still_backs_its_own_signed_figures(scenario):
    """Regression on the common case: quoting a real figure with its real sign is never blocked,
    and flipping that sign always is."""
    orders, marketing = load_data(scenario_dir(scenario))
    events = list(investigate("Revenue dropped last month. Why?", Toolkit(orders, marketing)))
    evidence = events[-1].data["investigation"].evidence
    signed, _ = _allowed(evidence)
    checked = 0
    for ev in evidence:
        if ev.value is None or abs(ev.value) < 0.05:
            continue
        real = f"{ev.value:+.2f}"
        flipped_sign = -1 if ev.value > 0 else 1
        flipped = f"{'+' if flipped_sign > 0 else '-'}{abs(ev.value):.2f}"
        assert check_text(f"The change was {real}.", evidence).ok, (ev.id, real)
        # Flipping is only detectable when nothing else in the evidence legitimately carries the
        # same magnitude with the opposite sign (an item's own details often echo its value).
        if not any(abs(m - abs(ev.value)) < 0.01 and sg == flipped_sign for m, sg in signed):
            assert not check_text(f"The change was {flipped}.", evidence).ok, (ev.id, flipped)
            checked += 1
    assert checked
