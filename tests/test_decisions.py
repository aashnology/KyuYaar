import pytest

from decisions import build_options
from narration import build_summary, narrate_step
from orchestrator import investigate
from report import build_report


@pytest.fixture(scope="module")
def inv(toolkit):
    return [e for e in investigate("why", toolkit) if e.kind == "done"][0].data["investigation"]


def test_supported_causes_map_to_options(inv):
    ds = build_options(inv.evidence)
    ids = [o.id for o in ds.options]
    assert "restore_marketing_North" in ids
    assert "test_marketing_North" in ids
    assert "rollback_price_Electronics" in ids
    assert "test_price_Electronics" in ids
    assert "sequence_fixes" in ids
    assert ids[-1] == "hold_and_monitor"
    assert not ds.insufficient


def test_weak_evidence_produces_no_action_options(inv):
    weak_only = [e for e in inv.evidence if e.strength == "weak"]
    ds = build_options(weak_only)
    assert ds.insufficient
    assert [o.id for o in ds.options] == ["hold_and_monitor"]


def test_every_option_states_assumptions_and_risks(inv):
    for opt in build_options(inv.evidence).options:
        assert opt.assumptions and opt.risks and opt.impact


def test_impact_figures_come_from_the_evidence(inv):
    stat = {e.id: e for e in inv.evidence}
    ds = build_options(inv.evidence)
    north = next(o for o in ds.options if o.id == "restore_marketing_North")
    expected = f"{stat['stat_marketing_North'].details['revenue_at_stake']:,.0f}"
    assert expected in north.impact


def test_overlapping_estimates_are_not_summed(inv):
    hold = build_options(inv.evidence).options[-1]
    total = sum(e.details["revenue_at_stake"] for e in inv.evidence if e.details.get("revenue_at_stake"))
    assert f"{total:,.0f}" not in hold.impact
    assert "should not be added" in hold.impact


def test_unsupported_hypotheses_are_listed(inv):
    ds = build_options(inv.evidence)
    joined = " ".join(ds.not_supported)
    assert "East" in joined and "Apparel" in joined


def test_summary_and_narration_use_evidence_numbers(inv):
    assert "23.9%" in build_summary(inv.evidence)
    assert "associations, not proof" in build_summary(inv.evidence)
    assert "does not support a specific cause" in build_summary([e for e in inv.evidence if e.strength == "weak"])


def test_report_records_the_human_choice(inv):
    ds = build_options(inv.evidence)
    memo = build_report(inv, ds, "hold_and_monitor", "Waiting one more month.")
    assert "Chosen by the person responsible" in memo
    assert "Waiting one more month." in memo
    assert "`marketing_effect`" in memo and "src/effects.py" in memo
    assert "No option chosen yet." in build_report(inv, ds)


def test_summary_is_split_into_readable_blocks(inv):
    summary = build_summary(inv.evidence)
    assert "\n\n" in summary                       # separate paragraphs
    # Two bullets for the supported causes, two for the order-pattern check on each,
    # one for the channel finding.
    assert "Does the order pattern match" in summary and "Which channel?" in summary
    assert summary.count("\n- ") == 5
