"""The downloadable report: usable before a choice, and carrying the follow-up questions."""

import pytest

from decisions import build_options
from followup import answer_question
from orchestrator import investigate
from report import build_report


@pytest.fixture(scope="module")
def inv(toolkit):
    return list(investigate("Revenue dropped. Why?", toolkit))[-1].data["investigation"]


def test_report_before_a_choice_is_an_investigation_report(inv):
    report = build_report(inv, build_options(inv.evidence))
    assert report.startswith("# KyuYaar investigation report")
    assert "No option chosen yet." in report
    assert "## Evidence chain" in report and "## Options considered" in report


def test_report_after_a_choice_is_a_decision_memo(inv):
    ds = build_options(inv.evidence)
    assert build_report(inv, ds, "hold_and_monitor").startswith("# KyuYaar decision memo")
    # An id that matches no option is not a choice.
    assert build_report(inv, ds, "nonexistent").startswith("# KyuYaar investigation report")


def test_follow_up_questions_are_recorded_with_the_evidence_they_used(inv):
    ds = build_options(inv.evidence)
    qa = [answer_question("Why did North drop?", inv), answer_question("What about customer segments?", inv)]
    report = build_report(inv, ds, followups=qa)
    assert "## Follow-up questions" in report
    assert "**Q:** Why did North drop?" in report
    assert "evidence used: `stat_marketing_North`, `assoc_region_North`" in report
    assert "not answerable from this investigation's evidence" in report
    assert "generated from the evidence" in report
    # The section sits between the evidence chain and the options.
    assert report.index("## Evidence chain") < report.index("## Follow-up questions") < report.index("## Options considered")


def test_no_follow_up_section_when_nothing_was_asked(inv):
    assert "Follow-up questions" not in build_report(inv, build_options(inv.evidence))
