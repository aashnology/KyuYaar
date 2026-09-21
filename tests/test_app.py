"""End-to-end click-through of the Streamlit app in offline mode."""

import os
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

os.environ["KYUYAAR_PACE"] = "0"
# The decision log is a real file; keep the tests away from the repository's own.
os.environ["KYUYAAR_DECISION_LOG"] = str(Path(tempfile.mkdtemp()) / "decisions.jsonl")
os.environ.pop("ANTHROPIC_API_KEY", None)


def click(at, prefix):
    button = next(b for b in at.button if b.label.startswith(prefix))
    return button.click().run()


def test_full_flow_reaches_a_recorded_decision():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any(b.label.startswith("Investigate") for b in at.button)

    at = click(at, "Investigate")
    assert not at.exception
    assert any("evidence chain is ready" in s.value for s in at.success)

    at = click(at, "See the evidence")
    assert not at.exception
    assert [s.value for s in at.subheader] == [
        "Trends by metric", "What changed", "Fewer orders, or smaller orders?",
        "Where the change is concentrated", "Candidate causes tested", "Marketing by channel",
        "Ask about these findings",
    ]

    at = click(at, "Continue to the decision")
    assert not at.exception
    choose = [b for b in at.button if b.label == "Choose this option"]
    assert len(choose) >= 3

    choose[0].click().run()
    assert not at.exception
    assert any("You chose this option" in s.value for s in at.success)


def test_later_screens_are_locked_until_an_investigation_exists():
    at = AppTest.from_file(APP, default_timeout=120).run()
    locked = {b.label: b.disabled for b in at.button if b.label[0] in "234"}
    assert all(locked.values())


def test_decision_screen_projects_each_option_from_the_persons_assumptions():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at = click(at, "Investigate")
    at = click(at, "See the evidence")
    at = click(at, "Continue to the decision")
    assert not at.exception

    share = at.slider(key="share_restore_marketing_North")
    revenue = lambda: next(m.value for m in at.metric if m.label.startswith("Revenue recovered"))
    before = revenue()
    share.set_value(100).run()
    assert not at.exception
    assert revenue() != before

    # The time frame is shared by every projection.
    at.slider(key="horizon_months").set_value(6).run()
    assert not at.exception
    assert any(m.label.endswith("6 mo") for m in at.metric)

    # Options the data cannot size say so instead of showing a number.
    assert any("Not projected" in i.value for i in at.info)

    # Choosing an option still works with the projections on screen.
    next(b for b in at.button if b.key == "choose_restore_marketing_North").click().run()
    assert not at.exception
    assert any("You chose this option" in s.value for s in at.success)


def _to_evidence(at):
    at = click(at, "Investigate")
    return click(at, "See the evidence")


def test_follow_up_is_answered_from_the_evidence_and_kept_for_the_report():
    at = _to_evidence(AppTest.from_file(APP, default_timeout=120).run())
    at.text_input(key="followup_text").set_value("Why did North drop?")
    at = click(at, "Ask")
    assert not at.exception
    (qa,) = at.session_state["followups"]
    assert qa.source == "template" and "stat_marketing_North" in qa.evidence_ids
    assert any("Evidence used" in c.value and "stat_marketing_North" in c.value for c in at.caption)

    # An example question works too, and a question the evidence cannot answer says so.
    at = click(at, "Are orders smaller or fewer?")
    at.text_input(key="followup_text").set_value("What about customer segments?")
    at = click(at, "Ask")
    assert not at.exception
    assert [q.covered for q in at.session_state["followups"]] == [True, True, False]
    assert any("Not answerable" in c.value for c in at.caption)

    # Starting a new investigation clears the questions of the old one.
    at = click(at, "1 ·")
    at = click(at, "Investigate")
    assert at.session_state["followups"] == []


def test_evidence_screen_shows_trend_charts_for_supported_causes():
    at = _to_evidence(AppTest.from_file(APP, default_timeout=120).run())
    assert not at.exception
    # Two supported causes (North marketing, Electronics price), each with its driver trend.
    assert set(at.session_state["trend_cache"]) >= {"stat_marketing_North", "stat_price_Electronics"}
    assert all(t is not None for t in at.session_state["trend_cache"].values())

    # The per-metric trend follows the person's choice of metric and split.
    at.radio(key="trend_metric").set_value("aov").run()
    at.radio(key="trend_group").set_value("region").run()
    assert not at.exception


def test_choosing_then_saving_writes_one_record_awaiting_an_outcome(monkeypatch, tmp_path):
    from decision_log import DecisionLog
    monkeypatch.setenv("KYUYAAR_DECISION_LOG", str(tmp_path / "decisions.jsonl"))
    at = _to_evidence(AppTest.from_file(APP, default_timeout=120).run())
    at = click(at, "Continue to the decision")
    save = lambda: next(b for b in at.button if b.label == "Save to decision log")
    assert save().disabled                                # nothing chosen yet

    next(b for b in at.button if b.key == "choose_restore_marketing_North").click().run()
    at = click(at, "Save to decision log")
    assert not at.exception
    assert any("awaiting an outcome" in s.value for s in at.success)
    assert save().disabled                                # cannot be saved twice

    (record,) = DecisionLog(tmp_path / "decisions.jsonl").all()
    assert record.option_id == "restore_marketing_North"
    assert record.status == "awaiting_outcome" and record.outcome is None
    assert record.projection is not None

    # Choosing a different option makes a new record possible.
    next(b for b in at.button if b.key == "choose_hold_and_monitor").click().run()
    assert not save().disabled
