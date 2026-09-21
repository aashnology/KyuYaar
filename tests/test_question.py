"""
The investigation-type check: the system says so when a question is not one it
can investigate, instead of running a revenue investigation regardless.
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from orchestrator import investigate
from question import REVENUE_CHANGE, classify_question

APP = str(Path(__file__).resolve().parents[1] / "app.py")

os.environ["KYUYAAR_PACE"] = "0"
os.environ["KYUYAAR_DECISION_LOG"] = str(Path(tempfile.mkdtemp()) / "decisions.jsonl")
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

SUPPORTED = [
    "Revenue dropped last month. Why did it happen, and what should I do?",   # the app's own default
    "Revenue dropped last month. Why?",                                       # the scripts' question
    "why did our sales fall in August?",
    "What happened to orders last month?",
    "Why is revenue down?",
    "Why did average order value fall?",
    "Our takings are lower than usual, what is going on?",
    "REVENUE DROPPED!!! WHY",
    "Revenue dropped 24% last month, will it keep falling?",                  # a forecast word, but about a change that happened
]

DECLINED = [
    "how many customers do we have?",
    "what's the weather",
    "hello",
    "Forecast next quarter revenue",
    "What will revenue be next month?",
    "Why did profit drop?",
    "why did churn go up",
    "which region sells the most?",
    "revenue",                                                                # a metric, but no change asked about
    "",
    "   ",
]


@pytest.mark.parametrize("question", SUPPORTED)
def test_supported_questions_are_accepted(question):
    check = classify_question(question)
    assert check.supported and check.kind == REVENUE_CHANGE


@pytest.mark.parametrize("question", DECLINED)
def test_other_questions_are_declined_with_a_reason(question):
    check = classify_question(question)
    assert not check.supported and check.kind is None
    assert check.reason.strip()
    assert "Revenue dropped last month" in check.reason or "Type a question" in check.reason   # says what to ask instead


def test_the_reason_names_what_is_out_of_scope():
    assert "profit" in classify_question("Why did profit drop?").reason
    assert "forecast" in classify_question("Forecast next quarter revenue").reason
    assert "customer" in classify_question("how many customers do we have?").reason
    assert "doesn't ask about a change" in classify_question("revenue").reason


def test_none_is_handled():
    assert not classify_question(None).supported


def test_a_declined_question_runs_no_tool_and_makes_no_model_call(toolkit):
    def explode(**kwargs):
        raise AssertionError("the model must not be called for a declined question")

    model = SimpleNamespace(messages=SimpleNamespace(create=explode))
    toolkit.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no tool should run"))
    try:
        events = list(investigate("how many customers do we have?", toolkit, client=model))
    finally:
        del toolkit.run                                                        # restore the class method

    assert [e.kind for e in events] == ["start", "unsupported", "done"]
    inv = events[-1].data["investigation"]
    assert inv.unsupported and inv.mode == "declined"
    assert inv.evidence == [] and inv.steps == []
    assert inv.summary == inv.unsupported == events[1].data["reason"]


def test_a_supported_question_still_investigates(toolkit):
    inv = list(investigate("Revenue dropped last month. Why?", toolkit))[-1].data["investigation"]
    assert inv.unsupported is None and inv.evidence


def test_app_declines_an_off_script_question_and_stays_on_the_command_center():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_area(key="question").set_value("how many customers do we have?").run()
    next(b for b in at.button if b.label.startswith("Investigate")).click().run()

    assert not at.exception
    assert any("doesn't match a supported investigation type" in w.value for w in at.warning)
    assert not any("evidence chain is ready" in s.value for s in at.success)
    nav = {b.label: b.disabled for b in at.button if b.label[0] in "234"}
    assert all(nav.values())                                                   # no investigation, so later screens stay locked


def test_app_clears_the_message_once_a_supported_question_runs():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_area(key="question").set_value("hello").run()
    next(b for b in at.button if b.label.startswith("Investigate")).click().run()
    assert any("doesn't match" in w.value for w in at.warning)

    at.text_area(key="question").set_value("Why did revenue drop last month?").run()
    next(b for b in at.button if b.label.startswith("Investigate")).click().run()
    assert not at.exception
    assert any("evidence chain is ready" in s.value for s in at.success)
    assert not any("doesn't match" in w.value for w in at.warning)
