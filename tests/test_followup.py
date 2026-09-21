"""
Follow-up questions. The offline path is checked against the real evidence; the
live path is checked against scripted fakes, which tests the flow, the guardrail
and the fallbacks but not the real model.
"""

import json
from types import SimpleNamespace

import pytest

from followup import COVERED_TEXT, MAX_FINDINGS, answer_question, template_answer
from guardrail import check_text
from llm import GeminiAdapter
from orchestrator import investigate


@pytest.fixture(scope="module")
def inv(toolkit):
    events = list(investigate("Revenue dropped. Why?", toolkit))
    return events[-1].data["investigation"]


class FakeAdapter:
    """Stands in for a provider adapter and records what it was asked."""

    provider, display, model = "fake", "Fake", "fake-1"

    def __init__(self, *texts, error=None):
        self.texts, self.error, self.started = list(texts), error, None

    def start(self, system, tools, question):
        self.started = {"system": system, "tools": tools, "question": question}

    def next_turn(self):
        if self.error:
            raise self.error
        return SimpleNamespace(texts=self.texts, tool_calls=[])


QUESTIONS = [
    "Why did North drop?", "Is the loss in North mainly Paid?", "Did prices cause the Electronics decline?",
    "Is South a problem?", "How sure are you?", "Are orders smaller or fewer?",
    "How much did revenue fall?", "Which region is hit hardest?", "Was it the marketing spend?",
    "What about customer segments?", "What should I do?", "hello",
]


# ------------------------------------------------------------ offline ---

@pytest.mark.parametrize("question", QUESTIONS)
def test_every_offline_answer_passes_the_numeric_guardrail(inv, question):
    """Template answers are built from Evidence fields, so no figure can be invented."""
    answer = answer_question(question, inv)
    assert answer.source == "template"
    assert check_text(answer.text, inv.evidence).ok, answer.text
    assert set(answer.evidence_ids) <= {e.id for e in inv.evidence}
    assert len(answer.evidence_ids) <= MAX_FINDINGS


def test_a_named_segment_pulls_that_segments_findings_first(inv):
    answer = answer_question("Why did North drop?", inv)
    assert "stat_marketing_North" in answer.evidence_ids
    assert all("North" in i for i in answer.evidence_ids)
    assert answer.text.startswith("Strong evidence")


def test_channel_question_about_a_region_reaches_the_channel_cell(inv):
    answer = answer_question("Is the loss in North mainly Paid?", inv)
    assert answer.evidence_ids[0] == "chan_North_Paid"


def test_orders_versus_order_value_is_answered_for_the_business_as_a_whole(inv):
    answer = answer_question("Are orders smaller or fewer?", inv)
    assert answer.evidence_ids == ["decomp_orders_overall", "decomp_aov_overall"]


def test_a_why_question_reaches_the_supported_causes_not_just_the_overall_trend(inv):
    answer = answer_question("Why did revenue drop?", inv)
    assert {"stat_marketing_North", "stat_price_Electronics"} <= set(answer.evidence_ids)
    assert answer.evidence_ids[0] == "obs_baseline_trend"


def test_a_weak_segment_is_reported_as_not_supported(inv):
    answer = answer_question("Is South a problem?", inv)
    assert answer.text.count("Weak evidence") >= 1
    assert "Strong evidence" not in answer.text
    assert "Nothing here is strong enough to support as an explanation." in answer.text


def test_a_question_the_evidence_cannot_answer_says_so_and_names_what_was_covered(inv):
    answer = answer_question("What about customer segments?", inv)
    assert not answer.covered and not answer.evidence_ids
    assert "did not test that" in answer.text
    assert COVERED_TEXT in answer.text


def test_advice_is_pointed_at_the_decision_screen_not_given(inv):
    answer = answer_question("What should I do about it?", inv)
    assert not answer.covered
    assert "Decision screen" in answer.text


def test_empty_and_overlong_questions_are_handled(inv):
    assert not answer_question("   ", inv).covered
    long = answer_question("why " * 500, inv)
    assert len(long.question) <= 600


def test_no_matching_evidence_never_produces_a_guess():
    assert not template_answer("Why did North drop?", []).covered


# --------------------------------------------------------------- live ---

def test_live_answer_is_used_when_it_cites_real_evidence_and_passes_the_guardrail(inv):
    fake = FakeAdapter(
        "North's marketing spend fell 33.8%, and orders there changed -38.6% against regions with "
        "typical spend [stat_marketing_North]. That accompanies the drop; it does not prove it."
    )
    answer = answer_question("Why did North drop?", inv, client=fake)
    assert answer.source == "llm" and answer.covered
    assert answer.evidence_ids == ["stat_marketing_North"]
    assert "[" not in answer.text and "33.8%" in answer.text
    assert not answer.notes


def test_live_call_gets_the_evidence_and_no_tools(inv):
    fake = FakeAdapter("Text [obs_baseline_trend].")
    answer_question("How much did revenue fall?", inv, client=fake)
    assert fake.started["tools"] == []
    prompt = fake.started["question"]
    assert "obs_baseline_trend" in prompt and "How much did revenue fall?" in prompt
    payload = json.loads(prompt.split("Evidence (JSON):\n")[1].split("\n\nFollow-up")[0])
    assert {p["id"] for p in payload} == {e.id for e in inv.evidence}
    assert "provenance" not in json.dumps(payload)


def test_live_answer_with_an_invented_figure_is_replaced_by_the_template_answer(inv):
    fake = FakeAdapter("North fell 61.7% [stat_marketing_North].")
    answer = answer_question("Why did North drop?", inv, client=fake)
    assert answer.source == "template"
    assert answer.blocked == ["61.7"]
    assert any("61.7" in n for n in answer.notes)
    assert check_text(answer.text, inv.evidence).ok


def test_live_answer_citing_an_id_that_does_not_exist_is_discarded(inv):
    fake = FakeAdapter("Spend fell 33.8% [stat_marketing_Atlantis].")
    answer = answer_question("Why did North drop?", inv, client=fake)
    assert answer.source == "template"
    assert any("Atlantis" in n for n in answer.notes)


def test_a_figure_the_person_typed_may_be_repeated_back(inv):
    reply = "You asked about 7.77; nothing in the evidence tests that, so it was not tested."
    # The figure is not in the evidence, so only the person having typed it can excuse it.
    assert not check_text(reply, inv.evidence).ok
    answer = answer_question("What if spend changed by 7.77?", inv, client=FakeAdapter(reply))
    assert answer.source == "llm"
    # Typing a number does not license any other figure.
    other = answer_question("What if spend changed by 7.77?", inv, client=FakeAdapter(reply + " North fell 61.7%."))
    assert other.source == "template" and other.blocked == ["61.7"]


def test_model_failure_degrades_to_the_template_answer_and_says_so(inv):
    answer = answer_question("Why did North drop?", inv, client=FakeAdapter(error=RuntimeError("quota")))
    assert answer.source == "template" and answer.evidence_ids
    assert any("model call failed" in n for n in answer.notes)


def test_empty_model_reply_falls_back(inv):
    answer = answer_question("Why did North drop?", inv, client=FakeAdapter())
    assert answer.source == "template"


# ------------------------------------------------- adapters, no tools ---

def test_gemini_omits_the_tools_key_when_there_are_none():
    calls = []

    def post(url, headers, payload, timeout):
        calls.append(json.loads(json.dumps(payload)))
        return {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]}

    adapter = GeminiAdapter("k", "m", post=post, min_interval=0)
    adapter.start("SYSTEM", [], "question")
    assert adapter.next_turn().texts == ["ok"]
    assert "tools" not in calls[0]


def test_anthropic_adapter_omits_tools_when_there_are_none():
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

    from llm import AnthropicAdapter
    adapter = AnthropicAdapter(SimpleNamespace(messages=SimpleNamespace(create=create)), "m")
    adapter.start("SYSTEM", [], "question")
    adapter.next_turn()
    assert "tools" not in seen[0]
