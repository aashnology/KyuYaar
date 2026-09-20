"""
The live loop is tested against a scripted fake client that mimics the parts of
the Anthropic SDK the orchestrator uses. This checks the tool-calling flow,
guardrail handling and fallbacks; it does not test the real model.
"""

from types import SimpleNamespace

import pytest

from orchestrator import STANDARD_PLAN, investigate


def text(t):
    return SimpleNamespace(type="text", text=t)


def call(i, name, **args):
    return SimpleNamespace(type="tool_use", id=f"tu_{i}", name=name, input=args)


def turn(*blocks):
    return SimpleNamespace(content=list(blocks), stop_reason="tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn")


class FakeClient:
    def __init__(self, turns):
        self.turns, self.requests = list(turns), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # Snapshot the message list, since the orchestrator keeps appending to it.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.turns.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def run(toolkit, client=None):
    events = list(investigate("Revenue dropped. Why?", toolkit, client=client))
    return events, events[-1].data["investigation"]


GOOD_SCRIPT = lambda: [
    turn(text("I'll confirm the metric moved first."), call(1, "baseline_trend", metric="revenue")),
    turn(text("Revenue is down 23.9%, a strong signal. Now splitting fewer orders from smaller orders."), call(2, "aov_volume_decomposition")),
    turn(text("The change is in order count, down 22.4%. Now locating it."), call(3, "segment_breakdown", dimension="region")),
    turn(text("North accounts for 45% of the change against a 26% share of prior revenue."), call(4, "segment_breakdown", dimension="category")),
    turn(text("Electronics carries 60% of the change."), call(5, "aov_volume_decomposition", dimension="region")),
    turn(text("North order count is down 47.6%."), call(6, "aov_volume_decomposition", dimension="category")),
    turn(text("Electronics order count is down 41.9%."), call(7, "marketing_effect")),
    turn(text("North shows a strong marketing-spend association."), call(8, "marketing_channel_analysis")),
    turn(text("The Paid channel in North shows a moderate association."), call(9, "price_effect")),
    turn(text("Electronics shows a strong price association."),),
]


def test_offline_mode_runs_the_standard_plan(toolkit):
    events, inv = run(toolkit)
    assert inv.mode == "offline" and inv.model is None
    assert [(s["tool"], s["args"]) for s in inv.steps] == STANDARD_PLAN
    assert all(s["narration_source"] == "template" for s in inv.steps)
    assert inv.summary_source == "template"
    assert not inv.guardrail_blocks


def test_live_loop_uses_the_models_narration_when_it_passes_the_guardrail(toolkit):
    good = GOOD_SCRIPT()
    good[-1] = turn(text("Revenue is down 23.9%. North and Electronics carry the change, each with strong evidence."))
    client = FakeClient(good)
    events, inv = run(toolkit, client)
    assert inv.mode == "live"
    assert [s["tool"] for s in inv.steps] == [t for t, _ in STANDARD_PLAN]
    assert not inv.guardrail_blocks
    assert not any(e.kind == "coverage" for e in events)
    assert inv.summary_source == "llm"
    assert inv.steps[0]["narration_source"] == "llm"
    # The tool schemas and system prompt went out with every request.
    assert all(r["tools"] and r["system"] for r in client.requests)


def test_tool_results_reach_the_model_as_json(toolkit):
    client = FakeClient(GOOD_SCRIPT())
    run(toolkit, client)
    last_user = client.requests[1]["messages"][-1]
    assert last_user["role"] == "user"
    block = last_user["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "tu_1"
    assert "hypothesis" in block["content"] and "provenance" not in block["content"]


def test_invented_figures_are_blocked_and_replaced(toolkit):
    script = GOOD_SCRIPT()
    script[1] = turn(text("Revenue is down about 31%, a disaster."), call(2, "aov_volume_decomposition"))
    events, inv = run(toolkit, FakeClient(script))
    assert len(inv.guardrail_blocks) >= 1
    assert "31" in inv.guardrail_blocks[0]["unsupported"]
    first = inv.steps[0]
    assert first["narration_source"] == "template"
    assert "31%" not in first["narration"]
    assert any(e.kind == "guardrail" for e in events)


def test_a_bad_final_summary_falls_back_to_the_template(toolkit):
    script = GOOD_SCRIPT()
    script[-1] = turn(text("Everything is 87% explained by North."))
    _, inv = run(toolkit, FakeClient(script))
    assert inv.summary_source == "template"
    assert "87" not in inv.summary


def test_skipped_tools_are_filled_in_and_flagged(toolkit):
    script = [
        turn(text("Checking the trend."), call(1, "baseline_trend")),
        turn(text("That is enough for me.")),
    ]
    events, inv = run(toolkit, FakeClient(script))
    assert [s["tool"] for s in inv.steps] == [t for t, _ in STANDARD_PLAN]
    assert sum(e.kind == "coverage" for e in events) == len(STANDARD_PLAN) - 1
    assert inv.summary_source == "template"        # the model's summary predates the missing evidence


def test_api_failure_degrades_to_offline_completion(toolkit):
    events, inv = run(toolkit, FakeClient([RuntimeError("boom")]))
    assert any(e.kind == "note" for e in events)
    assert [s["tool"] for s in inv.steps] == [t for t, _ in STANDARD_PLAN]
    assert inv.notes


def test_invalid_tool_arguments_are_returned_as_errors_not_crashes(toolkit):
    script = [
        turn(call(1, "segment_breakdown", dimension="colour")),
        turn(text("Understood.")),
    ]
    client = FakeClient(script)
    _, inv = run(toolkit, client)
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "dimension" in result["content"]
    assert len(inv.steps) == len(STANDARD_PLAN)    # coverage fill still completes the plan


def test_evidence_is_deduplicated_by_id(toolkit):
    script = [
        turn(call(1, "segment_breakdown", dimension="region"), call(2, "segment_breakdown", dimension="region")),
        turn(text("Done.")),
    ]
    _, inv = run(toolkit, FakeClient(script))
    ids = [e.id for e in inv.evidence]
    assert len(ids) == len(set(ids))
