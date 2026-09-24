"""
Featherless AI adapter (OpenAI-style chat completions), tested against a scripted
transport. These check request and response handling for that wire format; they do
not reach the real API, so a live run is still the final check.
"""

import json

import pytest

from llm import (
    FEATHERLESS_DEFAULT_MODEL, FEATHERLESS_ENDPOINT, FeatherlessAdapter, LLMError, ToolResult,
    _to_openai_tool, make_client,
)
from orchestrator import STANDARD_PLAN, investigate
from toolkit import TOOL_SPECS

from test_llm import Transport


def reply(content=None, *calls):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return {"choices": [{"message": message}]}


def tc(name, args=None, id=None, raw_args=None):
    call = {"type": "function", "function": {"name": name, "arguments": raw_args if raw_args is not None else json.dumps(args or {})}}
    if id:
        call["id"] = id
    return call


def adapter(responses, **kw):
    transport = Transport(responses)
    sleeps = []
    kw.setdefault("sleep", sleeps.append)
    a = FeatherlessAdapter("test-key", "org/model-test", post=transport, **kw)
    a.sleeps = sleeps
    return a, transport


def test_tool_specs_become_openai_function_tools():
    tools = {t["function"]["name"]: t for t in (_to_openai_tool(x) for x in TOOL_SPECS)}
    seg = tools["segment_breakdown"]["function"]["parameters"]
    assert tools["segment_breakdown"]["type"] == "function"
    assert seg["type"] == "object" and seg["required"] == ["dimension"]
    assert seg["properties"]["dimension"]["enum"] == ["region", "category"]
    # a tool with no arguments still sends an (empty) object schema, as this format expects
    assert tools["marketing_effect"]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_request_shape_and_authentication():
    a, transport = adapter([reply("Hello.")])
    a.start("system text", TOOL_SPECS, "the question")
    a.next_turn()
    call = transport.calls[0]
    assert call["url"] == FEATHERLESS_ENDPOINT
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["payload"]["model"] == "org/model-test"
    assert call["payload"]["messages"][:2] == [
        {"role": "system", "content": "system text"}, {"role": "user", "content": "the question"}]
    assert len(call["payload"]["tools"]) == len(TOOL_SPECS)


def test_no_tools_key_when_there_are_none():
    a, transport = adapter([reply("Plain answer.")])
    a.start("s", [], "q")
    a.next_turn()
    assert "tools" not in transport.calls[0]["payload"]


def test_tool_calls_round_trip_with_matching_ids():
    a, transport = adapter([
        reply("Checking.", tc("baseline_trend", {"metric": "revenue"}, id="abc1")),
        reply("Done."),
    ])
    a.start("s", TOOL_SPECS, "q")
    turn = a.next_turn()
    assert turn.texts == ["Checking."]
    assert [(c.id, c.name, c.args) for c in turn.tool_calls] == [("abc1", "baseline_trend", {"metric": "revenue"})]

    a.add_results([ToolResult("abc1", "baseline_trend", '{"ok": true}')])
    a.next_turn()
    sent = transport.calls[1]["payload"]["messages"]
    assert sent[2]["role"] == "assistant" and sent[2]["tool_calls"][0]["id"] == "abc1"
    assert sent[3] == {"role": "tool", "tool_call_id": "abc1", "content": '{"ok": true}'}


def test_a_missing_call_id_is_generated_once_and_reused():
    a, transport = adapter([reply(None, tc("marketing_effect")), reply("Done.")])
    a.start("s", TOOL_SPECS, "q")
    turn = a.next_turn()
    call_id = turn.tool_calls[0].id
    assert call_id
    a.add_results([ToolResult(call_id, "marketing_effect", "{}")])
    a.next_turn()
    sent = transport.calls[1]["payload"]["messages"]
    assert sent[2]["tool_calls"][0]["id"] == call_id and sent[3]["tool_call_id"] == call_id


def test_tool_errors_are_sent_as_json_error_bodies():
    a, transport = adapter([reply(None, tc("nope", id="t1")), reply("Done.")])
    a.start("s", TOOL_SPECS, "q")
    a.next_turn()
    a.add_results([ToolResult("t1", "nope", "unknown tool 'nope'", is_error=True)])
    a.next_turn()
    assert json.loads(transport.calls[1]["payload"]["messages"][3]["content"]) == {"error": "unknown tool 'nope'"}


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", ""])
def test_malformed_arguments_become_an_empty_dict(raw):
    a, _ = adapter([reply(None, tc("baseline_trend", id="x", raw_args=raw))])
    a.start("s", TOOL_SPECS, "q")
    assert a.next_turn().tool_calls[0].args == {}


def test_reasoning_blocks_are_not_treated_as_narration():
    a, _ = adapter([reply("<think>Revenue fell 99% probably.\nlet me check</think>\nRevenue is down 23.9%.")])
    a.start("s", TOOL_SPECS, "q")
    assert a.next_turn().texts == ["Revenue is down 23.9%."]
    b, _ = adapter([reply("<think>only thinking</think>")])
    b.start("s", TOOL_SPECS, "q")
    assert b.next_turn().texts == []


def test_empty_response_is_a_clear_error():
    a, _ = adapter([{"choices": []}])
    a.start("s", TOOL_SPECS, "q")
    with pytest.raises(LLMError, match="empty response"):
        a.next_turn()


def test_transient_errors_are_retried_like_gemini():
    a, transport = adapter([LLMError("HTTP 503: busy", status=503), reply("Fine.")], retry_delays=(3, 9))
    a.start("s", TOOL_SPECS, "q")
    assert a.next_turn().texts == ["Fine."]
    assert a.sleeps == [3] and len(transport.calls) == 2
    assert any("waiting" in n for n in a.drain_notices())


def test_api_key_never_appears_in_error_text():
    a, _ = adapter([LLMError("HTTP 401: invalid credentials", status=401)])
    a.start("s", TOOL_SPECS, "q")
    with pytest.raises(LLMError) as info:
        a.next_turn()
    assert "test-key" not in str(info.value)


def test_full_investigation_through_the_featherless_adapter(toolkit):
    steps = [
        ("Confirming the trend.", "baseline_trend", {"metric": "revenue"}),
        ("Revenue is down 23.9%.", "aov_volume_decomposition", {}),
        ("The change is in order count, down 22.4%.", "segment_breakdown", {"dimension": "region"}),
        ("North stands out.", "segment_breakdown", {"dimension": "category"}),
        ("Electronics stands out.", "aov_volume_decomposition", {"dimension": "region"}),
        ("North order count is down 47.6%.", "aov_volume_decomposition", {"dimension": "category"}),
        ("Electronics order count is down 41.9%.", "marketing_effect", {}),
        ("Marketing in North is supported.", "marketing_channel_analysis", {}),
        ("The Paid channel in North is supported.", "price_effect", {}),
    ]
    responses = [reply(text, tc(name, args, id=f"c{i}")) for i, (text, name, args) in enumerate(steps)]
    responses.append(reply("Revenue is down 23.9%. North and Electronics carry the change."))
    a, _ = adapter(responses)
    events = list(investigate("Revenue dropped. Why?", toolkit, client=a))
    inv = events[-1].data["investigation"]
    assert inv.mode == "live" and inv.provider == "featherless" and inv.model == "org/model-test"
    assert [s["tool"] for s in inv.steps] == [t for t, _ in STANDARD_PLAN]
    assert inv.summary_source == "llm" and not inv.guardrail_blocks
    assert events[0].data["display"] == "Featherless"


def test_guardrail_applies_to_featherless_prose_too(toolkit):
    a, _ = adapter([reply("Revenue is down a staggering 62%.", tc("baseline_trend", id="c0")), reply("Fine.")])
    inv = list(investigate("Revenue dropped. Why?", toolkit, client=a))[-1].data["investigation"]
    assert inv.guardrail_blocks and "62" in inv.guardrail_blocks[0]["unsupported"]


def test_make_client_selection(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "FEATHERLESS_API_KEY",
                "KYUYAAR_PROVIDER", "KYUYAAR_MODEL"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("FEATHERLESS_API_KEY", "f")
    c = make_client()                                   # the only key present
    assert c.provider == "featherless" and c.model == FEATHERLESS_DEFAULT_MODEL

    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert make_client().provider == "gemini"           # existing behaviour: not overridden by a second key
    monkeypatch.setenv("KYUYAAR_PROVIDER", "featherless")
    assert make_client().provider == "featherless"      # unless asked for explicitly

    monkeypatch.setenv("KYUYAAR_MODEL", "org/other")
    assert make_client().model == "org/other"

    monkeypatch.delenv("FEATHERLESS_API_KEY")
    assert make_client() is None                        # forced provider has no key
