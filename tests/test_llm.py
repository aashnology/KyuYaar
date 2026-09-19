"""
Provider adapters, tested against a scripted transport. These check request
and response handling for the Gemini wire format; they do not reach the real
API, so a live run is still the final check.
"""

import json

import pytest

from llm import (
    GEMINI_DEFAULT_MODEL, GeminiAdapter, LLMError, ToolResult, _to_gemini_tool, make_client,
)
from orchestrator import STANDARD_PLAN, investigate
from toolkit import TOOL_SPECS


class Transport:
    """Stands in for the HTTP layer; records requests and replays responses."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, headers, payload, timeout):
        # Deep-copy through JSON so later mutation cannot rewrite history.
        self.calls.append({"url": url, "headers": dict(headers), "payload": json.loads(json.dumps(payload))})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(*parts):
    return {"candidates": [{"content": {"role": "model", "parts": list(parts)}}]}


def fc(name, args=None, **extra):
    return {"functionCall": {"name": name, "args": args or {}, **extra}}


def adapter(responses, **kw):
    transport = Transport(responses)
    return GeminiAdapter("test-key", "gemini-test", post=transport, sleep=lambda s: None, **kw), transport


def test_tool_specs_are_translated_to_function_declarations():
    declarations = {d["name"]: d for d in (_to_gemini_tool(t) for t in TOOL_SPECS)}
    seg = declarations["segment_breakdown"]["parameters"]
    assert seg["type"] == "object" and seg["required"] == ["dimension"]
    assert seg["properties"]["dimension"]["enum"] == ["region", "category"]
    # Tools with no arguments must omit `parameters` rather than send an empty schema.
    assert "parameters" not in declarations["marketing_effect"]
    assert "required" not in declarations["baseline_trend"]["parameters"]


def test_request_shape_and_credentials():
    a, transport = adapter([reply({"text": "hello"})])
    a.start("SYSTEM", TOOL_SPECS, "the question")
    a.next_turn()
    call = transport.calls[0]
    assert call["url"].endswith("/models/gemini-test:generateContent")
    assert call["headers"]["x-goog-api-key"] == "test-key"
    assert "test-key" not in call["url"]
    assert call["payload"]["system_instruction"]["parts"][0]["text"] == "SYSTEM"
    assert len(call["payload"]["tools"][0]["function_declarations"]) == len(TOOL_SPECS)
    assert call["payload"]["contents"][0] == {"role": "user", "parts": [{"text": "the question"}]}


def test_text_and_function_calls_are_parsed_and_thoughts_skipped():
    a, _ = adapter([reply(
        {"text": "private reasoning", "thought": True},
        {"text": "Checking the trend."},
        fc("segment_breakdown", {"dimension": "region"}),
    )])
    a.start("s", TOOL_SPECS, "q")
    turn = a.next_turn()
    assert turn.texts == ["Checking the trend."]
    assert [(c.name, c.args) for c in turn.tool_calls] == [("segment_breakdown", {"dimension": "region"})]


def test_model_turn_is_echoed_back_verbatim_and_results_follow():
    call_part = {**fc("baseline_trend"), "thoughtSignature": "opaque-token"}
    a, transport = adapter([reply(call_part), reply({"text": "done"})])
    a.start("s", TOOL_SPECS, "q")
    turn = a.next_turn()
    a.add_results([ToolResult(turn.tool_calls[0].id, "baseline_trend", json.dumps([{"id": "x"}]))])
    a.next_turn()
    sent = transport.calls[1]["payload"]["contents"]
    assert sent[1] == {"role": "model", "parts": [call_part]}                 # signature preserved
    assert sent[2]["role"] == "user"
    part = sent[2]["parts"][0]["functionResponse"]
    assert part["name"] == "baseline_trend" and part["response"] == {"result": [{"id": "x"}]}


def test_tool_errors_are_reported_as_errors():
    a, transport = adapter([reply(fc("segment_breakdown", {"dimension": "colour"})), reply({"text": "ok"})])
    a.start("s", TOOL_SPECS, "q")
    turn = a.next_turn()
    a.add_results([ToolResult(turn.tool_calls[0].id, turn.tool_calls[0].name, "dimension must be one of", is_error=True)])
    a.next_turn()
    sent = transport.calls[1]["payload"]["contents"][-1]["parts"][0]["functionResponse"]
    assert sent == {"name": "segment_breakdown", "response": {"error": "dimension must be one of"}}


def test_rate_limits_are_retried_then_surface():
    a, transport = adapter([LLMError("HTTP 429: quota", status=429), reply({"text": "ok"})])
    a.start("s", TOOL_SPECS, "q")
    assert a.next_turn().texts == ["ok"]
    assert len(transport.calls) == 2

    a, transport = adapter([LLMError("HTTP 429: quota", status=429)] * 3)
    a.start("s", TOOL_SPECS, "q")
    with pytest.raises(LLMError):
        a.next_turn()
    assert len(transport.calls) == 3                                           # first try + two retries


def test_non_retryable_errors_are_not_retried():
    a, transport = adapter([LLMError("HTTP 400: bad request", status=400)])
    a.start("s", TOOL_SPECS, "q")
    with pytest.raises(LLMError):
        a.next_turn()
    assert len(transport.calls) == 1


def test_blocked_or_empty_responses_raise():
    a, _ = adapter([{"promptFeedback": {"blockReason": "SAFETY"}}])
    a.start("s", TOOL_SPECS, "q")
    with pytest.raises(LLMError, match="SAFETY"):
        a.next_turn()


def test_full_investigation_through_the_gemini_adapter(toolkit):
    responses = [
        reply({"text": "Confirming the trend."}, fc("baseline_trend", {"metric": "revenue"})),
        reply({"text": "Revenue is down 23.9%."}, fc("segment_breakdown", {"dimension": "region"})),
        reply({"text": "North stands out."}, fc("segment_breakdown", {"dimension": "category"})),
        reply({"text": "Electronics stands out."}, fc("marketing_effect")),
        reply({"text": "Marketing in North is supported."}, fc("price_effect")),
        reply({"text": "Revenue is down 23.9%. North and Electronics carry the change."}),
    ]
    a, transport = adapter(responses)
    events = list(investigate("why", toolkit, client=a))
    inv = events[-1].data["investigation"]
    assert inv.mode == "live" and inv.provider == "gemini" and inv.model == "gemini-test"
    assert [s["tool"] for s in inv.steps] == [t for t, _ in STANDARD_PLAN]
    assert inv.summary_source == "llm" and not inv.guardrail_blocks
    assert events[0].data["display"] == "Gemini"


def test_quota_failure_mid_run_falls_back_to_offline(toolkit):
    responses = [
        reply(fc("baseline_trend")),
        LLMError("HTTP 429: quota exceeded", status=429), LLMError("HTTP 429", status=429), LLMError("HTTP 429", status=429),
    ]
    a, _ = adapter(responses)
    events = list(investigate("why", toolkit, client=a))
    inv = events[-1].data["investigation"]
    assert any("429" in n for n in inv.notes)
    assert [s["tool"] for s in inv.steps] == [t for t, _ in STANDARD_PLAN]


def test_guardrail_applies_to_gemini_prose_too(toolkit):
    responses = [
        reply({"text": "Revenue is down a staggering 62%."}, fc("baseline_trend")),
        reply({"text": "Fine."}),
    ]
    a, _ = adapter(responses)
    inv = list(investigate("why", toolkit, client=a))[-1].data["investigation"]
    assert inv.guardrail_blocks and "62" in inv.guardrail_blocks[0]["unsupported"]


def test_make_client_selection(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "KYUYAAR_PROVIDER", "KYUYAAR_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert make_client() is None

    monkeypatch.setenv("GEMINI_API_KEY", "g")
    c = make_client()
    assert c.provider == "gemini" and c.model == GEMINI_DEFAULT_MODEL

    monkeypatch.setenv("KYUYAAR_MODEL", "gemini-custom")
    assert make_client().model == "gemini-custom"

    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "g2")
    assert make_client().provider == "gemini"

    monkeypatch.setenv("KYUYAAR_PROVIDER", "anthropic")
    assert make_client() is None                       # forced provider has no key


def test_api_key_never_appears_in_error_text():
    a, _ = adapter([LLMError("HTTP 403: API key not valid", status=403)])
    a.start("s", TOOL_SPECS, "q")
    with pytest.raises(LLMError) as info:
        a.next_turn()
    assert "test-key" not in str(info.value)
