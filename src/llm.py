"""
Model adapters for the orchestrator.

The orchestrator only needs three things from a model provider: start a
conversation, ask for the next turn (text plus tool calls), and hand tool
results back. Each provider has its own wire format, so each gets a small
adapter that speaks it. Everything above this module (tools, evidence,
guardrail, fallbacks) is provider-independent.

Gemini and Featherless AI (an OpenAI-compatible host for open-weight models) are
reached over their REST endpoints with the standard library only, so there is
no extra dependency and no SDK version to drift.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"
GEMINI_DEFAULT_MODEL = "gemini-flash-latest"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FEATHERLESS_DEFAULT_MODEL = "Qwen/Qwen3-32B"     # a family Featherless documents for native tool calling
FEATHERLESS_ENDPOINT = "https://api.featherless.ai/v1/chat/completions"


class LLMError(RuntimeError):
    """A provider call failed. The message is safe to show: it never contains credentials."""

    def __init__(self, message, status=None, retry_after=None, daily=False):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after      # seconds the provider asked us to wait
        self.daily = daily                  # a per-day quota: waiting a minute will not help


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str            # JSON text of the evidence, or an error message
    is_error: bool = False


@dataclass
class Turn:
    texts: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)


# ------------------------------------------------------------- Anthropic ---

class AnthropicAdapter:
    provider = "anthropic"
    display = "Claude"

    def __init__(self, client, model):
        self.client, self.model = client, model
        self.messages = []

    def start(self, system, tools, question):
        self.system, self.tools = system, tools
        self.messages = [{"role": "user", "content": question}]

    def next_turn(self) -> Turn:
        kwargs = {}
        if self.tools:          # a plain question-and-answer call has no tools to offer
            kwargs["tools"] = self.tools
        resp = self.client.messages.create(
            model=self.model, max_tokens=1500, system=self.system,
            messages=self.messages, **kwargs,
        )
        self.messages.append({"role": "assistant", "content": resp.content})
        return Turn(
            texts=[b.text for b in resp.content if b.type == "text" and b.text.strip()],
            tool_calls=[ToolCall(b.id, b.name, b.input) for b in resp.content if b.type == "tool_use"],
        )

    def add_results(self, results):
        blocks = []
        for r in results:
            block = {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content}
            if r.is_error:
                block["is_error"] = True
            blocks.append(block)
        self.messages.append({"role": "user", "content": blocks})


# ---------------------------------------------------------------- Gemini ---

def _seconds(value):
    """'23s' or '23.5s' -> 23.0; anything else -> None."""
    try:
        return float(str(value).rstrip("s"))
    except (TypeError, ValueError):
        return None


def gemini_error(status, body: bytes) -> LLMError:
    """Turn an HTTP error response into an LLMError, keeping the retry hints."""
    err = {}
    try:
        err = json.loads(body).get("error", {}) or {}
    except Exception:
        pass
    if not isinstance(err, dict):
        err = {"message": str(err)}
    message = (err.get("message") or "").split(" For more information")[0].strip()
    retry_after, daily = None, False
    for detail in err.get("details") or []:
        kind = str(detail.get("@type", ""))
        if kind.endswith("RetryInfo"):
            retry_after = _seconds(detail.get("retryDelay"))
        if kind.endswith("QuotaFailure"):
            for violation in detail.get("violations") or []:
                ident = f"{violation.get('quotaId', '')}{violation.get('quotaMetric', '')}"
                if "PerDay" in ident:
                    daily = True
    return LLMError(f"HTTP {status}: {message[:160]}".rstrip(": "), status=status,
                    retry_after=retry_after, daily=daily)


def _urllib_post(url, headers, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise gemini_error(exc.code, exc.read()) from None
    except urllib.error.URLError as exc:
        raise LLMError(f"network error: {exc.reason}") from None


def _to_gemini_tool(tool):
    """Translate an Anthropic-style tool spec into a Gemini function declaration."""
    declaration = {"name": tool["name"], "description": tool["description"]}
    schema = tool.get("input_schema", {})
    properties = schema.get("properties")
    if properties:
        params = {"type": "object", "properties": properties}
        if schema.get("required"):
            params["required"] = schema["required"]
        declaration["parameters"] = params
    return declaration


class GeminiAdapter:
    provider = "gemini"
    display = "Gemini"
    RETRY_STATUSES = (429, 503)
    MAX_WAIT = 70.0          # longest single wait; a per-minute quota clears within this

    def __init__(self, api_key, model, post=None, timeout=60, retry_delays=(5, 20, 40),
                 sleep=time.sleep, clock=time.monotonic, min_interval=None):
        self._api_key, self.model = api_key, model
        self._post = post or _urllib_post
        self._timeout, self._retry_delays = timeout, retry_delays
        self._sleep, self._clock = sleep, clock
        # Free tiers cap requests per minute; spacing calls avoids most 429s.
        if min_interval is None:
            min_interval = float(os.environ.get("KYUYAAR_MIN_INTERVAL", "4"))
        self._min_interval = min_interval
        self._last_call = None
        self.notices = []
        self.contents = []

    def drain_notices(self):
        notices, self.notices = self.notices, []
        return notices

    def _pace(self):
        if self._last_call is not None and self._min_interval:
            gap = self._min_interval - (self._clock() - self._last_call)
            if gap > 0:
                self._sleep(gap)
        self._last_call = self._clock()

    def start(self, system, tools, question):
        self._static = {"system_instruction": {"parts": [{"text": system}]}}
        if tools:               # Gemini rejects an empty function_declarations list
            self._static["tools"] = [{"function_declarations": [_to_gemini_tool(t) for t in tools]}]
        self.contents = [{"role": "user", "parts": [{"text": question}]}]

    def _request(self):
        url = GEMINI_ENDPOINT.format(model=self.model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}
        payload = {**self._static, "contents": self.contents}
        return self._send(url, headers, payload)

    def _send(self, url, headers, payload):
        delays = list(self._retry_delays)
        while True:
            self._pace()
            try:
                return self._post(url, headers, payload, self._timeout)
            except LLMError as exc:
                if exc.daily:
                    raise LLMError(
                        f"daily quota used up ({exc}); try again tomorrow or use another key",
                        status=exc.status,
                    ) from None
                if exc.status in self.RETRY_STATUSES and delays:
                    wait = min(exc.retry_after or delays.pop(0), self.MAX_WAIT)
                    if exc.retry_after:
                        delays.pop(0)
                    self.notices.append(f"Rate limited by the provider; waiting {wait:.0f}s before retrying.")
                    self._sleep(wait)
                    continue
                raise

    def next_turn(self) -> Turn:
        data = self._request()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
            raise LLMError(f"empty response ({reason})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        # The model's turn is echoed back exactly as received, which keeps any
        # provider-side reasoning state attached to the function calls.
        self.contents.append({"role": "model", "parts": parts})

        texts, calls = [], []
        for i, part in enumerate(parts):
            if part.get("thought"):
                continue
            if "text" in part and part["text"].strip():
                texts.append(part["text"])
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append(ToolCall(fc.get("id") or f"call_{len(self.contents)}_{i}", fc["name"], fc.get("args") or {}))
        return Turn(texts=texts, tool_calls=calls)

    def add_results(self, results):
        parts = []
        for r in results:
            if r.is_error:
                body = {"error": r.content}
            else:
                try:
                    body = {"result": json.loads(r.content)}
                except ValueError:
                    body = {"result": r.content}
            response = {"name": r.name, "response": body}
            if not r.call_id.startswith("call_"):
                response["id"] = r.call_id
            parts.append({"functionResponse": response})
        self.contents.append({"role": "user", "parts": parts})


# ----------------------------------------------------------- Featherless ---

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _to_openai_tool(tool):
    """Translate an Anthropic-style tool spec into an OpenAI-style function tool."""
    schema = tool.get("input_schema", {})
    params = {"type": "object", "properties": schema.get("properties") or {}}
    if schema.get("required"):
        params["required"] = schema["required"]
    return {"type": "function",
            "function": {"name": tool["name"], "description": tool["description"], "parameters": params}}


class FeatherlessAdapter(GeminiAdapter):
    """Featherless AI: open-weight models behind an OpenAI-style chat completions API.

    Retry, pacing and error handling are shared with the Gemini adapter; only the
    wire format differs.
    """

    provider = "featherless"
    display = "Featherless"

    def __init__(self, api_key, model, post=None, min_interval=None, **kw):
        if min_interval is None:
            min_interval = float(os.environ.get("KYUYAAR_MIN_INTERVAL", "0"))
        super().__init__(api_key, model, post=post, min_interval=min_interval, **kw)
        self.messages = []

    def start(self, system, tools, question):
        self._tools = [_to_openai_tool(t) for t in tools] if tools else []
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]

    def _request(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://github.com/aashnology/KyuYaar",
            "X-Title": "KyuYaar",
        }
        payload = {"model": self.model, "messages": self.messages, "max_tokens": 1500}
        if self._tools:
            payload["tools"] = self._tools
        return self._send(FEATHERLESS_ENDPOINT, headers, payload)

    def next_turn(self) -> Turn:
        data = self._request()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("empty response (no choices returned)")
        message = choices[0].get("message") or {}
        content = _THINK_BLOCK.sub("", message.get("content") or "").strip()

        calls, raw_calls = [], []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            call_id = tc.get("id") or f"call_{len(self.messages)}_{i}"
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            calls.append(ToolCall(call_id, fn.get("name", ""), args if isinstance(args, dict) else {}))
            raw_calls.append({"id": call_id, "type": "function",
                              "function": {"name": fn.get("name", ""), "arguments": fn.get("arguments") or "{}"}})

        entry = {"role": "assistant", "content": content}
        if raw_calls:
            entry["tool_calls"] = raw_calls
        self.messages.append(entry)
        return Turn(texts=[content] if content else [], tool_calls=calls)

    def add_results(self, results):
        for r in results:
            body = json.dumps({"error": r.content}) if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": body})


# --------------------------------------------------------------- factory ---

def make_client(provider=None, model=None):
    """Return an adapter for the first provider that has a key, else None.

    KYUYAAR_PROVIDER ("anthropic", "gemini" or "featherless") forces the choice;
    KYUYAAR_MODEL overrides the default model of whichever provider is used.
    Featherless is only picked automatically when no other provider has a key.
    """
    provider = (provider or os.environ.get("KYUYAAR_PROVIDER") or "").lower()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    featherless_key = os.environ.get("FEATHERLESS_API_KEY")
    model = model or os.environ.get("KYUYAAR_MODEL")

    if provider not in ("anthropic", "gemini", "featherless"):
        provider = ("anthropic" if anthropic_key else "gemini" if gemini_key
                    else "featherless" if featherless_key else "")

    if provider == "gemini" and gemini_key:
        return GeminiAdapter(gemini_key, model or GEMINI_DEFAULT_MODEL)
    if provider == "featherless" and featherless_key:
        return FeatherlessAdapter(featherless_key, model or FEATHERLESS_DEFAULT_MODEL)
    if provider == "anthropic" and anthropic_key:
        try:
            import anthropic
        except ImportError:
            return None
        return AnthropicAdapter(anthropic.Anthropic(), model or ANTHROPIC_DEFAULT_MODEL)
    return None


def as_adapter(client, model=None):
    """Accept an adapter, or a bare Anthropic-style client (used by the tests)."""
    if client is None or hasattr(client, "next_turn"):
        return client
    return AnthropicAdapter(client, model or os.environ.get("KYUYAAR_MODEL") or ANTHROPIC_DEFAULT_MODEL)
