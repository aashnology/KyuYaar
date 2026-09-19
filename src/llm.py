"""
Model adapters for the orchestrator.

The orchestrator only needs three things from a model provider: start a
conversation, ask for the next turn (text plus tool calls), and hand tool
results back. Each provider has its own wire format, so each gets a small
adapter that speaks it. Everything above this module (tools, evidence,
guardrail, fallbacks) is provider-independent.

Gemini is reached over its REST endpoint with the standard library only, so
there is no extra dependency and no SDK version to drift.
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"
GEMINI_DEFAULT_MODEL = "gemini-flash-latest"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    """A provider call failed. The message is safe to show: it never contains credentials."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


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
        resp = self.client.messages.create(
            model=self.model, max_tokens=1500, system=self.system,
            tools=self.tools, messages=self.messages,
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

def _urllib_post(url, headers, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read()).get("error", {}).get("message", "")
        except Exception:
            pass
        raise LLMError(f"HTTP {exc.code}: {detail[:200]}".rstrip(": "), status=exc.code) from None
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

    def __init__(self, api_key, model, post=None, timeout=60, retry_delays=(3, 8), sleep=time.sleep):
        self._api_key, self.model = api_key, model
        self._post = post or _urllib_post
        self._timeout, self._retry_delays, self._sleep = timeout, retry_delays, sleep
        self.contents = []

    def start(self, system, tools, question):
        self._static = {
            "system_instruction": {"parts": [{"text": system}]},
            "tools": [{"function_declarations": [_to_gemini_tool(t) for t in tools]}],
        }
        self.contents = [{"role": "user", "parts": [{"text": question}]}]

    def _request(self):
        url = GEMINI_ENDPOINT.format(model=self.model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}
        payload = {**self._static, "contents": self.contents}
        delays = list(self._retry_delays)
        while True:
            try:
                return self._post(url, headers, payload, self._timeout)
            except LLMError as exc:
                if exc.status in self.RETRY_STATUSES and delays:
                    self._sleep(delays.pop(0))
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


# --------------------------------------------------------------- factory ---

def make_client(provider=None, model=None):
    """Return an adapter for the first provider that has a key, else None.

    KYUYAAR_PROVIDER ("anthropic" or "gemini") forces the choice;
    KYUYAAR_MODEL overrides the default model of whichever provider is used.
    """
    provider = (provider or os.environ.get("KYUYAAR_PROVIDER") or "").lower()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    model = model or os.environ.get("KYUYAAR_MODEL")

    if provider not in ("anthropic", "gemini"):
        provider = "anthropic" if anthropic_key else "gemini" if gemini_key else ""

    if provider == "gemini" and gemini_key:
        return GeminiAdapter(gemini_key, model or GEMINI_DEFAULT_MODEL)
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
