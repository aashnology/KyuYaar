"""
Investigation orchestrator -- Layer 2.

The model decides which evidence tool to call next and writes short readouts
of what came back. It never computes a number: tools do that, and the
guardrail checks every figure in the model's prose against the evidence
before the prose is shown. If a check fails, deterministic template text is
used in its place.

The same event stream is produced with or without an LLM. With no client, a
fixed plan runs the same tools and the templates narrate, so the app stays
usable without an API key and the demo cannot be broken by a network problem.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Iterator

from evidence import Evidence
from guardrail import check_text
from narration import build_summary, narrate_step
from toolkit import Toolkit, ToolError, evidence_to_payload

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TURNS = 10

# The standard investigation. In live mode the model is expected to cover all
# of it; anything it skips is filled in afterwards and flagged.
STANDARD_PLAN = [
    ("baseline_trend", {"metric": "revenue"}),
    ("segment_breakdown", {"dimension": "region", "metric": "revenue"}),
    ("segment_breakdown", {"dimension": "category", "metric": "revenue"}),
    ("marketing_effect", {}),
    ("price_effect", {}),
]

OFFLINE_PLAN_TEXT = (
    "Standard investigation: confirm the metric really moved, find where the "
    "change is concentrated, then test marketing spend and pricing as candidate causes."
)

SYSTEM_PROMPT = """You are the investigation engine inside KyuYaar, a tool that helps a small \
business understand why a metric changed. You work with four deterministic tools that return \
Evidence objects. You never compute numbers yourself.

Work in this order:
1. baseline_trend to confirm the metric actually moved.
2. segment_breakdown on region and on category to see where the change is concentrated.
3. marketing_effect and price_effect to test candidate causes.

Rules:
- Quote figures exactly as the tool results give them. Do not introduce any other numerals: no \
sums, averages, ratios or recalculated percentages. If a figure is not in a tool result, do not state it.
- Between tool calls, write one or two sentences on what the latest result shows. State the \
strength honestly and mention the key caveat. Where evidence is weak, say the data does not \
support that explanation.
- These are associations and statistical comparisons, not proof. Never write "caused"; use \
wording like "accompanies" or "is consistent with".
- Do not recommend actions. Decision options are produced separately.
- When you have finished, write a final summary of at most 120 words: what changed, where it is \
concentrated, which explanations the evidence supports, which it does not, and what remains unknown."""

QUESTION_TEMPLATE = (
    "Business question: {question}\n\nInvestigate using the tools, then summarise."
)


@dataclass
class Event:
    kind: str          # start | plan | tool_call | evidence | narration | guardrail | coverage | note | summary | done
    data: dict = field(default_factory=dict)


@dataclass
class Investigation:
    question: str
    mode: str                              # "live" | "offline"
    model: str | None
    evidence: list[Evidence]
    steps: list[dict]
    summary: str
    summary_source: str                    # "llm" | "template"
    guardrail_blocks: list[dict]
    notes: list[str]


class _State:
    def __init__(self, question, mode, model):
        self.question, self.mode, self.model = question, mode, model
        self.evidence: dict[str, Evidence] = {}
        self.steps: list[dict] = []
        self.step_evidence: list[list[Evidence]] = []
        self.calls: set[tuple] = set()
        self.blocks: list[dict] = []
        self.notes: list[str] = []
        self.final_text: str | None = None
        self.filled_after_model = False

    def all_evidence(self):
        return list(self.evidence.values())


def make_client():
    """Anthropic client if a key and the SDK are available, else None."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic()


def _call_key(name, provenance_args):
    return (name, json.dumps(provenance_args, sort_keys=True))


def _do_tool(state, toolkit, name, args):
    """Run one tool, record it, and emit the events for it. Returns the evidence."""
    evs = toolkit.run(name, args)
    prov_args = evs[0].details["provenance"]["args"] if evs else (args or {})
    state.calls.add(_call_key(name, prov_args))
    step_no = len(state.steps) + 1

    yield Event("tool_call", {"step": step_no, "tool": name, "args": prov_args})
    for ev in evs:
        state.evidence[ev.id] = ev
    state.steps.append({
        "step": step_no, "tool": name, "args": prov_args,
        "evidence_ids": [e.id for e in evs],
        "narration": None, "narration_source": None,
    })
    state.step_evidence.append(evs)
    yield Event("evidence", {"step": step_no, "tool": name, "args": prov_args, "items": evs})
    return evs


def _template_narration(state, step_index):
    step = state.steps[step_index]
    return narrate_step(step["tool"], step["args"], state.step_evidence[step_index])


def _handle_model_text(state, text):
    """Validate a piece of model prose written between tool calls."""
    if not state.steps:
        # Written before any tool has run, so there are no figures it could cite.
        if check_text(text, []).ok:
            yield Event("plan", {"text": text, "source": "llm"})
        else:
            yield from _block(state, text, None)
        return

    idx = len(state.steps) - 1
    result = check_text(text, state.all_evidence())
    if result.ok:
        state.steps[idx]["narration"], state.steps[idx]["narration_source"] = text, "llm"
        yield Event("narration", {"step": idx + 1, "text": text, "source": "llm"})
    else:
        yield from _block(state, text, idx, result.unsupported)


def _block(state, text, step_index, unsupported=None):
    state.blocks.append({"text": text, "unsupported": unsupported or [], "step": step_index})
    yield Event("guardrail", {"text": text, "unsupported": unsupported or [], "step": step_index})
    if step_index is not None:
        fallback = _template_narration(state, step_index)
        state.steps[step_index]["narration"] = fallback
        state.steps[step_index]["narration_source"] = "template"
        yield Event("narration", {"step": step_index + 1, "text": fallback, "source": "template"})


def _live_loop(state, toolkit, client, model, max_turns):
    messages = [{"role": "user", "content": QUESTION_TEMPLATE.format(question=state.question)}]
    for _ in range(max_turns):
        try:
            resp = client.messages.create(
                model=model, max_tokens=1500, system=SYSTEM_PROMPT,
                tools=toolkit.specs(), messages=messages,
            )
        except Exception as exc:  # network, auth, rate limit: degrade rather than fail
            note = f"LLM call failed ({type(exc).__name__}); finishing the investigation in offline mode."
            state.notes.append(note)
            yield Event("note", {"text": note})
            return

        messages.append({"role": "assistant", "content": resp.content})
        texts = [b.text for b in resp.content if b.type == "text" and b.text.strip()]
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        for text in texts:
            if tool_uses:
                yield from _handle_model_text(state, text)
            else:
                state.final_text = text
        if not tool_uses:
            return

        results = []
        for tu in tool_uses:
            try:
                evs = yield from _do_tool(state, toolkit, tu.name, tu.input)
                content = json.dumps([evidence_to_payload(e) for e in evs])
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content})
            except ToolError as exc:
                results.append({
                    "type": "tool_result", "tool_use_id": tu.id,
                    "content": str(exc), "is_error": True,
                })
        messages.append({"role": "user", "content": results})


def _fill_coverage(state, toolkit, live):
    for name, args in STANDARD_PLAN:
        if _call_key(name, args) in state.calls:
            continue
        if live:
            state.filled_after_model = True
            msg = f"The model did not run {name}{args or ''}; it was run to complete the standard investigation."
            state.notes.append(msg)
            yield Event("coverage", {"text": msg})
        yield from _do_tool(state, toolkit, name, args)
        idx = len(state.steps) - 1
        text = _template_narration(state, idx)
        state.steps[idx]["narration"], state.steps[idx]["narration_source"] = text, "template"
        yield Event("narration", {"step": idx + 1, "text": text, "source": "template"})


def investigate(question, toolkit: Toolkit, client=None, model=None,
                max_turns=MAX_TURNS) -> Iterator[Event]:
    """Run an investigation, yielding events as it goes. The last event is
    'done' and carries the finished Investigation."""
    model = model or os.environ.get("KYUYAAR_MODEL", DEFAULT_MODEL)
    live = client is not None
    state = _State(question, "live" if live else "offline", model if live else None)
    yield Event("start", {"question": question, "mode": state.mode, "model": state.model})

    if live:
        yield from _live_loop(state, toolkit, client, model, max_turns)
    else:
        yield Event("plan", {"text": OFFLINE_PLAN_TEXT, "source": "template"})

    yield from _fill_coverage(state, toolkit, live)

    evidence = state.all_evidence()
    template_summary = build_summary(evidence)
    summary, source = template_summary, "template"
    if live and state.final_text and not state.filled_after_model:
        result = check_text(state.final_text, evidence)
        if result.ok:
            summary, source = state.final_text, "llm"
        else:
            state.blocks.append({"text": state.final_text, "unsupported": result.unsupported, "step": None})
            yield Event("guardrail", {"text": state.final_text, "unsupported": result.unsupported, "step": None})
    yield Event("summary", {"text": summary, "source": source})

    yield Event("done", {"investigation": Investigation(
        question=question, mode=state.mode, model=state.model, evidence=evidence,
        steps=state.steps, summary=summary, summary_source=source,
        guardrail_blocks=state.blocks, notes=state.notes,
    )})
