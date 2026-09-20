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
from llm import LLMError, ToolResult, as_adapter, make_client  # noqa: F401  (make_client re-exported)
from narration import build_summary, narrate_step
from toolkit import Toolkit, ToolError, evidence_to_payload

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
- You may call several independent tools in the same turn, and you should: after baseline_trend, \
run segment_breakdown for region and for category together, then marketing_effect and price_effect \
together. Fewer, larger turns keep the investigation fast.
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
    provider: str | None
    evidence: list[Evidence]
    steps: list[dict]
    summary: str
    summary_source: str                    # "llm" | "template"
    guardrail_blocks: list[dict]
    notes: list[str]


class _State:
    def __init__(self, question, mode, model):
        self.question, self.mode, self.model = question, mode, model
        self.provider = None
        self.evidence: dict[str, Evidence] = {}
        self.steps: list[dict] = []
        self.step_evidence: list[list[Evidence]] = []
        self.calls: set[tuple] = set()
        self.blocks: list[dict] = []
        self.notes: list[str] = []
        self.final_text: str | None = None
        self.filled_after_model = False
        self.call_failed = False

    def all_evidence(self):
        return list(self.evidence.values())


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
        result = check_text(text, [])
        if result.ok:
            yield Event("plan", {"text": text, "source": "llm"})
        else:
            yield from _block(state, text, None, result.unsupported)
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


def _drain_notices(state, adapter):
    drain = getattr(adapter, "drain_notices", None)
    for text in (drain() if drain else []):
        state.notes.append(text)
        yield Event("note", {"text": text})


def _live_loop(state, toolkit, adapter, max_turns):
    adapter.start(SYSTEM_PROMPT, toolkit.specs(), QUESTION_TEMPLATE.format(question=state.question))
    for _ in range(max_turns):
        try:
            turn = adapter.next_turn()
            yield from _drain_notices(state, adapter)
        except Exception as exc:  # network, auth, quota: degrade rather than fail
            yield from _drain_notices(state, adapter)
            state.call_failed = True
            detail = f": {exc}" if isinstance(exc, LLMError) else ""
            note = (
                f"The model call failed ({type(exc).__name__}{detail}); "
                f"finishing the investigation in offline mode."
            )
            state.notes.append(note)
            yield Event("note", {"text": note})
            return

        for text in turn.texts:
            if turn.tool_calls:
                yield from _handle_model_text(state, text)
            else:
                state.final_text = text
        if not turn.tool_calls:
            return

        results = []
        for call in turn.tool_calls:
            try:
                evs = yield from _do_tool(state, toolkit, call.name, call.args)
                content = json.dumps([evidence_to_payload(e) for e in evs])
                results.append(ToolResult(call.id, call.name, content))
            except ToolError as exc:
                results.append(ToolResult(call.id, call.name, str(exc), is_error=True))
        adapter.add_results(results)


def _fill_coverage(state, toolkit, live):
    # After an API failure the model never had the chance to skip anything, so
    # the remaining steps are simply the offline plan, not a "coverage gap".
    announce = live and not state.call_failed
    for name, args in STANDARD_PLAN:
        if _call_key(name, args) in state.calls:
            continue
        if announce:
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
    adapter = as_adapter(client, model)
    live = adapter is not None
    state = _State(question, "live" if live else "offline", adapter.model if live else None)
    state.provider = adapter.provider if live else None
    yield Event("start", {
        "question": question, "mode": state.mode, "model": state.model,
        "provider": state.provider, "display": adapter.display if live else None,
    })

    if live:
        yield from _live_loop(state, toolkit, adapter, max_turns)
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
        question=question, mode=state.mode, model=state.model, provider=state.provider,
        evidence=evidence,
        steps=state.steps, summary=summary, summary_source=source,
        guardrail_blocks=state.blocks, notes=state.notes,
    )})
