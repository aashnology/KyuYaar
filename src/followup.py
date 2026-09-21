"""
Follow-up questions over a finished investigation -- Layer 6.

A person can ask a free-text question about what the investigation found. The
answer is drawn from the Evidence objects the investigation already produced.
Nothing new is computed and no tool is run: a question the evidence cannot
answer gets a plain "this was not tested" instead of a guess.

Two paths, mirroring the orchestrator:

  live      one model call, no tools. The model gets the evidence as JSON and
            must cite the evidence ids behind its answer. Its prose goes
            through the same numeric guardrail as the investigation readouts,
            and an answer that cites an id that does not exist is discarded.
  offline   keyword retrieval over the evidence, answered with sentences built
            from the Evidence fields, so the numbers are correct by
            construction. This is also the fallback whenever the live path is
            rejected or fails.

Every answer lists the evidence it used, so the reader can open the finding
behind a sentence instead of trusting the wording.
"""

import json
import re
from dataclasses import dataclass, field

from guardrail import check_text, numbers_in
from llm import as_adapter
from toolkit import evidence_to_payload

MAX_QUESTION_CHARS = 600
MAX_FINDINGS = 4
_STRENGTH_ORDER = {"strong": 0, "moderate": 1, "weak": 2}
# Within a strength level, the finding that speaks most directly to the question comes first.
_TYPE_ORDER = {"observation": 0, "statistical": 1, "association": 2, "decomposition": 3, "channel": 4}
_OVERALL_TOPICS = {"aov", "orders", "trend"}
_STRENGTH_WORD = {"strong": "Strong", "moderate": "Moderate", "weak": "Weak"}

FOLLOWUP_SYSTEM = """You answer follow-up questions about one finished investigation inside \
KyuYaar. You are given the complete evidence from that investigation as JSON. You have no tools \
and cannot run any new analysis.

Rules:
- Use only the evidence provided. Quote figures exactly as they appear in it. Do not introduce \
any other numerals: no sums, averages, differences or recalculated percentages.
- Cite the evidence behind every claim with its id in square brackets, for example \
[stat_marketing_North]. Use only ids that appear in the evidence.
- Report strength honestly. A weak finding means the data does not support that explanation; \
say so instead of speculating.
- These are associations and statistical comparisons, not proof. Never write "caused"; use \
wording like "accompanies" or "is consistent with".
- If the evidence does not cover the question, say plainly that this investigation did not test \
it and name what it did test. Do not guess.
- Do not recommend actions. Decision options are shown separately on the Decision screen.
- Answer in at most 120 words."""

COVERED_TEXT = (
    "the overall revenue trend, whether the change is fewer or smaller orders, where it is "
    "concentrated (by region and by product category), and marketing spend (by region and by "
    "channel) and pricing as candidate causes"
)

_CITATION = re.compile(r"\[([A-Za-z0-9_|=.\-]+)\]")

_ADVICE = re.compile(
    r"\b(what should i do|what do i do|should i (act|cut|restore|raise|lower|roll)|recommend\w*|"
    r"next steps?|what to do|which option|options?)\b", re.I)

# Topic -> which evidence it points at. Matching is deliberately simple: a
# question that matches nothing is answered as "not tested", never guessed at.
_TOPICS = [
    ("aov", re.compile(r"\b(aov|basket|order (value|size)|average order|smaller|larger|bigger)\b", re.I)),
    ("orders", re.compile(r"\b(orders?|volume|fewer|count)\b", re.I)),
    ("channel", re.compile(r"\b(channels?|paid|organic|email|referral)\b", re.I)),
    ("marketing", re.compile(r"\b(marketing|spend\w*|ads?|advertis\w*|campaigns?)\b", re.I)),
    ("price", re.compile(r"\b(pric\w*|expensive|cheaper|discount\w*)\b", re.I)),
    ("where", re.compile(r"\b(where|which (region|categor\w*)|concentrat\w*|regions?|categor\w*)\b", re.I)),
    ("trend", re.compile(r"\b(revenue|sales|drop\w*|fell|fall|decline\w*|trend|baseline|how (much|big))\b", re.I)),
    ("why", re.compile(r"\b(why|cause\w*|reason\w*|explain\w*|driver\w*|responsible)\b", re.I)),
    ("confidence", re.compile(r"\b(sure|confiden\w*|certain\w*|reliable|trust\w*|strength|strong|weak|moderate)\b", re.I)),
]


@dataclass
class Answer:
    question: str
    text: str
    source: str                          # "llm" | "template"
    evidence_ids: list = field(default_factory=list)
    covered: bool = True                 # False when the evidence cannot answer the question
    notes: list = field(default_factory=list)
    blocked: list = field(default_factory=list)   # figures the guardrail rejected in a model answer


# -------------------------------------------------------------- template ---

def _segment_names(evidence):
    """Words that can appear in a question and point at a segment: North, Electronics, Paid ..."""
    names = {}
    for ev in evidence:
        if not ev.segment:
            continue
        for part in ev.segment.split("|"):
            if "=" in part:
                names.setdefault(part.split("=", 1)[1].lower(), part.split("=", 1)[1])
    return names


def _topic_match(topic, ev):
    if topic == "aov":
        return ev.evidence_type == "decomposition" and ev.metric != "orders_change_pct"
    if topic == "orders":
        return ev.evidence_type == "decomposition" and ev.metric == "orders_change_pct"
    if topic == "channel":
        return ev.evidence_type == "channel"
    if topic == "marketing":
        return ev.id.startswith("stat_marketing_") or ev.evidence_type == "channel"
    if topic == "price":
        return ev.id.startswith("stat_price_")
    if topic == "where":
        return ev.evidence_type == "association"
    if topic == "trend":
        return ev.evidence_type == "observation" or (
            ev.evidence_type == "decomposition" and ev.segment is None)
    if topic == "why":
        return ev.evidence_type in ("statistical", "channel", "association") and ev.strength != "weak"
    if topic == "confidence":
        return ev.strength != "weak"
    return False


def _describe(ev):
    """One sentence per finding, built only from the Evidence fields."""
    parts = [f"{_STRENGTH_WORD[ev.strength]} evidence: {ev.hypothesis}"]
    stats = []
    p = ev.details.get("p_value_text")
    if p:
        stats.append(f"p {p}")
    stats.append(f"{ev.sample_size} orders")
    text = f"{parts[0]} ({', '.join(stats)})."
    if ev.strength == "weak" and ev.evidence_type in ("statistical", "channel"):
        text += " That does not support it as an explanation."
    elif ev.caveats:
        cav = ev.caveats[0]
        text += " Caveat: " + cav[0].lower() + cav[1:] + "."
    return text


def _select(question, evidence):
    names = _segment_names(evidence)
    named = {n for key, n in names.items() if re.search(rf"\b{re.escape(key)}\b", question, re.I)}
    topics = [t for t, rx in _TOPICS if rx.search(question)]

    def has_name(ev):
        return bool(ev.segment) and any(
            p.split("=", 1)[1] in named for p in ev.segment.split("|") if "=" in p)

    by_name = [e for e in evidence if has_name(e)] if named else []
    by_topic = [e for e in evidence if any(_topic_match(t, e) for t in topics)] if topics else []
    if by_name and by_topic:
        both = [e for e in by_name if e in by_topic]
        return both or by_name
    if by_topic and not named and set(topics) <= _OVERALL_TOPICS:
        # "Fewer orders or smaller orders?" with no place named is about the business as a whole.
        # A question that also asks why, where or how sure needs the wider evidence.
        overall = [e for e in by_topic if e.segment is None]
        return overall or by_topic
    return by_name or by_topic


def template_answer(question, evidence) -> Answer:
    """Answer from the Evidence fields alone."""
    if _ADVICE.search(question):
        return Answer(
            question, "Recommendations are not made here. The Decision screen lists options built "
            "from the supported findings, each with its assumptions and risks, and leaves the "
            "choice to you.", "template", [], covered=False)

    picked = sorted(
        _select(question, evidence),
        key=lambda e: (_STRENGTH_ORDER[e.strength], _TYPE_ORDER[e.evidence_type], -abs(e.value or 0)),
    )
    if not picked:
        return Answer(
            question, "This investigation did not test that, so the evidence cannot answer it. "
            f"It covered {COVERED_TEXT}. A new investigation would be needed for anything else.",
            "template", [], covered=False)

    shown = picked[:MAX_FINDINGS]
    lines = [_describe(e) for e in shown]
    if len(picked) > len(shown):
        lines.append("Other matching findings are weaker or less relevant and are listed in full on the evidence screen.")
    if all(e.strength == "weak" for e in shown):
        lines.append("Nothing here is strong enough to support as an explanation.")
    lines.append("These are associations, not proof of cause.")
    return Answer(question, " ".join(lines), "template", [e.id for e in shown])


# ------------------------------------------------------------------- live ---

def _compact(ev):
    if ev.strength == "weak":
        return {
            "id": ev.id, "type": ev.evidence_type, "strength": ev.strength,
            "segment": ev.segment, "hypothesis": ev.hypothesis, "value": ev.value,
            "sample_size": ev.sample_size, "caveats": ev.caveats[:1],
        }
    payload = evidence_to_payload(ev)
    payload["details"] = {k: v for k, v in payload["details"].items() if k != "history_changes"}
    return payload


def _evidence_prompt(inv, question):
    body = json.dumps([_compact(e) for e in inv.evidence])
    return (
        f"Investigation question: {inv.question}\n\n"
        f"Evidence (JSON):\n{body}\n\n"
        f"Follow-up question: {question}"
    )


def _strip_citations(text):
    cleaned = _CITATION.sub("", text)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _live_answer(question, inv, adapter) -> Answer:
    adapter.start(FOLLOWUP_SYSTEM, [], _evidence_prompt(inv, question))
    turn = adapter.next_turn()
    raw = "\n".join(turn.texts).strip()
    if not raw:
        raise ValueError("the model returned no text")

    known = {e.id for e in inv.evidence}
    cited = list(dict.fromkeys(_CITATION.findall(raw)))
    unknown = [c for c in cited if c not in known]
    text = _strip_citations(raw)

    # Figures the person typed themselves may be repeated back.
    typed = {round(v, d) for v, d, _ in numbers_in(question)}
    check = check_text(text, inv.evidence)
    unsupported = [t for t in check.unsupported if not any(
        abs(float(t.replace(",", "").lstrip("+-\u2212")) - v) < 1e-9 for v in typed)]
    if unknown or unsupported:
        answer = template_answer(question, inv.evidence)
        answer.blocked = unsupported
        detail = []
        if unsupported:
            detail.append("figures not in the evidence (" + ", ".join(unsupported) + ")")
        if unknown:
            detail.append("evidence ids that do not exist (" + ", ".join(unknown) + ")")
        answer.notes.append(
            "The model's answer was discarded because it used " + " and ".join(detail)
            + ". The answer below is generated from the evidence instead.")
        return answer
    return Answer(question, text, "llm", cited, covered=bool(cited))


def answer_question(question, inv, client=None, model=None) -> Answer:
    """Answer one follow-up question from `inv.evidence`. Never raises for a
    model failure: it falls back to the template answer and says so."""
    question = " ".join((question or "").split())[:MAX_QUESTION_CHARS]
    if not question:
        return Answer(question, "Ask a question about the findings above.", "template", [], covered=False)

    adapter = as_adapter(client, model)
    if adapter is None:
        return template_answer(question, inv.evidence)
    try:
        return _live_answer(question, inv, adapter)
    except Exception as exc:  # network, quota, auth, empty reply: degrade, do not fail
        answer = template_answer(question, inv.evidence)
        answer.notes.append(
            f"The model call failed ({type(exc).__name__}); the answer is generated from the evidence directly.")
        return answer
