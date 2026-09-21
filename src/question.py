"""
Is this question one the system can investigate?

This version supports exactly one investigation type: why a revenue or order
metric changed. That is a deliberately small allow-list, not language
understanding. A question is supported when it names a metric the tools can
analyse AND asks about a change in it. Anything else is declined with a reason,
so the system does not confidently run a revenue investigation against a
question about something else.

The check is plain keyword matching and is easy to get wrong at the edges. A
false decline costs the person a rephrase; a false accept is what this guards
against, so the rules lean towards declining.
"""

import re
from dataclasses import dataclass

REVENUE_CHANGE = "revenue_change"

_METRIC = re.compile(
    r"\b(revenues?|sales|income|turnover|takings|earnings|orders?|order (?:value|size|volume)|"
    r"aov|average order value)\b"
)
_CHANGE = re.compile(
    r"\b(drop\w*|declin\w*|fall\w*|fell|down|decreas\w*|dip\w*|slump\w*|shr[iu]nk\w*|shrank|"
    r"lower|less|worse|weaker|plung\w*|tank\w*|why|what happened|what went wrong|"
    r"what(?:'s| is) going on|lost|loss|hit)\b"
)
# Past-tense change: an explanation of something that already happened.
_PAST_CHANGE = re.compile(
    r"\b(dropped|fell|declined|decreased|dipped|slumped|plunged|shrank|shrunk|lost|"
    r"(?:is|are|was|were|went) down|lower than|less than)\b"
)
_FORECAST = re.compile(r"\b(forecast\w*|predict\w*|projection\w*|will|going to|expect\w*)\b")

# Topics the data cannot answer here, named so the decline can say so plainly.
_UNSUPPORTED_TOPICS = [
    (re.compile(r"\b(profit\w*|margins?|costs?|expens\w*)\b"), "profit, margin or cost"),
    (re.compile(r"\b(churn\w*|retention|subscribers?)\b"), "churn or retention"),
    (re.compile(r"\b(customers?|users?|visitors?|traffic|conversion)\b"), "customer or traffic counts"),
    (re.compile(r"\b(inventory|stock|staff|employees?)\b"), "inventory or staffing"),
]

_EXAMPLE = "Try something like: \"Revenue dropped last month. Why did it happen?\""


@dataclass(frozen=True)
class QuestionCheck:
    supported: bool
    kind: str | None
    reason: str


def classify_question(question: str) -> QuestionCheck:
    text = (question or "").strip().lower()
    if not text:
        return QuestionCheck(False, None, f"Type a question first. {_EXAMPLE}")

    has_metric = bool(_METRIC.search(text))
    has_change = bool(_CHANGE.search(text))

    if _FORECAST.search(text) and not _PAST_CHANGE.search(text):
        return QuestionCheck(
            False, None,
            "This doesn't match a supported investigation type yet. It explains why a metric "
            "already changed; it does not forecast what will happen. " + _EXAMPLE,
        )

    if has_metric and has_change:
        return QuestionCheck(True, REVENUE_CHANGE, "Investigating why revenue or orders changed.")

    for pattern, topic in _UNSUPPORTED_TOPICS:
        if pattern.search(text):
            return QuestionCheck(
                False, None,
                f"This doesn't match a supported investigation type yet. It covers why revenue "
                f"or orders changed, not {topic}. " + _EXAMPLE,
            )

    if has_metric:
        return QuestionCheck(
            False, None,
            "This doesn't match a supported investigation type yet. It can explain a change in "
            "revenue or orders, but the question doesn't ask about a change. " + _EXAMPLE,
        )
    return QuestionCheck(
        False, None,
        "This doesn't match a supported investigation type yet. It covers why revenue or "
        "orders changed. " + _EXAMPLE,
    )
