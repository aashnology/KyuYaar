"""
The evidence-strength scale, defined once.

Every tool grades its own evidence with the rules documented in
docs/EVIDENCE_STRENGTH.md and returns one of these three labels. Anything that
reads a strength (the decision options, the narration, the follow-up answers,
the report, the UI, and any recommendation logic built on top) must import
from here rather than keep its own copy of the labels or of the actionable
threshold. A private copy can drift from what the tools actually produce, and a
gate wired to the wrong labels would build options on noise.
"""

# Weakest first, so an index is a level and "downgrade one level" is index - 1.
STRENGTH_ORDER = ("weak", "moderate", "strong")

# Sort key, strongest first.
RANK = {"strong": 0, "moderate": 1, "weak": 2}

WORD = {"strong": "Strong", "moderate": "Moderate", "weak": "Weak"}

# The names used in the original brief for the same three levels.
BRIEF_NAME = {"strong": "High", "moderate": "Medium", "weak": "Low"}

# Evidence at these levels may support a decision option. Weak evidence never does.
ACTIONABLE = ("strong", "moderate")


def label_of(evidence_or_label) -> str:
    label = getattr(evidence_or_label, "strength", evidence_or_label)
    if label not in STRENGTH_ORDER:
        raise ValueError(f"unknown strength label {label!r}; expected one of {STRENGTH_ORDER}")
    return label


def is_actionable(evidence_or_label) -> bool:
    """True for High/Medium (strong/moderate) evidence. Raises on an unknown label."""
    return label_of(evidence_or_label) in ACTIONABLE


def weakest(labels) -> str:
    """The lowest level among `labels`, or 'n/a' when there are none."""
    labels = [label_of(x) for x in labels]
    return min(labels, key=STRENGTH_ORDER.index) if labels else "n/a"
