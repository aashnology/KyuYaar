"""
Decision log -- Layer 6.

The data structure for decision memory: what was decided, on what evidence,
under which assumptions, with what projection. Each record also has an empty
`outcome` slot. That slot is the point of the design and the part this
repository does not fill.

Closing the loop (did the projected recovery happen?) needs a later period of
data that the synthetic dataset does not have. Nothing here estimates,
simulates or back-fills an outcome. `record_outcome()` stores only figures a
person supplies once the real months exist, and no screen calls it yet. The
comparison of projected against observed is documented future work.

Storage is one JSON object per line in a local file, so a record survives a
restart and can be read without this code. Set KYUYAAR_DECISION_LOG to point
elsewhere; the default file is git-ignored.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO_ROOT / "decision_log" / "decisions.jsonl"
LOG_ENV = "KYUYAAR_DECISION_LOG"

AWAITING, RECORDED = "awaiting_outcome", "outcome_recorded"


@dataclass
class DecisionRecord:
    record_id: str
    recorded_at: str
    question: str
    data_period: str | None          # latest month of data the evidence covers
    option_id: str
    option_title: str
    option_kind: str                 # "act" | "test" | "hold"
    confidence: str                  # weakest evidence strength behind the option
    addresses: list                  # evidence ids the option rests on
    evidence: list                   # snapshot of those findings as they stood
    assumptions: dict                # the projection assumptions the person set
    projection: dict | None          # what was projected, or None if the option is not sized
    note: str = ""
    status: str = AWAITING
    outcome: dict | None = None      # supplied later by a person; never estimated


def _snapshot(ev):
    return {
        "id": ev.id, "type": ev.evidence_type, "strength": ev.strength, "segment": ev.segment,
        "hypothesis": ev.hypothesis, "value": ev.value, "sample_size": ev.sample_size,
    }


def record_from_decision(inv, decision_set, option_id, note="", scenario=None,
                         now=None, record_id=None) -> DecisionRecord:
    """Build the record for one chosen option from a finished investigation."""
    option = next((o for o in decision_set.options if o.id == option_id), None)
    if option is None:
        raise ValueError(f"unknown option {option_id!r}")

    wanted = set(option.addresses) | {e.id for e in inv.evidence if e.evidence_type == "observation"}
    evidence = [_snapshot(e) for e in inv.evidence if e.id in wanted]
    period = next((e.details.get("latest_period") for e in inv.evidence
                   if e.evidence_type == "observation"), None)

    projection = assumptions = None
    if scenario is not None:
        a = scenario.assumptions
        assumptions = {
            "recovery_share": a.recovery_share, "lag_months": a.lag_months,
            "horizon_months": a.horizon_months, "test_share": a.test_share,
        }
        if scenario.projectable:
            projection = {
                "revenue_total": scenario.revenue_total,
                "gross_profit_total": scenario.gross_profit_total,
                "break_even_share": scenario.break_even_share,
                "summary": scenario.summary,
            }

    stamp = (now or datetime.now(timezone.utc)).replace(microsecond=0).isoformat()
    return DecisionRecord(
        record_id=record_id or uuid.uuid4().hex[:10], recorded_at=stamp, question=inv.question,
        data_period=period, option_id=option.id, option_title=option.title, option_kind=option.kind,
        confidence=option.confidence, addresses=list(option.addresses), evidence=evidence,
        assumptions=assumptions or {}, projection=projection, note=note.strip(),
    )


class DecisionLog:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get(LOG_ENV) or DEFAULT_PATH)

    def all(self) -> list[DecisionRecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(DecisionRecord(**json.loads(line)))
        return records

    def get(self, record_id):
        return next((r for r in self.all() if r.record_id == record_id), None)

    def append(self, record: DecisionRecord) -> DecisionRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")
        return record

    def awaiting_outcome(self):
        return [r for r in self.all() if r.status == AWAITING]

    def record_outcome(self, record_id, metric, observed_value, observed_period, note=""):
        """Attach what actually happened, as reported by a person. Only a
        decision still awaiting an outcome can take one."""
        records = self.all()
        target = next((r for r in records if r.record_id == record_id), None)
        if target is None:
            raise KeyError(record_id)
        if target.status == RECORDED:
            raise ValueError(f"record {record_id} already has an outcome")
        target.outcome = {
            "metric": metric, "observed_value": observed_value,
            "observed_period": observed_period, "note": note.strip(),
        }
        target.status = RECORDED
        self.path.write_text("".join(json.dumps(asdict(r)) + "\n" for r in records), encoding="utf-8")
        return target
