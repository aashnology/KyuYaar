"""Decision log: what is stored, that it survives a restart, and that no outcome is ever invented."""

import json
from datetime import datetime, timezone

import pytest

from decision_log import AWAITING, RECORDED, DecisionLog, record_from_decision
from decisions import build_options
from orchestrator import investigate
from scenario import Assumptions, run_scenario

NOW = datetime(2026, 9, 21, 9, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def inv(toolkit):
    return list(investigate("Revenue dropped. Why?", toolkit))[-1].data["investigation"]


@pytest.fixture(scope="module")
def ds(inv):
    return build_options(inv.evidence)


def option(ds, option_id):
    return next(o for o in ds.options if o.id == option_id)


def test_record_captures_the_decision_its_evidence_and_the_projection(inv, ds, orders):
    a = Assumptions(recovery_share=0.6, lag_months=2, horizon_months=6)
    sc = run_scenario(option(ds, "restore_marketing_North"), inv.evidence, orders, a)
    rec = record_from_decision(inv, ds, "restore_marketing_North", "  Try it for a quarter. ", sc,
                               now=NOW, record_id="abc123")
    assert rec.record_id == "abc123" and rec.recorded_at == "2026-09-21T09:30:00+00:00"
    assert rec.option_kind == "act" and rec.confidence == "strong" and rec.note == "Try it for a quarter."
    assert rec.data_period == "2026-08" and rec.question == inv.question
    assert rec.assumptions == {"recovery_share": 0.6, "lag_months": 2, "horizon_months": 6, "test_share": 0.25}
    assert rec.projection["revenue_total"] == sc.revenue_total
    assert rec.projection["gross_profit_total"] == sc.gross_profit_total
    assert rec.projection["break_even_share"] == sc.break_even_share
    ids = {e["id"] for e in rec.evidence}
    assert ids == {"stat_marketing_North", "obs_baseline_trend"}
    snap = next(e for e in rec.evidence if e["id"] == "stat_marketing_North")
    assert snap["strength"] == "strong" and snap["hypothesis"].startswith("In North")


def test_a_new_record_has_an_empty_outcome_and_waits_for_one(inv, ds):
    rec = record_from_decision(inv, ds, "hold_and_monitor")
    assert rec.outcome is None and rec.status == AWAITING


def test_an_option_the_data_cannot_size_is_stored_without_a_projection(inv, ds, orders):
    sc = run_scenario(option(ds, "sequence_fixes"), inv.evidence, orders)
    assert not sc.projectable
    rec = record_from_decision(inv, ds, "sequence_fixes", scenario=sc)
    assert rec.projection is None and rec.assumptions      # assumptions are still recorded


def test_unknown_option_is_rejected(inv, ds):
    with pytest.raises(ValueError):
        record_from_decision(inv, ds, "make_it_better")


def test_records_survive_a_restart_and_come_back_intact(inv, ds, tmp_path):
    path = tmp_path / "log" / "decisions.jsonl"          # the directory does not exist yet
    DecisionLog(path).append(record_from_decision(inv, ds, "hold_and_monitor", record_id="r1"))
    DecisionLog(path).append(record_from_decision(inv, ds, "test_price_Electronics", record_id="r2"))
    again = DecisionLog(path)
    assert [r.record_id for r in again.all()] == ["r1", "r2"]
    assert again.get("r2").option_id == "test_price_Electronics"
    assert again.get("nope") is None
    assert all(json.loads(line) for line in path.read_text().splitlines())   # plain JSON lines


def test_missing_log_reads_as_empty(tmp_path):
    assert DecisionLog(tmp_path / "none.jsonl").all() == []


def test_path_can_be_set_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("KYUYAAR_DECISION_LOG", str(tmp_path / "elsewhere.jsonl"))
    assert DecisionLog().path == tmp_path / "elsewhere.jsonl"


def test_an_outcome_is_stored_only_as_supplied_and_only_once(inv, ds, tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl")
    log.append(record_from_decision(inv, ds, "hold_and_monitor", record_id="r1"))
    log.append(record_from_decision(inv, ds, "test_price_Electronics", record_id="r2"))
    assert [r.record_id for r in log.awaiting_outcome()] == ["r1", "r2"]

    updated = log.record_outcome("r1", "monthly_revenue", 431000.0, "2026-11", note="Recovered some.")
    assert updated.status == RECORDED
    assert updated.outcome == {"metric": "monthly_revenue", "observed_value": 431000.0,
                               "observed_period": "2026-11", "note": "Recovered some."}
    assert log.get("r1").outcome["observed_value"] == 431000.0
    assert log.get("r2").outcome is None and log.get("r2").status == AWAITING   # the other record is untouched
    assert [r.record_id for r in log.awaiting_outcome()] == ["r2"]

    with pytest.raises(ValueError):
        log.record_outcome("r1", "monthly_revenue", 1.0, "2026-12")
    with pytest.raises(KeyError):
        log.record_outcome("missing", "monthly_revenue", 1.0, "2026-12")
