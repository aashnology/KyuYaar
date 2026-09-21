"""The engine against scenarios other than the one it was built on.

Each scenario has a known ground truth (src/synthetic.py). A cause scenario must
be recovered, and a scenario with no cause must come out as "the data does not
support a cause" rather than a story.
"""

import dataclasses
import hashlib
import io

import pytest

from data_loader import load_data, scenario_dir
from decisions import build_options
from orchestrator import investigate
from synthetic import SCENARIOS, generate
from toolkit import Toolkit

QUESTION = "Revenue dropped last month. Why?"


def run_investigation(orders, marketing):
    events = list(investigate(QUESTION, Toolkit(orders, marketing)))
    return events[-1].data["investigation"]


def supported_causes(inv):
    return sorted(e.id for e in inv.evidence if e.evidence_type == "statistical" and e.strength != "weak")


@pytest.fixture(scope="module")
def committed():
    """Offline investigation of each committed scenario."""
    out = {}
    for name in SCENARIOS:
        orders, marketing = load_data(scenario_dir(name))
        out[name] = run_investigation(orders, marketing)
    return out


def test_generator_is_deterministic_and_the_committed_files_are_its_output():
    for name in SCENARIOS:
        tables = generate(name)
        for table, frame in tables.items():
            buf = io.StringIO()
            frame.to_csv(buf, index=False)
            on_disk = (scenario_dir(name) / f"{table}.csv").read_bytes()
            assert hashlib.md5(buf.getvalue().encode()).hexdigest() == hashlib.md5(on_disk).hexdigest(), (
                f"{name}/{table}.csv is out of date; run scripts/generate_data.py --all"
            )


@pytest.mark.parametrize("name", [n for n, s in SCENARIOS.items() if s.expected_causes])
def test_injected_causes_are_recovered_and_nothing_else_is_supported(committed, name):
    assert supported_causes(committed[name]) == sorted(SCENARIOS[name].expected_causes)


def test_channel_loss_is_placed_in_the_channel_not_the_whole_region(committed):
    inv = committed["channel_loss"]
    west_paid = next(e for e in inv.evidence if e.id == "chan_West_Paid")
    assert west_paid.strength == "strong"
    assert west_paid.details["channel_pattern"]["status"] == "concentrated"
    assert west_paid.details["efficiency"]["verdict"] == "unchanged"
    options = build_options(inv.evidence)
    assert not options.insufficient
    assert any(o.id == "restore_marketing_West" for o in options.options)


@pytest.mark.parametrize("name", ["demand_shock", "flat"])
def test_scenarios_with_no_cause_produce_no_supported_cause_and_no_recommendation(committed, name):
    inv = committed[name]
    assert supported_causes(inv) == []
    assert not [e for e in inv.evidence if e.evidence_type == "channel" and e.strength != "weak"]
    options = build_options(inv.evidence)
    assert options.insufficient
    assert [o.kind for o in options.options] == ["hold"]
    assert "does not support a specific cause" in inv.summary


def test_a_concentration_without_a_cause_is_not_stated_as_fact(committed):
    summary = committed["demand_shock"].summary
    assert "The change is concentrated in" not in summary
    assert "may not be a real concentration" in summary


@pytest.mark.parametrize("name,seed", [("demand_shock", 101), ("demand_shock", 102), ("flat", 101)])
def test_no_cause_holds_for_other_random_draws_too(name, seed):
    import pandas as pd
    from data_loader import enrich

    tables = generate(dataclasses.replace(SCENARIOS[name], seed=seed))
    tables["orders"]["order_date"] = pd.to_datetime(tables["orders"]["order_date"])
    tables["marketing"]["date"] = pd.to_datetime(tables["marketing"]["date"])
    orders, marketing = enrich(tables["orders"], tables["products"], tables["customers"], tables["marketing"])
    inv = run_investigation(orders, marketing)
    assert supported_causes(inv) == []
    assert build_options(inv.evidence).insufficient


def test_scenario_metadata_is_complete():
    for name, sc in SCENARIOS.items():
        assert sc.name == name and sc.title and sc.description
    assert SCENARIOS["default"].expected_causes == ("stat_marketing_North", "stat_price_Electronics")
