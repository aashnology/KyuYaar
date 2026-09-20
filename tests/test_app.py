"""End-to-end click-through of the Streamlit app in offline mode."""

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

os.environ["KYUYAAR_PACE"] = "0"
os.environ.pop("ANTHROPIC_API_KEY", None)


def click(at, prefix):
    button = next(b for b in at.button if b.label.startswith(prefix))
    return button.click().run()


def test_full_flow_reaches_a_recorded_decision():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any(b.label.startswith("Investigate") for b in at.button)

    at = click(at, "Investigate")
    assert not at.exception
    assert any("evidence chain is ready" in s.value for s in at.success)

    at = click(at, "See the evidence")
    assert not at.exception
    assert [s.value for s in at.subheader] == [
        "What changed", "Fewer orders, or smaller orders?",
        "Where the change is concentrated", "Candidate causes tested",
    ]

    at = click(at, "Continue to the decision")
    assert not at.exception
    choose = [b for b in at.button if b.label == "Choose this option"]
    assert len(choose) >= 3

    choose[0].click().run()
    assert not at.exception
    assert any("You chose this option" in s.value for s in at.success)


def test_later_screens_are_locked_until_an_investigation_exists():
    at = AppTest.from_file(APP, default_timeout=120).run()
    locked = {b.label: b.disabled for b in at.button if b.label[0] in "234"}
    assert all(locked.values())
