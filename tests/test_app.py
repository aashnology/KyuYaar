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
        "Where the change is concentrated", "Candidate causes tested", "Marketing by channel",
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


def test_decision_screen_projects_each_option_from_the_persons_assumptions():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at = click(at, "Investigate")
    at = click(at, "See the evidence")
    at = click(at, "Continue to the decision")
    assert not at.exception

    share = at.slider(key="share_restore_marketing_North")
    revenue = lambda: next(m.value for m in at.metric if m.label.startswith("Revenue recovered"))
    before = revenue()
    share.set_value(100).run()
    assert not at.exception
    assert revenue() != before

    # The time frame is shared by every projection.
    at.slider(key="horizon_months").set_value(6).run()
    assert not at.exception
    assert any(m.label.endswith("6 mo") for m in at.metric)

    # Options the data cannot size say so instead of showing a number.
    assert any("Not projected" in i.value for i in at.info)

    # Choosing an option still works with the projections on screen.
    next(b for b in at.button if b.key == "choose_restore_marketing_North").click().run()
    assert not at.exception
    assert any("You chose this option" in s.value for s in at.success)
