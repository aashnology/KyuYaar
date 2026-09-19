from guardrail import check_text
from evidence import Evidence


def make(**kw):
    base = dict(
        id="x", hypothesis="North orders moved -39% (p<0.001) on 220 orders",
        evidence_type="statistical", metric="m", value=-38.6, baseline=0.0,
        segment="region=North", strength="strong", sample_size=220,
        caveats=[], details={"p_value": 1.59e-07, "p_value_text": "<0.001", "spend_prior": 32550.0},
    )
    base.update(kw)
    return Evidence(**base)


def test_rounded_figures_pass():
    ev = [make()]
    assert check_text("Orders fell 39% across 220 orders.", ev).ok
    assert check_text("Orders fell 38.6% in North.", ev).ok


def test_sign_is_ignored_but_magnitude_is_not():
    ev = [make()]
    assert check_text("A decline of -39%.", ev).ok
    assert not check_text("A decline of 41%.", ev).ok


def test_invented_and_derived_numbers_are_caught():
    ev = [make()]
    res = check_text("Orders fell 39%, roughly 4x worse than the rest.", ev)
    assert not res.ok and "4" in res.unsupported
    assert not check_text("That is 250 orders.", ev).ok


def test_precision_matters():
    ev = [make()]
    assert not check_text("Orders fell 38.9%.", ev).ok


def test_thousands_separators_and_scientific_notation():
    ev = [make()]
    assert check_text("Spend was 32,550 before the cut.", ev).ok
    assert check_text("p = 1.59e-07", ev).ok


def test_text_without_numbers_passes_with_no_evidence():
    assert check_text("Let me check the overall trend first.", []).ok
    assert not check_text("Revenue is down 20%.", []).ok
