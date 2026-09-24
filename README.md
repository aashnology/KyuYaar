# KyuYaar

AI that investigates before it recommends.

Most dashboards tell a business owner *what* changed — revenue dropped, orders fell, a channel underperformed. Almost none tell them *why*, and the tools that try tend to generate a confident-sounding explanation with no real evidence behind it.

KyuYaar is built for small businesses, student ventures, and small organizations that have operational data but no dedicated analyst. Given a question like "revenue dropped last month — why, and what should I do?", it runs a real investigation: checks the trend, breaks it down by segment, tests it against marketing and pricing, and builds an evidence chain — each hypothesis tagged with how strong the supporting evidence actually is. Where the data doesn't clearly support a conclusion, it says so rather than guessing.

**Core principle:** the LLM reasons and orchestrates; it never does the math. Every number shown to the user comes from a deterministic Python calculation traceable to a specific function in this repo. The model's job is to plan the investigation, interpret results, and communicate uncertainty honestly — not to produce statistics from a prompt.

KyuYaar doesn't decide anything on its own. It surfaces evidence-backed options, each with its assumptions and risks stated plainly, and leaves the call to the person who has to live with the outcome.

See [`DEMO.md`](DEMO.md) for screenshots and a full walkthrough of what the system finds when run.

## How an investigation works

```
question -> orchestrator -> tools -> Evidence objects -> guardrail -> UI -> decision options -> your choice
              (LLM)        (pandas)   (strength, caveats)  (number check)     (templates)
```

0. **`question.py`** first checks the question against the one supported investigation type (why revenue or orders changed). Anything else ("how many customers do we have?", a forecast, profit) is declined with a reason and an example, before any tool or model call.
1. **`baseline_trend`** confirms the metric actually moved.
2. **`aov_volume_decomposition`** splits a revenue change into fewer orders vs. smaller orders, overall and per segment.
3. **`segment_breakdown`** shows where the change is concentrated (region, category).
4. **`marketing_effect` / `price_effect`** test each candidate cause against a control group of segments whose driver didn't move.
5. **`marketing_channel_analysis`** repeats the marketing test per paid channel, and checks whether a loss is channel-specific or region-wide.
6. The **orchestrator** (LLM) chooses which tool to call next and writes short readouts; every figure it states is checked against the evidence (`src/guardrail.py`) and discarded/replaced if unsupported.
7. **`decisions.py`** maps supported (strong or moderate) causes to option templates — each with one or more assumptions, one or more risks, and an impact estimate. Weak evidence gets no option. `decisions.py` itself does not rank or choose between them; step 9 below does that as a separate annotation.
8. **`run_scenario()`** projects what a chosen option is worth under assumptions you set — plain arithmetic over the evidence, shown step by step, no model involved.
9. **`recommend()`** ranks the options that passed step 7 by their projected gross profit from step 8, adjusted for risk, and marks one "Recommended" — or none, if no option clears the evidence bar or none projects a gain. It ranks and annotates; it never changes an option's evidence, assumptions or risks.

### What "strong", "moderate" and "weak" mean

Each tool grades its own evidence with one of three labels, defined once in `src/strength.py` (`strong` is the original brief's High, `moderate` is Medium, `weak` is Low). Only strong and moderate evidence can support an option. Strength is how clearly the data supports a finding, not how important it is and not proof of cause.

It is graded on effect size, statistical certainty against a comparison group, and sample size — not a percentage threshold. A size-only rule ("over 15% is High") can't separate a cause from background movement here: in the final month every region fell 17.6–43.6% and four of five categories fell over 15%. The rule used instead singles out North (marketing) and Electronics (price) and leaves the other regions and categories weak.

- [`docs/EVIDENCE_STRENGTH.md`](docs/EVIDENCE_STRENGTH.md): the thresholds per tool, the rules shared by all of them, and how they are tested.
- [`docs/ORCHESTRATOR_SPEC.md`](docs/ORCHESTRATOR_SPEC.md): the orchestrator's rules as built, and where they differ from the original brief.
- Anything built on the label reads it through `src/strength.py`; `is_actionable()` raises on an unknown label, and a test fails if another module keeps its own copy of the scale.

The statistical methods themselves are described in the module docstrings under `src/`.

## Status

| Layer | What it adds | State |
|---|---|---|
| 1 | Synthetic dataset with a known root cause, `baseline_trend()`, `segment_breakdown()` | done |
| 2 | Cause-testing tools, LLM orchestrator, decision options, Streamlit UI | done, submittable end to end |
| 3 | `aov_volume_decomposition()`: fewer orders vs. smaller orders, checked against each cause's predicted pattern | done |
| 4 | `marketing_channel_analysis()`: which channel, region-wide or concentrated, less bought or less effective | done |
| 5 | `run_scenario()`: what each option is worth under stated assumptions, arithmetic shown step by step | done |
| 6 | Follow-up Q&A over the evidence, trend charts, downloadable report, decision log data structure | done; outcome-tracking loop documented as future work |
| 7 | Three more ground-truth scenarios (channel-only loss, unexplained demand fall, flat month) checked over many random draws; upload your own four CSVs with validation | done |
| — | Hardening before recommendation logic: one strength scale, the strength rule documented, end-to-end determinism proof, off-script questions declined | done |
| 8 | `recommend()`: ranks the actionable options by projected gross profit adjusted for risk, marks one "Recommended" (or none, if the evidence or the projections don't support it) | done |
| 9 | Real-data validation: `src/adapters/olist.py` maps the real Olist dataset onto the canonical schema; tools run unmodified | done — core evidence tools ran clean on real data; upload gate correctly rejected a real cross-dataset time gap; two tools crashed on real data's sparse tail (fixed in Layer 11, see `DEMO.md`) |
| 11 | Harden `effects.py` and `channels.py`: comparisons that cannot be made on sparse or messy data return "insufficient data" Evidence instead of raising; no thresholds changed | done; Olist rerun in `DEMO.md` |
| 13 | Uploaded region, category, channel and segment names treated as untrusted: 60-character cap, instruction-phrasing deny-list, explicit "literal labels" line in the system prompt | done; behavioral tests in `tests/test_untrusted_labels.py`, one live check in `scripts/check_live_injection.py` |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The investigation can run on a live model or fully offline:

| Setup | Provider |
|---|---|
| `GEMINI_API_KEY` set (free tier works) | Gemini, default `gemini-flash-latest` |
| `ANTHROPIC_API_KEY` set | Claude, default `claude-sonnet-5` |
| `FEATHERLESS_API_KEY` set (with `KYUYAAR_PROVIDER=featherless`, or on its own) | [Featherless AI](https://featherless.ai) open-weight models over its OpenAI-compatible API, default `Qwen/Qwen3-32B` |
| neither | offline: same tools, deterministic templates instead of model narration |

`KYUYAAR_PROVIDER` (`gemini`, `anthropic` or `featherless`) forces a choice when several keys are present; `KYUYAAR_MODEL` overrides the default model. A failed live call finishes the investigation offline and says so.

```bash
python scripts/check_live.py       # one real investigation; reports whether the model drove it
python -m pytest                   # full test suite
python scripts/generate_data.py --all   # rewrite every scenario's CSVs (seeded, reproducible)
```

Verification scripts per layer (`scripts/verify_layer1.py` through `verify_layer9.py`) check each layer's output against known ground truth — see `DEMO.md` for what they report. `verify_layer9.py` needs the real Olist CSVs, not committed to the repo (see `DEMO.md`).

**Bring your own data:** the app accepts four CSVs (customers, orders, products, marketing) and validates schema, values, and keys before running — see `src/validation.py` for the exact rules, and the app's own error messages for what to fix.

## Known limits

- One investigation type supported (why did revenue change); validated on real data in Layer 9 (Olist Brazilian E-Commerce + Marketing Funnel, ~100k real orders) as well as synthetic scenarios — see `DEMO.md` for what held up and what didn't.
- Where a cause test cannot be run (a marketing table that doesn't cover the latest months, no spend or like-for-like product to measure a change from, no orders on one side), the tool returns weak evidence flagged as insufficient data instead of raising. It is untested, not "tested and not supported". Tiny cells (one or two orders) still get a graded result under the small-sample rule, and `marketing_channel_analysis` reads a missing latest month as zero spend; see `DEMO.md`, Layer 11 and Known limits.
- Evidence is association, not proof of causation — every supported finding says so explicitly.
- Layer 5 (scenario) projections depend on an assumed recovery share the user sets; the data measures the drop, not how much of it comes back.
- Gross margin used in scenarios is product cost only — shipping, returns, fees, and labor aren't modeled, so projected gross profit is an upper bound.
- Follow-up questions have no memory of earlier ones in a session, and offline retrieval is keyword-based.
- The decision log is a local file and won't persist on a hosted deployment with an ephemeral disk.
- Customer-segment evidence is not yet exposed to the orchestrator (needs a significance test before it can be ranked reliably).
- The dataset is synthetic; real data has messier attribution, shorter history, and more than one thing changing at once.

Full limitations detail (with the specific figures behind each one) is in `DEMO.md`.
