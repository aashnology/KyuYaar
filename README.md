# KyuYaar
AI that investigates before it recommends.

Most dashboards can tell a business owner what changed — revenue dropped, orders fell, a channel underperformed. Almost none can tell them why and the tools that try (a generic "AI business assistant" bolted onto a chatbot) tend to generate a confident-sounding explanation with no real evidence behind it.

KyuYaar is very humble and built for small businesses, student ventures and small organizations that have operational data but no dedicated analyst. Given a question like "revenue dropped last month — why and what should I do?", it runs a real investigation: it checks the trend, breaks it down by customer segment and product, tests it against marketing spend and builds an evidence chain — each hypothesis tagged with how strong the supporting evidence actually is. Where the data doesn't clearly support a conclusion, it says so rather than guessing.

The core design principle: the LLM reasons and orchestrates the investigation; it never does the math. Every number shown to the user comes from a deterministic Python/DuckDB calculation that can be traced back to a specific query in this repo — the model's job is to plan the investigation, interpret results, and communicate uncertainty honestly, not to produce statistics from a prompt.

KyuYaar doesn't autonomously decide anything. It surfaces evidence-backed options, each with its assumptions and risks stated plainly and leaves the actual call to the person who has to live with the outcome.

## Status

| Layer | What it adds | State |
|---|---|---|
| 1 | Synthetic dataset with a known root cause, `baseline_trend()`, `segment_breakdown()` | done |
| 2 | Cause-testing tools, LLM orchestrator, decision options, Streamlit UI | done, submittable end to end |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The investigation can be driven by a live model or run offline:

| Setup | Provider |
|---|---|
| `GEMINI_API_KEY` set (free tier from Google AI Studio works) | Gemini, default model `gemini-flash-latest` |
| `ANTHROPIC_API_KEY` set | Claude, default model `claude-sonnet-5` |
| neither | offline: the same tools on a fixed plan, narrated by deterministic templates |

`KYUYAAR_PROVIDER` (`gemini` or `anthropic`) forces the choice when both keys are present, and `KYUYAAR_MODEL` overrides the default model. If a live call fails midway (quota, network), the investigation finishes in offline mode and says so. The tools, evidence, guardrail and decision options are identical across providers; only the model behind the plan and readouts changes.

```bash
python scripts/check_live.py      # one real investigation; reports whether the model drove it
```

```bash
python scripts/verify_layer1.py   # Layer 1 output on the dataset
python scripts/verify_layer2.py   # checks the investigation against the injected root cause
python -m pytest                  # full test suite
```

## How an investigation works

```
question -> orchestrator -> tools -> Evidence objects -> guardrail -> UI -> decision options -> your choice
              (LLM)        (pandas)   (strength, caveats)  (number check)     (templates)
```

1. **`baseline_trend`** confirms the metric actually moved.
2. **`segment_breakdown`** (region, category) shows where the change is concentrated.
3. **`marketing_effect`** and **`price_effect`** test each candidate cause, region by region and category by category. A segment is only a candidate if its driver (spend, price) moved away from the typical change, and its orders are compared with segments whose driver did not move, within strata of the other dimension so a category-wide effect is not mistaken for a regional one. Effects are pooled log rate ratios with a z-test.
4. The **orchestrator** lets the model choose which tool to call next and write short readouts. Every figure in that prose is checked against the evidence (`src/guardrail.py`); prose with an unsupported figure is discarded and replaced with text generated from the Evidence object.
5. **`decisions.py`** maps supported (strong or moderate) causes to option templates, each with assumptions, risks and an impact estimate computed from the evidence. Nothing is ranked or chosen for you.

Every Evidence object records the tool and arguments that produced it, and the UI shows this under "How this was computed".

## What was verified

On the synthetic dataset (a 45% cut to paid marketing in North and a 10% price rise on Electronics, both in the final month):

- North (marketing) and Electronics (price) come out strong; the other three regions and four categories come out weak.
- Across 81 hypotheses tested in earlier months with no injected cause, none was flagged.
- Layer 1 `segment_breakdown` previously ranked segments as strong even when total revenue barely moved. It now marks shares as unstable below a 10% total change.
- The orchestrator loop, guardrail and fallbacks are covered by tests using a scripted fake client.

## Known limits

- Both live provider paths (Gemini over REST, Claude via the SDK) are tested against scripted fake transports, not the real APIs, during development. Run `scripts/check_live.py` once with a key before relying on them.
- Evidence is association, not proof of cause. A driver and an order change in the same month cannot rule out another simultaneous change; every supported finding says so.
- The two effects overlap in North x Electronics and the data cannot separate their interaction. Options say so, and impact estimates for the two are not additive.
- One dataset, one investigation type (why did revenue change). Customer segment and channel are not exposed to the orchestrator: small segments need a significance test on the association itself before they can be ranked reliably.
- `baseline_trend` compares calendar-month totals, so a short month (February) can read as a drop. Per-day normalisation is not implemented.
- The dataset is synthetic. Effect sizes here are much cleaner than real data would be.
