# Orchestrator specification

This is the specification the orchestrator was built to, updated to match what
is implemented. The wording the model actually receives is `SYSTEM_PROMPT` in
`src/orchestrator.py`. Where this differs from the original brief, the
difference is listed at the end.

## Purpose

A person asks a business question about their data, for example "why did
revenue drop last month, and what should I do?". The orchestrator plans and
narrates an investigation using deterministic tools. It does not answer from its
own reasoning about the numbers, and it never decides what the person should do.

## Rules

1. **No number that a tool did not return.** The model may not state a figure,
   percentage or statistic that is absent from the tool results. Every figure in
   model-written text is checked against the evidence (`src/guardrail.py`); text
   with an unsupported figure is discarded and replaced by text generated from
   the Evidence objects.
2. **Fixed tools, fixed plan.** The six tools are `baseline_trend`,
   `aov_volume_decomposition`, `segment_breakdown`, `marketing_effect`,
   `marketing_channel_analysis` and `price_effect`. They run in three rounds
   (confirm and split the change, locate it, test causes). The model chooses
   the order within a round, and anything it skips is run afterwards and
   reported (`STANDARD_PLAN`), so the evidence set is always complete.
3. **Strength labels come from the tools.** Each tool grades its own evidence
   with the rules in `docs/EVIDENCE_STRENGTH.md`. The model reports the label as
   returned and may not assign, upgrade or soften one.
4. **Weak evidence reads as weak.** No confident wording to compensate.
5. **The model never decides.** It presents decision options built from the
   evidence, each with **one or more assumptions and one or more risks**,
   and stops. The choice is the person's.
6. **Every finding is traceable.** Each Evidence object records the tool and
   arguments that produced it, shown under "How this was computed".
7. **Same data, same evidence.** The tools are pure calculations, so the same
   question on the same data gives identical evidence, plan and options
   (`tests/test_determinism.py`, including across processes and across model
   behaviour). Narration wording may vary; evidence may not.

## Workflow

1. **Match the question to a supported investigation type** (`src/question.py`).
   The one supported type is "why did revenue or orders change". Anything else,
   such as a customer count, a forecast, or profit, is declined with a reason
   and an example, before any tool runs or any model call is made.
2. Run the tools in the fixed rounds.
3. For each result, one plain-language evidence statement: metric, before, now,
   change. Describe what the number shows; do not editorialise about causation.
4. Apply each tool's strength rule (in the tool, not the model).
5. Narrate as visible steps for the investigation-progress screen.
6. Map supported (strong or moderate) causes to decision-option templates. Weak
   evidence gets no option; if nothing is supported the only option is to hold
   and gather more data, and it says why.
7. Conclude with the evidence and the options. No single "the answer is X".

## Tone

Plain, precise, analyst-like. Hedge only where the evidence does not support
certainty, and say why in one sentence.

## Differences from the original brief

| Original brief | As built | Why |
|---|---|---|
| Strength: High above 15% change with consistent direction, and so on | Effect size, p-value and sample-size rules per tool (`docs/EVIDENCE_STRENGTH.md`) | A size-only rule rates almost every segment High or Medium on the demo data |
| Labels High / Medium / Low | `strong` / `moderate` / `weak`, mapped in `src/strength.py` | Same three levels, named as the code produces them |
| Tools in a strict single sequence | Three rounds, tools within a round in any order, skipped tools filled in | Fewer model calls, same complete evidence |
| Tools limited to the first four | Six tools, adding the marketing test by region, the price test and the channel test | Testing causes is what separates the finding from a dashboard |
| One assumption and one risk per option | One or more of each | Cutting real assumptions to match a count would hide information |
| Question parsed into a known type | Keyword allow-list with an explicit decline | Prevents a confident revenue investigation of an unrelated question |
