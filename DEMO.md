# KyuYaar — Demo Walkthrough

This is what KyuYaar actually finds when run on its dataset. See `README.md` for the general project overview and architecture.

## The four screens

| Command center | Investigation |
|---|---|
| ![Command center](docs/screenshots/1-command-center.png) | ![Investigation progress](docs/screenshots/2-investigation.png) |

| Evidence | Decision |
|---|---|
| ![Evidence](docs/screenshots/3-evidence.png) | ![Decision options](docs/screenshots/4-decision.png) |

## Fewer orders, or smaller orders?

Revenue is orders x average order value (AOV), so a drop is fewer orders, smaller orders, or both, and the fix differs. `aov_volume_decomposition` compares the latest month with the one before it and returns two separately graded hypotheses per group.

- **The split.** Volume effect = (N1 - N0) x (A0 + A1) / 2 and AOV effect = (A1 - A0) x (N0 + N1) / 2. The two effects sum to the revenue change exactly, with no leftover interaction term, reported in points of prior-month revenue. When the factors offset each other, one effect can be larger than the total (North: orders -49.4 points, AOV +5.8, revenue -43.6%).
- **Is the move real?** Each factor's latest month-on-month change is compared with all earlier month-on-month changes (prediction-interval t statistic). A move that is ordinary for this business grades weak however large it looks; fewer than six earlier changes gives weak with a caveat.
- **Strength bars.** 15% with p < 0.01 = strong, 8% with p < 0.05 = moderate — same bars as the Layer 2 cause tests.
- **Checking causes against their pattern.** A marketing change should move order count and leave order size alone; a price change should move both, in opposite directions. `signature_check()` compares each supported cause with the decomposition and surfaces this on the evidence screen, the summary, and the decision options.

![Fewer orders, or smaller orders?](docs/screenshots/5-orders-vs-order-value.png)

## Which channel?

`marketing_effect` says a region's total spend moved with its orders. `marketing_channel_analysis` asks the same question per paid channel in each region (12 cells; Organic has no spend) and adds two checks the regional test can't make.

- **The test.** A cell is a candidate only if its channel spend moved 10+ points from the typical change for that channel. Orders under that channel are compared with the same channel in regions where spend stayed typical, within product category, same pooled log rate ratio and strength bars as Layer 2.
- **Concentrated or region-wide?** The tool also measures the region's other channels against the same comparison regions. `concentrated`: the cut channel fell significantly more than the rest. `region_wide`: the other channels fell too, so channel data can't show the cut channel drove the loss. `unclear`: too little data.
- **Less bought, or less effective?** Cost per attributed order before/after, with a Poisson noise test on order count.
- Channel evidence is its own type and never double-counted as a separate cause on top of the regional finding.

## What would each option be worth?

`run_scenario()` projects a decision option over a chosen horizon — arithmetic, not a model. Every step is a multiplication or subtraction over figures the evidence already carries, returned with its formula so it can be checked by hand ("Show the math" on the decision screen).

- Revenue recovered is the headline; gross profit follows because restoring spend costs money and rolling back a price gives up margin on orders you'd have kept anyway.
- **Break-even share**: the share of the revenue at stake that must return for gross profit to be unchanged. Above 100% means the option can't pay for itself within the horizon.
- On the default dataset: restoring North's spend is marginal on gross profit (full restoration costs ~11,000/month against ~12,700 of gross profit at stake — break-even share 130% over 3 months, 104% over 6, 95% over 12, 91% over 24). The Electronics price rollback breaks even at 55.5% of revenue at stake won back.

## Asking about the findings

The evidence screen has a question box (`answer_question()` in `src/followup.py`) — answers only from evidence already produced, no new computation.

- **Offline:** keyword retrieval, ranked by strength, sentences built from Evidence fields (correct by construction).
- **Live:** one model call, evidence supplied as JSON, must cite ids; any figure not in the evidence or any invalid cited id is discarded and replaced by the offline answer.
- Every answer lists the evidence it used; recommendations aren't made here — that's the Decision screen.

## Layer 7: is it tuned to one dataset?

Everything through Layer 6 was checked against one synthetic dataset with two injected causes. Layer 7 adds three more scenarios with known answers:

| Scenario | What is injected | Correct answer |
|---|---|---|
| `default` | Paid marketing in North cut 45%, Electronics prices +10% | North marketing and Electronics price supported, nothing else |
| `channel_loss` | Paid spend in West cut 70%, only West's Paid orders fall | West marketing supported; channel test places the loss in Paid, not the whole region |
| `demand_shock` | Orders fall 25% everywhere; spend and prices unchanged | No cause supported — decision layer ends in "insufficient data," proposes only to hold and gather more |
| `flat` | Nothing | No cause supported |

Checked over many random draws (`scripts/verify_layer7.py 20`, seeds 1000-1019), not just the shipped seed:

- `default`: North marketing and Electronics price both recovered in 20/20 draws, 0 false causes.
- `channel_loss`: West marketing recovered in 20/20 draws, 0 false causes.
- `demand_shock` and `flat`: 0 false causes in 40/40 draws — every one correctly ended in "insufficient data" with only a hold option, even though `demand_shock` showed a real 14.6%-27.6% revenue drop.

## What was verified, layer by layer

On the synthetic dataset (45% cut to paid marketing in North, 10% price rise on Electronics, both in the final month):

- **Overall:** North and Electronics come out strong; the other three regions and four categories come out weak. Across 81 hypotheses tested in earlier months with no injected cause, none was flagged.
- **Layer 3:** order count fell 22.4% (strong) vs. AOV -3.2% (weak) — revenue fell because of fewer orders, not smaller ones. North is a pure order-count loss (the pattern a marketing cut predicts); Electronics shows fewer orders + higher AOV (the pattern a price rise predicts). Over four earlier months, 3/80 hypotheses flagged, all moderate, none strong — in line with a 5% bar across 20 tests/month.
- **Layer 4:** North/Paid is the only supported channel cell (spend -44.9% vs. -1.7% typical; Paid orders -33.9%, p=0.013, n=71). The loss is region-wide, not Paid-specific (North's other channels fell 40.3%, close to Paid, difference p=0.610) — matching how the data was generated. Cost per Paid order barely moved (196.08 → 191.90, p=0.885): less was bought, it didn't perform worse. Over ten earlier months, 0/120 cells flagged.
- **Layer 5:** engine projections for North and Electronics match an independent recomputation from raw CSVs. For Electronics, projected gross profit is exactly zero at its break-even share.
- **Layer 6:** all 10 offline answers to a question battery pass the guardrail and cite only real evidence. Chart series match the findings they illustrate exactly (checked in `verify_layer6.py`).
- **Determinism:** the same question on the same data gives identical evidence, plan and decision options on every scenario, across Python hash seeds, and under different live-model wording and tool order. Only narration wording may vary (`tests/test_determinism.py`).
- **Strength scale:** every tool on every scenario emits only the three labels, no decision option is built on weak evidence, and no module keeps its own copy of the actionable threshold (`tests/test_strength.py`, rule in `docs/EVIDENCE_STRENGTH.md`).
- **Off-script questions:** a customer count, a forecast or a profit question is declined with a reason before any tool or model call (`tests/test_question.py`).

## Known limits (full detail)

- Follow-up questions have no memory of earlier ones in a session ("and South?" doesn't carry over). Offline retrieval is keyword-based. In live mode the guardrail checks figures and cited ids, not whether the model's claim about them is right.
- Follow-ups cover evidence only, not decision options or projections.
- The decision log is a local file — doesn't persist on a hosted deployment with an ephemeral disk. The investigation report is Markdown, so it carries findings but not charts.
- Layer 5 projections rest on the assumed recovery share; the data measures the drop, not how much comes back.
- Gross margin is product cost only — shipping, returns, fees, staff time are excluded, so gross profit is an upper bound on profit.
- Marketing spend in scenarios is paid from month 1, revenue arrives after the lag — penalizes short horizons; payback after the horizon isn't counted.
- The price scenario assumes a full price rollback; a partial "soften" isn't modeled.
- Both live provider paths are tested against scripted fake transports during development, not the real APIs — run `check_live.py` once with a key before relying on them.
- Evidence is association, not proof of cause; every supported finding says so.
- North x Electronics effects overlap in the data and can't be separated; impact estimates for the two aren't additive.
- One investigation type, tested on four synthetic scenarios, never on real data. Customer segment isn't exposed to the orchestrator yet.
- `aov_volume_decomposition` uses the prior month as its baseline while `baseline_trend` uses the average of all earlier months, so the two headline percentages differ (revenue -24.9% vs. prior month, -23.9% vs. baseline) — the summary labels which basis each figure uses.
- Some findings sit close to the significance bar (Electronics AOV p=0.049; South/Beauty order count p=0.061/0.053, both graded weak).
- Channel cells are small (~20-120 orders/month) — less statistical power than the regional test; a real Paid effect could be missed.
- Channel attribution (last-touch, first-touch, self-reported) is taken at face value from the input data.
- `marketing.csv` impressions aren't used yet — cost-per-thousand-impressions would separate "media got pricier" from "ads converted worse."
- AOV blends products — a fall can reflect a shift in what's bought rather than smaller baskets.
- `baseline_trend` compares calendar-month totals, so a short month (February) can read as a drop.
- The dataset is synthetic; real-data effect sizes would be far noisier.
- On a uniform demand fall, per-segment "share of revenue change" can still rank some segments as strong (product-mix noise) — a significance gate was tried and rejected because it also removed the real North/Electronics signal. The wording was changed instead: with no supported cause, the summary says segments "stand out by share" and "may not be a real concentration."
- `baseline_trend` grades by size alone: on the `flat` scenario, revenue vs. baseline ranged from -2.7% to +24.7% across draws with nothing injected — it still leads nowhere because no cause test supports anything, but the headline alone isn't evidence.
- A single-region channel loss barely moves total revenue (on `channel_loss`, median -3.7% overall while West itself fell ~half) — the argument for testing segments instead of watching the total.
- Cause tests compare a segment against the "typical" one, so they can't separate a change hitting half or more of the regions from a fully general one.
- Uploads are limited to 500,000 orders; refunds (negative revenue) are excluded rather than modeled; dates are read by pandas' inference (use YYYY-MM-DD).
