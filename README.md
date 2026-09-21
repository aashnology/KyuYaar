# KyuYaar
AI that investigates before it recommends.

Most dashboards can tell a business owner what changed — revenue dropped, orders fell, a channel underperformed. Almost none can tell them why and the tools that try (a generic "AI business assistant" bolted onto a chatbot) tend to generate a confident-sounding explanation with no real evidence behind it.

KyuYaar is very humble and built for small businesses, student ventures and small organizations that have operational data but no dedicated analyst. Given a question like "revenue dropped last month — why and what should I do?", it runs a real investigation: it checks the trend, breaks it down by customer segment and product, tests it against marketing spend and builds an evidence chain — each hypothesis tagged with how strong the supporting evidence actually is. Where the data doesn't clearly support a conclusion, it says so rather than guessing.

The core design principle: the LLM reasons and orchestrates the investigation; it never does the math. Every number shown to the user comes from a deterministic Python/DuckDB calculation that can be traced back to a specific query in this repo — the model's job is to plan the investigation, interpret results, and communicate uncertainty honestly, not to produce statistics from a prompt.

KyuYaar doesn't autonomously decide anything. It surfaces evidence-backed options, each with its assumptions and risks stated plainly and leaves the actual call to the person who has to live with the outcome.

## The four screens

| Command center | Investigation |
|---|---|
| ![Command center](docs/screenshots/1-command-center.png) | ![Investigation progress](docs/screenshots/2-investigation.png) |

| Evidence | Decision |
|---|---|
| ![Evidence](docs/screenshots/3-evidence.png) | ![Decision options](docs/screenshots/4-decision.png) |

## Status

| Layer | What it adds | State |
|---|---|---|
| 1 | Synthetic dataset with a known root cause, `baseline_trend()`, `segment_breakdown()` | done |
| 2 | Cause-testing tools, LLM orchestrator, decision options, Streamlit UI | done, submittable end to end |
| 3 | `aov_volume_decomposition()`: fewer orders vs. smaller orders, and a check of each supported cause against the order pattern it predicts | done |
| 4 | `marketing_channel_analysis()`: which paid channel, whether the loss is concentrated in it or region-wide, and whether it got less effective | done |
| 5 | `run_scenario()`: what each option is worth under assumptions you set, with the arithmetic shown step by step | done |
| 6 | Follow-up questions answered from the evidence, per-metric and per-cause trend charts, a downloadable investigation report, and the decision log data structure | done; the outcome-tracking loop is documented future work |
| 7 | Three more datasets with known ground truth (a channel-only loss, a demand fall no data explains, and a flat month), checked over many random draws; upload your own four CSVs with validation | done |

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

Free tiers limit requests per minute and per day. The Gemini adapter spaces calls (`KYUYAAR_MIN_INTERVAL`, default 4 seconds), waits as long as the provider asks when it is rate limited, and treats a per-day quota as final. The model is also asked to batch independent tools into one turn, which keeps a full investigation to a handful of calls. If a live call still fails, the investigation finishes offline and says why.

```bash
python scripts/verify_layer1.py   # Layer 1 output on the dataset
python scripts/verify_layer2.py   # checks the investigation against the injected root cause
python scripts/verify_layer3.py   # orders vs. order value, ground-truth check and false-alarm count
python scripts/verify_layer4.py   # marketing by channel, ground-truth check and false-alarm count
python scripts/verify_layer5.py   # scenario arithmetic recomputed from the raw CSVs, sensitivity to the assumptions
python scripts/verify_layer6.py   # follow-up answers vs. the guardrail and ground truth, chart series vs. findings, decision log round trip
python scripts/verify_layer7.py   # every scenario over 10 random draws (pass a number for more), false-cause count, messy-upload demo
python -m pytest                  # full test suite
```

The app opens on the original dataset. **Data** on the first screen switches between the demo scenarios and an upload of your own files (see "Layer 7" below).

```bash
python scripts/generate_data.py --all                    # rewrite every scenario's CSVs (seeded, so the files do not change)
python scripts/generate_data.py --scenario channel_loss  # one scenario
```

## How an investigation works

```
question -> orchestrator -> tools -> Evidence objects -> guardrail -> UI -> decision options -> your choice
              (LLM)        (pandas)   (strength, caveats)  (number check)     (templates)
```

1. **`baseline_trend`** confirms the metric actually moved.
2. **`aov_volume_decomposition`** splits the revenue change into fewer/more orders and smaller/larger orders (details below), overall and for each region and category.
3. **`segment_breakdown`** (region, category) shows where the change is concentrated.
4. **`marketing_effect`** and **`price_effect`** test each candidate cause, region by region and category by category. A segment is only a candidate if its driver (spend, price) moved away from the typical change, and its orders are compared with segments whose driver did not move, within strata of the other dimension so a category-wide effect is not mistaken for a regional one. Effects are pooled log rate ratios with a z-test.
5. **`marketing_channel_analysis`** repeats the marketing test for each paid channel in each region and checks whether the order loss is concentrated in the channel whose spend moved (details below).
6. The **orchestrator** lets the model choose which tool to call next and write short readouts. Every figure in that prose is checked against the evidence (`src/guardrail.py`); prose with an unsupported figure is discarded and replaced with text generated from the Evidence object.
7. **`decisions.py`** maps supported (strong or moderate) causes to option templates, each with assumptions, risks and an impact estimate computed from the evidence. Nothing is ranked or chosen for you.

### Fewer orders, or smaller orders?

Revenue is orders x average order value (AOV), so a drop is fewer orders, smaller orders, or both, and the fix differs. `aov_volume_decomposition` compares the latest month with the one before it and returns two separately graded hypotheses per group, *order count changed* and *average order value changed*.

- **The split.** Volume effect = (N1 - N0) x (A0 + A1) / 2 and AOV effect = (A1 - A0) x (N0 + N1) / 2. Each factor's change is priced at the average of the two months' value of the other, so the two effects sum to the revenue change exactly, with no leftover interaction term. They are reported in points of prior-month revenue. When the factors offset each other one effect can be larger than the total (North: orders -49.4 points, AOV +5.8, revenue -43.6%).
- **Is the move real?** Order counts and AOV wander from month to month, and a raw two-month split has no control group to cancel that. Each factor's latest month-on-month change is compared with all earlier month-on-month changes (prediction-interval t statistic). A move that is ordinary for this business grades weak however large it looks, and fewer than six earlier changes gives weak with a caveat.
- **Strength.** The same size and p-value bars as the Layer 2 cause tests (15% with p < 0.01 strong, 8% with p < 0.05 moderate). It rates how clearly the figure moved, not why and not how much it matters to the total.
- **Checking causes against their pattern.** A marketing change should move order count and leave order size alone; a price change should move order count and order value in opposite directions. `signature_check()` compares each supported cause with the decomposition. The result shows on the evidence screen, in the summary, and in the decision options: the "average order value stays roughly where it is" assumption on a marketing option is now checked against the data instead of only stated.

![Fewer orders, or smaller orders?](docs/screenshots/5-orders-vs-order-value.png)

### Which channel?

`marketing_effect` says a region's total spend moved with its orders. `marketing_channel_analysis` asks the same question for each paid channel in each region (12 cells here; Organic has no spend) and adds two checks the regional test cannot make.

- **The test.** A cell is a candidate only if its channel spend moved 10 points or more from the typical change for that channel. Orders recorded under that channel are then compared with the same channel in regions where spend stayed typical in every paid channel, within product category, with the same pooled log rate ratio and the same strength bars as Layer 2.
- **Concentrated or region-wide?** A cut in Paid spend should cost Paid orders. The tool also measures the region's other channels (Organic, which has no spend, Email, Referral) against the same comparison regions and tests the difference of the two effects. `concentrated`: the cell fell significantly more than the rest. `region_wide`: the other channels fell in the same direction with a supported effect of their own, so channel data cannot show that the cut channel drove the loss. `unclear`: too little data to say.
- **Less bought, or less effective?** Cost per attributed order before and after, with a Poisson noise test on the order count. If spend and orders fall together the cost per order stays put: the channel performed as before, there was just less of it. If orders fall faster than spend, the channel also got less effective.
- **Where it shows up.** Both results appear on the evidence card, in the summary, and in the decision options: on the marketing option, "orders respond to restored spend about as they responded to the cut" is now checked against cost per order, and a region-wide result adds a checked assumption, a risk and an unresolved question instead of leaving the option to imply that restoring one channel restores the region.

Channel evidence is its own type (`channel`) and is never counted as a separate cause, so it cannot double the revenue at stake that the regional finding already carries.

### What would each option be worth?

`run_scenario()` in `src/scenario.py` projects a decision option over a horizon you choose. It is arithmetic, not a model, and no LLM is involved: every step is a multiplication or subtraction over figures the evidence already carries, and each step is returned with its formula so it can be checked by hand ("Show the math" on the decision screen, and in the memo for the option you choose).

- **What the data supplies.** Monthly revenue at stake (the same figure the option's "Expected impact" quotes) and the range it takes across the 95% interval on the measured drop, the marketing spend that would be restored, and gross margin computed from `products.csv` cost.
- **What you supply.** The share of the revenue at stake the option wins back, how many months before it starts to pay off, how many months to look ahead and, for a test, what share of the segment is treated. The data cannot say how much of a drop comes back, so the tool never estimates this. The starting values (50%, 1 month, 3 months, 25%) are placeholders and the screen says so.
- **Revenue, then gross profit.** Revenue recovered is the headline. Gross profit follows because revenue is not profit: restoring spend costs money, and rolling back a price gives up margin on orders you would have kept anyway.
  - Marketing: revenue recovered x the region's gross margin in the prior month, minus the spend restored, paid from month 1 while revenue only arrives after the lag.
  - Price: the price returns to its earlier level, so revenue after the rollback earns the earlier margin. Gross profit change = earlier margin x (current revenue + revenue recovered) - current margin x current revenue.
- **Break-even share.** The share of the revenue at stake that must come back for gross profit to be unchanged. Above 100% means the option cannot pay for itself within the horizon, whatever happens.
- **Test options** use the same math scaled by the share treated. The hold option shows the monthly revenue still at stake per lever and does not add them.
- **What it declines to size.** "Change one lever first, then the other" is not projected. Its two estimates overlap where the same customers buy the same products, the data cannot separate the combined effect from the individual ones, and any number would either double count or assume an overlap.

Every Evidence object records the tool and arguments that produced it, and the UI shows this under "How this was computed".

### Asking about the findings

The evidence screen has a question box. `answer_question()` in `src/followup.py` answers from the Evidence objects the investigation already produced. It runs no tool and computes nothing new, and a question the investigation did not test is answered as not tested, with a list of what it did cover.

- **Offline.** Keyword retrieval picks the findings a question points at (a named region, category or channel; a topic such as orders, order value, marketing, price, "where", "why", "how sure"), ranks them by strength and answers with sentences built from the Evidence fields, so every figure is correct by construction. A weak-only match says nothing here is strong enough to support as an explanation.
- **Live.** One model call with no tools. The model receives the evidence as JSON, must cite the ids behind its answer in square brackets, and its prose goes through the same numeric guardrail as the investigation readouts. An answer with a figure that is not in the evidence, or with a cited id that does not exist, is discarded and replaced by the offline answer, with a note saying why. A figure the person typed in the question may be repeated back; nothing else is exempt.
- **Every answer lists the evidence it used**, so the sentence can be checked against the finding behind it. Recommendations are not made in this box: it points to the Decision screen.

### Seeing the move

`src/trends.py` builds the series behind the charts from the same order and marketing tables the findings came from. Nothing is modelled.

- **Trends by metric.** Revenue, orders or average order value by month, overall or split by region, category or channel, with the latest month marked.
- **The trend behind a finding.** Each supported cause has a two-panel chart: the driver (marketing spend, or like-for-like price using the same product matching as `price_effect`) above, orders below, each drawn against the comparison segments the finding was measured against. Both are indexed so the average of the earlier months is 100. The chart's last-month change reproduces the finding's own `driver_change_pct`, `raw_orders_change_pct` and `control_orders_change_pct` (checked in `verify_layer6.py`).

### The report and the decision log

- **Report.** The evidence screen downloads a Markdown investigation report at any point; once an option is chosen the decision screen downloads the memo. Both include the follow-up questions asked, the answers, and the evidence each used.
- **Decision log.** "Save to decision log" stores the chosen option, the evidence behind it as it stood, the assumptions the person set and the projection, as one JSON line in `decision_log/decisions.jsonl` (git-ignored; `KYUYAAR_DECISION_LOG` moves it). Every record has an empty `outcome` and the status `awaiting_outcome`.
- **What is deliberately not built.** Closing the loop (did the projected recovery happen?) needs months of data after the decision, and the dataset ends where the decision is made. `DecisionLog.record_outcome()` exists so the structure is complete, and it stores only figures a person supplies. No screen calls it, and nothing estimates or back-fills an outcome. Comparing projected against observed is future work.

## Layer 7: is it tuned to one dataset?

Everything up to Layer 6 was checked against one synthetic dataset with two injected causes. That leaves the obvious question of whether the system only works there. The generator (`src/synthetic.py`) now builds four scenarios, each with a known answer, and `tests/test_scenarios.py` and `scripts/verify_layer7.py` check the investigation against them.

| Scenario | What is injected | Correct answer |
|---|---|---|
| `default` | Paid marketing in North cut 45%, Electronics prices +10%; both cost orders | North marketing and Electronics price supported, nothing else |
| `channel_loss` | A Paid-heavy business; Paid spend in West cut 70% and only West's Paid orders fall | West marketing supported, and the channel test places the loss in Paid, not the whole region |
| `demand_shock` | Orders fall 25% in every region and category; spend and prices unchanged | No cause supported; the decision layer ends in "insufficient data" and proposes only to hold and gather more |
| `flat` | Nothing | No cause supported |

The default scenario's four CSVs are byte-identical to before (a test regenerates and compares). The other three are committed under `data/scenarios/` and are reproducible from `scripts/generate_data.py --all`.

Two design points matter for how far this proves anything. The scenarios were checked over many random draws, not the one seed each ships with, so a lucky seed cannot carry the result. And the no-cause scenarios are the test that counts: the risk in a system like this is producing a plausible explanation when the data has none.

### Bring your own data

`src/validation.py` reads four CSVs and either accepts them or says exactly what to fix. It never changes data silently and never crashes on it.

- **Schema.** Required columns are listed in the app. Header case, spaces and hyphens are matched loosely (`Order Date` works); a missing column is named, with a rename suggestion when a close one exists. Semicolon-separated files and Latin-1 encoding are read.
- **Messy values.** Text such as `$1,234.50` is read as a number and the count reported. A few unusable rows (blank or unparseable values, non-positive quantity, negative revenue, repeated `order_id`, orders whose product or customer is unknown) are excluded and counted with the reason and an example. More than 5% of a table unusable is an error, because that usually means the wrong file. Spellings that differ only by capitalisation are merged and reported.
- **Keys.** Repeated `product_id` or `customer_id` in a reference table is an error, since it would multiply orders in the join.
- **Enough data to say anything.** Fewer than 3 months of orders is an error, fewer than 8 a warning (the orders-versus-order-value check needs about that much history). A latest month that ends before the month does is flagged, because every figure for a partial month looks like a drop; a checkbox leaves it out. Marketing that stops before the orders do, regions that never match between files, and a single region or category (nothing to compare against) are errors.
- **Last line of defence.** After cleaning, the standard investigation plan runs once. A shape the tools cannot handle comes back as a readable error instead of a traceback.

## What was verified

On the synthetic dataset (a 45% cut to paid marketing in North and a 10% price rise on Electronics, both in the final month):

- North (marketing) and Electronics (price) come out strong; the other three regions and four categories come out weak.
- Across 81 hypotheses tested in earlier months with no injected cause, none was flagged.
- Layer 1 `segment_breakdown` previously ranked segments as strong even when total revenue barely moved. It now marks shares as unstable below a 10% total change.
- The orchestrator loop, guardrail and fallbacks are covered by tests using a scripted fake client.

Layer 3 on the same dataset:

- Overall, order count fell 22.4% (strong) while AOV fell 3.2% (weak): revenue fell because of fewer orders, not smaller ones.
- North is a pure order-count loss (AOV change not distinguishable from ordinary variation), the pattern a marketing cut predicts. Electronics shows fewer orders and a higher AOV (moderate), the pattern a price rise predicts. All other regions and categories stay weak.
- Run over the four earlier months that have enough history, 3 of 80 hypotheses were flagged, all moderate and none strong. That is roughly what a 5% bar produces across 20 tests a month; unlike the Layer 2 tests it is not zero, because a raw split has no control group.

Layer 4 on the same dataset:

- North / Paid is the only supported channel cell (moderate): spend -44.9% against -1.7% typical, Paid-attributed orders -33.9% against comparison regions (p 0.013, 71 orders). The other 11 cells are weak.
- The loss is region-wide, not Paid-specific. Orders in North's other channels (Organic, Email, Referral) fell 40.3% against comparison regions, close to Paid (difference p 0.610). This matches how the data was generated, where the demand drop applies to the whole region, and it is the case a naive "Paid spend fell, Paid orders fell" test would have credited to Paid.
- Cost per Paid order went from 196.08 to 191.90 (-2.1%, p 0.885): less was bought, it did not perform worse.
- 3 of the 11 cells with unchanged spend look significant on orders alone (West Paid p < 0.001, South Paid p 0.005, North Email p 0.008). Cell-level counts are small and channel is drawn at random per order in the generator, so this is chance; the spend gate keeps all three weak.
- Run over the ten earlier months, 0 of 120 cells were flagged.
- The specificity, efficiency and gate logic is also tested on noise-free constructed worlds where the answer is known (a Paid-only loss, a region-wide loss, cost per order rising), so the tool is checked on the case this dataset does not contain.

Layer 5 on the same dataset:

- The engine's projections for North (marketing) and Electronics (price) match a recomputation from the raw CSVs that does not go through `src/scenario.py` (revenue, gross profit, break-even). For Electronics, projected gross profit is exactly zero at its break-even share (North's is above 100% here, so the same round trip is tested on a constructed marketing case).
- Restoring North's spend is marginal on gross profit. Full restoration costs about 11,000 a month against about 12,700 of gross profit at stake, so the break-even share is 130% over 3 months with a 1-month lag, 104% over 6, 95% over 12 and 91% over 24. The Electronics rollback breaks even at 55.5% of the revenue at stake won back. This is a property of the generator's spend level, not a tuned result.
- Every assumption is a slider, and the tests check that a marketing option's revenue and gross profit never fall as the share won back rises, that a test scales the act option exactly by the share treated, and that a lag as long as the horizon recovers nothing and says so.

Layer 6 on the same dataset:

- All 10 offline answers to a battery of questions pass the numeric guardrail and cite only evidence that exists. "Why did revenue drop?" surfaces both injected causes (North marketing, Electronics price); asking about each of the 7 untouched regions and categories reports weak evidence and never strong or moderate; questions about customer segments and "what should I do?" are answered as not tested and left to the Decision screen.
- The chart series match the findings they illustrate: North's spend moves -33.8% and its orders -47.6% in both the chart and the finding, and Electronics' like-for-like price +10.0%, orders -41.9%.
- The live follow-up path (guardrail, citation check, fallbacks, no-tool request for both providers) is tested against scripted fakes, not the real APIs.

Layer 7, over 20 random draws of each scenario (`python scripts/verify_layer7.py 20`, seeds 1000 to 1019):

- `default`: North marketing and Electronics price both recovered in 20 of 20 draws; 0 false causes.
- `channel_loss`: West marketing recovered in 20 of 20 draws; 0 false causes. The channel test placed the loss in West's Paid channel on the committed draw (Paid orders -70.0% against +3.9% in West's other channels).
- `demand_shock` and `flat`: 0 false causes in 40 of 40 draws, and all 40 ended in "insufficient data" with only a hold option. On `demand_shock` revenue was down 14.6% to 27.6% against baseline and the system still declined to name a cause.
- The validator is covered by 22 tests (missing files and columns, currency text, bad rows, duplicate keys, partial months, short history, marketing that ends early, non-matching regions, a tool failure), and the clean dataset passes through the upload path with the same figures as the folder loader.

## Known limits

- Follow-up questions are answered one at a time with no memory of the earlier ones, so "and South?" does not carry over. Offline retrieval is keyword-based: a differently worded question can be reported as not tested when the evidence does exist. In live mode the guardrail checks figures and cited ids, not whether the model's claim about them is right.
- Follow-ups cover the evidence only, not the decision options or projections; those stay on the Decision screen.
- The decision log is a local file: on a hosted deployment with an ephemeral disk it does not persist. The investigation report is Markdown, so it carries the findings but not the charts.

- Layer 5 projections rest on the share you assume the option wins back. The data measures the drop, not how much of it returns.
- Gross margin is product cost only. Shipping, returns, payment fees and staff time are left out, so gross profit here is an upper bound on profit.
- Marketing spend is paid from month 1 and revenue arrives after the lag, which penalises short horizons; payback after the horizon is not counted. The sensitivity is in `verify_layer5.py`.
- The price scenario assumes the price goes fully back to its earlier level. A partial "soften" is not modelled, and neither is the dip in the lag months, when the lower price applies before demand returns.

- Both live provider paths (Gemini over REST, Claude via the SDK), for investigations and for follow-up questions, are tested against scripted fake transports, not the real APIs, during development. Run `scripts/check_live.py` once with a key before relying on them.
- Evidence is association, not proof of cause. A driver and an order change in the same month cannot rule out another simultaneous change; every supported finding says so.
- The two effects overlap in North x Electronics and the data cannot separate their interaction. Options say so, and impact estimates for the two are not additive.
- One investigation type (why did revenue change), tested on four synthetic scenarios and never on real data. Customer segment is not exposed to the orchestrator: small segments need a significance test on the association itself before they can be ranked reliably. Channel is covered by a controlled test (`marketing_channel_analysis`), not by ranking.
- The orders-vs-order-value comparison uses the prior month, while `baseline_trend` uses the average of all earlier months, so the two headline percentages differ (revenue -24.9% vs. prior month, -23.9% vs. baseline). The summary labels which basis each figure uses.
- Order count and AOV are graded against only about ten earlier month-on-month changes, so the noise estimate is itself rough. Some findings sit close to a bar (Electronics AOV p = 0.049; South and Beauty order count p = 0.061 and 0.053, both graded weak). A month-length caveat appears when the two months differ in days.
- Channel cells are small (about 20 to 120 orders a month), so the channel test has less power than the regional one. North / Paid grades moderate where the regional North finding grades strong; a real Paid effect could be missed, which is why a weak cell reads "not supported", not "no effect".
- Which channel an order belongs to depends on how the channel field is attributed (last touch, first touch, self-reported). The tool takes the recorded field at face value and says so on every supported cell. In the default dataset the channel on each order is drawn independently of spend, so a Paid-specific effect cannot appear in it; the `channel_loss` scenario and constructed test worlds cover that path.
- `marketing.csv` impressions are not used yet. Cost per thousand impressions would separate "media got more expensive" from "the ads converted worse".
- AOV blends products, so a fall can mean a shift in what is bought rather than smaller baskets. The tool does not separate mix from price-per-item; that would be a further split.
- `baseline_trend` compares calendar-month totals, so a short month (February) can read as a drop. Per-day normalisation is not implemented.
- The dataset is synthetic. Effect sizes here are much cleaner than real data would be.
- Layer 7 shows the system does not invent a cause on data built to contain none. It does not show how it behaves on real data, where several things change at once, orders are attributed imperfectly and history is short. The upload path is where that would be found.
- On a uniform demand fall, the per-segment "share of the revenue change" (Layer 1) still ranks some regions and categories as strong. Revenue by segment is noisy month to month (product mix), and a significance gate on it was tried and rejected because it also removes the real North and Electronics signal in the default dataset, whose cause tests work on order counts, not revenue. What changed instead is the wording: with no supported cause the summary says those segments "stand out by share" and "may not be a real concentration" rather than "the change is concentrated in".
- `baseline_trend` grades by size alone, so an ordinary month can read as a move: on the `flat` scenario revenue against baseline ranged from -2.7% to +24.7% across draws with nothing injected. It still leads nowhere, because no cause test supports anything, but the headline figure alone is not evidence of a change.
- A single-region channel loss barely moves total revenue. On `channel_loss` the median change against baseline was -3.7% (range -13.9% to +0.4%) while West itself fell about half, and the cause was found in every draw. That is the argument for testing segments instead of watching the total.
- Cause tests compare a region or category with the typical one, so they cannot separate a change hitting half or more of the regions from a general one. The validator warns at fewer than 3 regions; it cannot know how many regions a real change touched.
- Uploads are limited to 500,000 orders, refunds (negative revenue) are excluded rather than modelled, and date strings are read by pandas' inference, so use YYYY-MM-DD.
