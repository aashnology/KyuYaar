# Evidence strength

Every finding carries one of three labels. They are defined once, in
`src/strength.py`, and graded by the tool that produced the finding.

| Label in the code | Name in the original brief | May support a decision option? |
|---|---|---|
| `strong` | High | yes |
| `moderate` | Medium | yes |
| `weak` | Low | no |

Strength rates **how clearly the data supports a finding**. It is not a measure
of business importance, and it is never proof of cause: a supported cause is an
association measured against a comparison group, and each one says so.

## Why this is not a simple percentage threshold

The original brief graded evidence by size of change: High for more than 15%
with a consistent direction across at least two sub-segments, Medium for more
than 15% in one segment or 5-15% consistently, Low otherwise. That rule cannot
separate a cause from background movement on the demo data. In the final month
every region fell by 17.6% to 43.6% and four of five categories fell by more
than 15%, so a size-only rule rates almost everything High or Medium, including
segments where nothing was changed.

The rules below ask a different question: is this segment moving *differently
from comparable segments whose driver did not change*, by an amount that is
unlikely to be chance? On the demo data that singles out North (marketing
spend) and Electronics (price) and leaves the other regions and categories weak.

## The rules, by tool

The same two ideas recur: an effect must be large enough and statistically
distinguishable from noise, and a small sample cannot support a strong claim.

| Tool | Module | What is graded | Strong | Moderate | Otherwise |
|---|---|---|---|---|---|
| `baseline_trend` | `evidence.py` | Latest month vs. the mean of earlier months | 20% or more | 10% or more | weak |
| `segment_breakdown` | `evidence.py` | A segment's share of the change vs. its share of prior revenue (relative excess) | 50% or more | 20% or more | weak; only positive excess counts |
| `marketing_effect`, `price_effect` | `effects.py` | Order-volume change vs. comparison segments, adjusted for the other dimension, with a z-test | 15% or more with p < 0.01 | 8% or more with p < 0.05 | weak |
| `aov_volume_decomposition` | `decomposition.py` | Each factor's month-on-month change vs. its own history (prediction-interval t-test) | same bars as the cause tests | same | weak; fewer than 6 earlier changes is always weak |
| `marketing_channel_analysis` | `channels.py` | A channel's orders in one region vs. the same channel in comparison regions | same bars as the cause tests | same | weak |

Rules that apply to every tool:

- **The driver has to have moved.** A region is only a marketing candidate if its
  spend moved 10 points or more from the typical regional change, and a category
  is only a price candidate if its price moved 3 points or more. Otherwise it is
  weak whatever its orders did, and the finding says the driver did not move.
- **The direction has to fit.** A spend cut should lower orders and a price rise
  should lower orders. Movement the other way is weak.
- **Comparison groups exclude the segment under test and any segment whose
  driver also moved.** Comparing against "everyone else" lets the segment that
  really changed contaminate its own control group.
- **Small samples are downgraded.** Fewer than 30 orders lowers the label by one
  level, and says so.
- **Unstable shares are not ranked.** If the overall change is under 10%,
  `segment_breakdown` marks every segment weak, because dividing by a near-zero
  change makes the shares meaningless.
- **Channel checks describe, they do not re-grade.** Whether a channel loss is
  concentrated or region-wide, and whether cost per order moved, are reported
  alongside the label, and the decision options use them as checked assumptions.

## How the rule is checked

- `scripts/verify_layer2.py` compares the supported causes with the cause the
  generator injected.
- `tests/test_effects.py` runs the cause tests on every earlier month, where no
  cause was injected, and requires all of them to grade weak (81 hypotheses,
  none flagged).
- `tests/test_scenarios.py` runs the same engine on scenarios with different
  causes and on one with no cause at all, which must come out as "the data does
  not support a cause".
- `tests/test_strength.py` requires every tool, on every scenario, to emit only
  the three labels, and requires that no decision option is built on weak
  evidence.

## For anything built on the label

Recommendation or ranking logic must read the label through `src/strength.py`
(`is_actionable`, `weakest`, `RANK`), not compare raw strings and not keep its
own copy of the scale. `is_actionable` raises on a label it does not know, so
logic written against the brief's High/Medium/Low names fails loudly instead of
quietly treating every real label as not actionable. A test fails if any other
module defines its own actionable threshold.
