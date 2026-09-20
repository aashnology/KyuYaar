"""
Scenario engine -- Layer 5.

run_scenario() projects what a decision option would be worth under
assumptions the person sets. It is arithmetic, not a model: every step is a
multiplication or subtraction over figures the evidence already carries, and
every step is returned with its formula so the projection can be checked by
hand. No LLM is involved.

What comes from the data
    revenue at stake per month (and the 95% range around it), the spend that
    would be restored, revenue in the last two months, and gross margin
    computed from product cost.

What comes from the person
    the share of the revenue at stake the option wins back, how many months
    pass before it starts to pay off, how far ahead to look, and, for a test,
    what share of the segment is treated. The data cannot say how much of a
    drop comes back, so none of these has a value the tool estimates; the
    starting values below are placeholders.

Recovery is defined in revenue terms: the share of the estimated monthly
revenue shortfall (already net of any extra revenue per order a price rise
earned) that the option wins back. Gross profit follows from that revenue and
from the margin at the price level the option leaves in place.
"""

from dataclasses import dataclass, field

from effects import _revenue_at_stake

# Placeholders shown until the person sets their own. They are not estimates.
DEFAULT_RECOVERY_SHARE = 0.5
DEFAULT_LAG_MONTHS = 1
DEFAULT_HORIZON_MONTHS = 3
DEFAULT_TEST_SHARE = 0.25

MAX_HORIZON_MONTHS = 24


@dataclass(frozen=True)
class Assumptions:
    recovery_share: float = DEFAULT_RECOVERY_SHARE
    lag_months: int = DEFAULT_LAG_MONTHS
    horizon_months: int = DEFAULT_HORIZON_MONTHS
    test_share: float = DEFAULT_TEST_SHARE

    def __post_init__(self):
        if not 0.0 <= self.recovery_share <= 1.0:
            raise ValueError("recovery_share must be between 0 and 1")
        if not 0.0 < self.test_share <= 1.0:
            raise ValueError("test_share must be above 0 and at most 1")
        if not 1 <= self.horizon_months <= MAX_HORIZON_MONTHS:
            raise ValueError(f"horizon_months must be between 1 and {MAX_HORIZON_MONTHS}")
        if not 0 <= self.lag_months <= self.horizon_months:
            raise ValueError("lag_months must be between 0 and horizon_months")

    @property
    def effective_months(self) -> int:
        return self.horizon_months - self.lag_months


@dataclass
class Step:
    label: str
    value: float
    unit: str                    # "money" | "share" | "months"
    formula: str = ""

    @property
    def shown(self) -> str:
        if self.unit == "money":
            return f"{self.value:,.0f}"
        if self.unit == "share":
            return f"{self.value:.1%}"
        return f"{self.value:g}"


@dataclass
class Scenario:
    option_id: str
    kind: str
    projectable: bool
    assumptions: Assumptions
    steps: list[Step] = field(default_factory=list)
    revenue_total: float | None = None
    revenue_range: tuple[float, float] | None = None
    gross_profit_total: float | None = None
    gross_profit_range: tuple[float, float] | None = None
    added_cost_total: float | None = None
    break_even_share: float | None = None
    summary: str = ""
    not_modelled: list[str] = field(default_factory=list)


# ------------------------------------------------------------- helpers ---

def _money(x):
    return f"{x:,.0f}"


def _months(n):
    return f"{n} month" + ("" if n == 1 else "s")


def _name(ev):
    return ev.segment.split("=", 1)[1] if ev.segment and "=" in ev.segment else ev.segment


def _stake_range(ev):
    """Monthly revenue at stake at each end of the 95% interval on the measured
    order effect, or None when the evidence carries no interval."""
    d = ev.details
    lo_effect, hi_effect = d.get("ci_low_pct"), d.get("ci_high_pct")
    if lo_effect is None or hi_effect is None or d.get("revenue_prior") is None:
        return None
    if ev.id.startswith("stat_price_"):
        gap = (d.get("driver_change_pct") or 0.0) - (d.get("driver_typical_change_pct") or 0.0)

        def net(effect):
            return ((1 + gap / 100) * (1 + effect / 100) - 1) * 100
    else:
        def net(effect):
            return effect
    prior = d["revenue_prior"]
    point = d.get("revenue_at_stake") or 0.0
    low = _revenue_at_stake(prior, net(hi_effect))
    high = _revenue_at_stake(prior, net(lo_effect))
    return min(low, point), max(high, point)


def gross_margin(orders, segment, period):
    """Gross margin of one segment's orders in one month, from product cost.
    `segment` is "column=value" as on an Evidence object. None when the
    orders carry no cost or the segment sold nothing that month."""
    if orders is None or "unit_cost" not in orders.columns or not segment or "=" not in segment:
        return None
    column, value = segment.split("=", 1)
    month = orders["order_date"].dt.to_period("M").astype(str)
    sub = orders[(orders[column] == value) & (month == period)]
    revenue = float(sub["revenue"].sum())
    if revenue <= 0:
        return None
    cost = float((sub["unit_cost"] * sub["quantity"]).sum())
    return 1 - cost / revenue


# -------------------------------------------------------------- levers ---

def _marketing_totals(stake, a, scale, ctx):
    """Revenue, added spend and gross profit over the horizon."""
    monthly = stake * a.recovery_share * scale
    revenue = monthly * a.effective_months
    cost = ctx["spend_month"] * scale * a.horizon_months
    profit = None if ctx["margin"] is None else revenue * ctx["margin"] - cost
    return revenue, cost, profit


def _price_totals(stake, a, scale, ctx):
    """Revenue and gross profit over the horizon. The price returns to its
    earlier level, so all revenue after the rollback earns the earlier margin;
    the margin given up on orders that would have been kept is what the
    extra recovered revenue has to pay for."""
    monthly = stake * a.recovery_share * scale
    revenue = monthly * a.effective_months
    if ctx["margin_prior"] is None or ctx["margin_last"] is None:
        return revenue, None, None
    after = ctx["margin_prior"] * (ctx["revenue_last"] + stake * a.recovery_share)
    now = ctx["margin_last"] * ctx["revenue_last"]
    profit = (after - now) * scale * a.effective_months
    return revenue, None, profit


def _lever_context(ev, orders):
    d = ev.details
    if ev.id.startswith("stat_marketing_"):
        return {
            "spend_month": max(d["spend_prior"] - d["spend_last"], 0.0),
            "margin": gross_margin(orders, ev.segment, d["prior_period"]),
        }
    return {
        "revenue_last": d["revenue_last"],
        "margin_prior": gross_margin(orders, ev.segment, d["prior_period"]),
        "margin_last": gross_margin(orders, ev.segment, d["period"]),
    }


def _break_even(ev, a, ctx, stake):
    """Share of the revenue at stake that has to come back for the option to
    break even on gross profit. Above 100% means it cannot."""
    if stake <= 0:
        return None
    if ev.id.startswith("stat_marketing_"):
        if ctx["margin"] in (None, 0) or a.effective_months == 0:
            return None
        return ctx["spend_month"] * a.horizon_months / (stake * ctx["margin"] * a.effective_months)
    if ctx["margin_prior"] in (None, 0) or ctx["margin_last"] is None:
        return None
    be = (ctx["margin_last"] - ctx["margin_prior"]) * ctx["revenue_last"] / (ctx["margin_prior"] * stake)
    return max(be, 0.0)


def _steps_for_lever(ev, a, scale, ctx, stake, revenue, cost, profit, be):
    seg = _name(ev)
    is_marketing = ev.id.startswith("stat_marketing_")
    steps = [Step(
        f"Monthly revenue at stake in {seg}", stake, "money",
        "from the evidence: prior-month revenue x measured loss against comparable segments",
    )]
    if scale < 1:
        steps.append(Step(f"Share of {seg} treated in the test", scale, "share", "your assumption"))
    steps.append(Step("Share of it the option wins back", a.recovery_share, "share", "your assumption"))
    monthly = stake * a.recovery_share * scale
    steps.append(Step(
        "Monthly revenue recovered", monthly, "money",
        f"{_money(stake)} x {a.recovery_share:.0%}" + (f" x {scale:.0%}" if scale < 1 else ""),
    ))
    steps.append(Step(
        "Months it pays off within the horizon", a.effective_months, "months",
        f"{a.horizon_months} - {a.lag_months} months of lag",
    ))
    steps.append(Step(
        f"Revenue recovered over {_months(a.horizon_months)}", revenue, "money",
        f"{_money(monthly)} x {a.effective_months}",
    ))
    if is_marketing:
        if ctx["margin"] is not None:
            steps.append(Step(
                f"Gross margin on {seg} orders", ctx["margin"], "share",
                f"1 - product cost / revenue on {seg} orders in {ev.details['prior_period']}",
            ))
            steps.append(Step(
                "Gross profit on the recovered revenue", revenue * ctx["margin"], "money",
                f"{_money(revenue)} x {ctx['margin']:.1%}",
            ))
        steps.append(Step(
            f"Added marketing spend over {_months(a.horizon_months)}", cost, "money",
            f"{_money(ctx['spend_month'] * scale)} a month restored x {a.horizon_months}, "
            f"paid from month 1",
        ))
        if profit is not None:
            steps.append(Step(
                "Gross profit after the added spend", profit, "money",
                f"{_money(revenue * ctx['margin'])} - {_money(cost)}",
            ))
    elif profit is not None:
        after = ctx["margin_prior"] * (ctx["revenue_last"] + stake * a.recovery_share)
        now = ctx["margin_last"] * ctx["revenue_last"]
        steps.append(Step(
            f"Gross margin on {seg} orders before / after the increase",
            ctx["margin_prior"], "share",
            f"{ctx['margin_prior']:.1%} at the earlier price, {ctx['margin_last']:.1%} at the current one",
        ))
        steps.append(Step(
            "Monthly gross profit change", (after - now) * scale, "money",
            f"{ctx['margin_prior']:.1%} x ({_money(ctx['revenue_last'])} + {_money(stake * a.recovery_share)}) "
            f"- {ctx['margin_last']:.1%} x {_money(ctx['revenue_last'])}"
            + (f", x {scale:.0%}" if scale < 1 else ""),
        ))
        steps.append(Step(
            f"Gross profit change over {_months(a.horizon_months)}", profit, "money",
            f"monthly change x {a.effective_months}",
        ))
    if be is not None:
        steps.append(Step(
            "Break-even share on gross profit", be, "share",
            "share of the revenue at stake that must come back for gross profit to be unchanged",
        ))
    return steps


def _lever_scenario(option, ev, orders, a):
    stake = ev.details.get("revenue_at_stake") or 0.0
    is_marketing = ev.id.startswith("stat_marketing_")
    scale = a.test_share if option.kind == "test" else 1.0
    ctx = _lever_context(ev, orders)
    totals = _marketing_totals if is_marketing else _price_totals

    revenue, cost, profit = totals(stake, a, scale, ctx)
    be = _break_even(ev, a, ctx, stake)

    rev_range = profit_range = None
    span = _stake_range(ev)
    if span:
        low = totals(span[0], a, scale, ctx)
        high = totals(span[1], a, scale, ctx)
        rev_range = (low[0], high[0])
        if profit is not None:
            profit_range = (min(low[2], high[2]), max(low[2], high[2]))

    seg = _name(ev)
    lead = f"If it wins back {a.recovery_share:.0%} of the revenue at stake in {seg}"
    if scale < 1:
        lead += f" across the {scale:.0%} treated in the test"
    text = f"{lead}, revenue is up about {_money(revenue)} over {_months(a.horizon_months)}"
    if rev_range:
        text += f" (range {_money(rev_range[0])} to {_money(rev_range[1])} across the 95% interval on the measured drop)"
    text += "."
    if profit is not None:
        if is_marketing:
            text += f" After {_money(cost)} of added spend, gross profit changes by {profit:+,.0f}."
        else:
            text += (f" Gross profit changes by {profit:+,.0f}, because the lower price also "
                     f"cuts margin on orders you would have kept.")
    if be is not None:
        text += (f" It breaks even on gross profit at {be:.0%} of the revenue at stake won back."
                 if be <= 1 else
                 " It does not break even on gross profit within this horizon, even if all of the "
                 "revenue at stake comes back.")
    if a.effective_months == 0:
        text += " The lag uses up the whole horizon, so nothing is recovered inside it."

    not_modelled = [
        "How much of the drop comes back is your assumption; the data cannot measure it.",
        "Gross margin covers product cost only. Shipping, returns, payment fees and staff time are left out.",
        "Payback after the horizon, and any recovery that builds gradually inside it.",
    ]
    if not is_marketing:
        not_modelled.append(
            "Revenue in the lag months, before demand returns, when the lower price already applies."
        )

    return Scenario(
        option_id=option.id, kind=option.kind, projectable=True, assumptions=a,
        steps=_steps_for_lever(ev, a, scale, ctx, stake, revenue, cost, profit, be),
        revenue_total=revenue, revenue_range=rev_range,
        gross_profit_total=profit, gross_profit_range=profit_range,
        added_cost_total=cost, break_even_share=be, summary=text, not_modelled=not_modelled,
    )


# --------------------------------------------------------- entry point ---

def run_scenario(option, evidence, orders=None, assumptions=None) -> Scenario:
    """Project one decision option under the person's assumptions.

    `option` is a DecisionOption, `evidence` the investigation's Evidence
    list and `orders` the enriched orders frame (used only for gross margin;
    without a `unit_cost` column the gross-profit lines are left out).
    """
    a = assumptions or Assumptions()
    by_id = {e.id: e for e in evidence}

    if option.kind == "hold":
        levers = [by_id[i] for i in option.addresses if i in by_id]
        steps = [
            Step(f"Monthly revenue still at stake in {_name(ev)}",
                 ev.details.get("revenue_at_stake") or 0.0, "money",
                 "from the evidence; overlaps with the other lever, so not added")
            for ev in levers
        ]
        text = ("Nothing is recovered and nothing is spent. The revenue gap stays open for as long as "
                "you wait, at the monthly figures below if the drop persists.") if levers else \
               "No supported cause, so there is nothing to size."
        return Scenario(
            option_id=option.id, kind=option.kind, projectable=True, assumptions=a,
            steps=steps, revenue_total=0.0, added_cost_total=0.0, summary=text,
            not_modelled=["Whether the drop persists, deepens or fades on its own."],
        )

    if option.id == "sequence_fixes":
        return Scenario(
            option_id=option.id, kind=option.kind, projectable=False, assumptions=a,
            summary=(
                "Not projected. The two estimates overlap where the same customers buy the same "
                "products, and the data cannot separate their combined effect from their individual "
                "ones. Adding them would double count, and choosing an overlap would be invention. "
                "Size each lever on its own option."
            ),
        )

    ev = by_id.get(option.addresses[0]) if option.addresses else None
    if ev is not None and option.id.split("_", 1)[0] in {"restore", "test", "rollback"} \
            and (ev.id.startswith("stat_marketing_") or ev.id.startswith("stat_price_")):
        return _lever_scenario(option, ev, orders, a)

    return Scenario(
        option_id=option.id, kind=option.kind, projectable=False, assumptions=a,
        summary="Not projected. This option names no specific action to size.",
    )
