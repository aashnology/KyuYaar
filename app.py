"""
KyuYaar -- Streamlit front end.

Four screens: command center -> investigation progress -> evidence -> decision.
The app only renders. All numbers come from src/evidence.py and src/effects.py,
all wording from the orchestrator (follow-up answers from src/followup.py), all
options from src/decisions.py, all projections from src/scenario.py and all
trend series from src/trends.py.

Run with:  streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from charts import STRENGTH_COLOR, driver_figure, metric_figure  # noqa: E402
from data_loader import load_data, scenario_dir  # noqa: E402
from decision_log import DecisionLog, record_from_decision  # noqa: E402
from decisions import build_options  # noqa: E402
from decomposition import signature_check  # noqa: E402
from followup import MAX_QUESTION_CHARS, answer_question  # noqa: E402
from narration import label, metric_note  # noqa: E402
from orchestrator import investigate, make_client  # noqa: E402
from palette import DEFAULT_PALETTE, PALETTES  # noqa: E402
from question import classify_question  # noqa: E402
from recommend import recommend  # noqa: E402
from report import build_report  # noqa: E402
from strength import RANK  # noqa: E402
from synthetic import SCENARIOS  # noqa: E402
from validation import TABLES, load_upload  # noqa: E402
from scenario import (  # noqa: E402
    DEFAULT_HORIZON_MONTHS, DEFAULT_LAG_MONTHS, DEFAULT_RECOVERY_SHARE, DEFAULT_TEST_SHARE,
    Assumptions, run_scenario,
)
from toolkit import Toolkit  # noqa: E402
from trends import GROUPS, METRICS, driver_trend, metric_by_group  # noqa: E402

DEFAULT_QUESTION = "Revenue dropped last month. Why did it happen, and what should I do?"
SCREENS = [("command", "1 · Command center"), ("progress", "2 · Investigation"),
           ("evidence", "3 · Evidence"), ("decision", "4 · Decision")]
KIND_LABEL = {"act": "Act", "test": "Test first", "hold": "Hold"}

# Plain-language heading for each investigation step, for readers who don't
# read tool calls. Keyed on (tool, dimension) where a tool takes a
# `dimension` argument, on the tool name alone otherwise. The underlying
# `tool(args)` call is still shown, just de-emphasized (see _call_text).
STEP_LABELS = {
    "baseline_trend": "Checking whether revenue actually changed",
    ("aov_volume_decomposition", None): "Splitting the change into fewer orders vs. smaller orders",
    ("aov_volume_decomposition", "region"): "Checking which region's orders or order size moved",
    ("aov_volume_decomposition", "category"): "Checking which category's orders or order size moved",
    ("segment_breakdown", "region"): "Checking which region the change is concentrated in",
    ("segment_breakdown", "category"): "Checking which category the change is concentrated in",
    "marketing_effect": "Testing whether marketing spend explains it",
    "marketing_channel_analysis": "Checking which marketing channel is behind it",
    "price_effect": "Testing whether a price change explains it",
}

st.set_page_config(page_title="KyuYaar", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
      .badge {display:inline-block; padding:2px 10px; border-radius:999px;
              font-size:0.74rem; font-weight:600; color:#fff; letter-spacing:.02em;}
      .tag {display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.72rem;
            font-weight:600; border:1px solid rgba(128,128,128,.5); margin-left:6px;}
      .stage-note {opacity:.7; font-size:.85rem;}
      div[data-testid="stMetricValue"] {font-size:1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- state ---

@st.cache_resource
def get_scenario_data(name):
    return load_data(scenario_dir(name))


def active_dataset():
    """The dataset every screen works on: a demo scenario or a validated upload."""
    ds = st.session_state.get("dataset")
    if ds is None:
        orders, marketing = get_scenario_data("default")
        ds = {"kind": "scenario", "name": "default", "label": SCENARIOS["default"].title,
              "orders": orders, "marketing": marketing}
    return ds


def init_state():
    defaults = {
        "screen": "command", "pending": None, "inv": None, "decisions": None,
        "chosen": None, "note": "", "question": DEFAULT_QUESTION,
        "followups": [], "pending_followup": None, "saved_record": None, "trend_cache": {},
        "dataset": None, "upload_result": None, "data_source": "Demo scenario",
        "question_error": None, "data_chosen": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go_to(screen):
    st.session_state.screen = screen


def scroll_to_top():
    """Streamlit keeps the scroll position between screens; reset it.

    The script carries a fresh nonce each call. Without it Streamlit sees an
    identical hidden iframe and does not re-run the script on later screens.
    """
    st.session_state["_scroll_nonce"] = st.session_state.get("_scroll_nonce", 0) + 1
    components.html(
        f"<script>/* {st.session_state['_scroll_nonce']} */"
        "const m = window.parent.document.querySelector('[data-testid=stMain]');"
        "if (m) m.scrollTo({top: 0});</script>",
        height=0,
    )


def reset_investigation():
    """Forget everything derived from the previous dataset or question."""
    st.session_state.pending = None
    st.session_state.inv = None
    st.session_state.decisions = None
    st.session_state.chosen = None
    st.session_state.note = ""
    st.session_state.followups = []
    st.session_state.pending_followup = None
    st.session_state.saved_record = None
    st.session_state.trend_cache = {}


def use_scenario():
    name = st.session_state.scenario_choice
    orders, marketing = get_scenario_data(name)
    st.session_state.dataset = {"kind": "scenario", "name": name, "label": SCENARIOS[name].title,
                                "orders": orders, "marketing": marketing}
    st.session_state.upload_result = None
    st.session_state.data_chosen = True
    reset_investigation()


def apply_upload():
    sources = {t: st.session_state.get(f"upload_{t}") for t in TABLES}
    result = load_upload(sources, drop_partial_last_month=st.session_state.get("upload_trim", False))
    st.session_state.upload_result = result
    if result.ok:
        orders, marketing = result.data
        st.session_state.dataset = {"kind": "upload", "name": "upload", "label": "Your uploaded data",
                                    "orders": orders, "marketing": marketing}
        st.session_state.data_chosen = True
        reset_investigation()


def start_investigation():
    check = classify_question(st.session_state.question)
    if not check.supported:
        # Stay on the command center and say why, instead of running an
        # investigation that does not fit the question.
        st.session_state.question_error = check.reason
        return
    st.session_state.question_error = None
    st.session_state.pending = st.session_state.question
    st.session_state.inv = None
    st.session_state.decisions = None
    st.session_state.chosen = None
    st.session_state.note = ""
    st.session_state.followups = []
    st.session_state.pending_followup = None
    st.session_state.saved_record = None
    st.session_state.trend_cache = {}
    st.session_state.screen = "progress"


# --------------------------------------------------------------- helpers ---

def badge(strength):
    return f'<span class="badge" style="background:{STRENGTH_COLOR[strength]}">{strength.upper()}</span>'


def tag(text):
    return f'<span class="tag">{text}</span>'


def _fmt_args(args):
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _call_text(tool, args):
    """`tool(arg=...)` as one inline code span, with no empty backticks."""
    return f"`{tool}({_fmt_args(args)})`"


def step_label(tool, args):
    """Plain-English heading for one investigation step. Falls back to the
    tool name for anything not in STEP_LABELS."""
    if tool in ("aov_volume_decomposition", "segment_breakdown"):
        return STEP_LABELS.get((tool, args.get("dimension")), tool)
    return STEP_LABELS.get(tool, tool)


def active_palette():
    """The colour preset the person picked in the sidebar, or the default
    before that widget has run once."""
    return st.session_state.get("palette_name", DEFAULT_PALETTE)


def monthly_revenue(orders):
    series = orders.set_index("order_date")["revenue"].resample("ME").sum()
    series.index = series.index.to_period("M").astype(str)
    return series


def revenue_chart(orders, palette=None):
    pal = PALETTES.get(palette, PALETTES[DEFAULT_PALETTE])
    series = monthly_revenue(orders)
    n = len(series)
    colors = [pal["comparison"]] * (n - 1) + [pal["latest"]]
    customdata = []
    for i, val in enumerate(series.values):
        if i == 0:
            pct_text = "n/a"
        else:
            pct_text = f"{(val / series.values[i - 1] - 1) * 100:+.1f}%"
        customdata.append([pct_text, " (latest month)" if i == n - 1 else ""])
    fig = go.Figure(go.Bar(
        x=series.index, y=series.values, marker_color=colors, customdata=customdata,
        hovertemplate="%{x}%{customdata[1]}: %{y:,.0f}<br>vs. prior month: %{customdata[0]}<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        title="Monthly revenue", yaxis_title=None, xaxis_title=None,
    )
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


def causes_chart(evidence):
    rows = [e for e in evidence if e.evidence_type == "statistical" and e.value is not None]
    if not rows:
        return None
    rows.sort(key=lambda e: e.value)
    names = []
    for e in rows:
        kind = "marketing" if e.id.startswith("stat_marketing_") else "price"
        names.append(f"{e.segment.split('=', 1)[1]} · {kind}")
    fig = go.Figure(go.Bar(
        x=[e.value for e in rows], y=names, orientation="h",
        marker_color=[STRENGTH_COLOR[e.strength] for e in rows],
        error_x=dict(
            type="data", symmetric=False,
            array=[e.details["ci_high_pct"] - e.value for e in rows],
            arrayminus=[e.value - e.details["ci_low_pct"] for e in rows],
        ),
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(300, 34 * len(rows) + 90), margin=dict(l=10, r=10, t=50, b=10),
        title="Order change vs. comparable segments (%)",
        xaxis_title="Whiskers: 95% interval", yaxis_title=None,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def cached_driver_trend(ev, orders, marketing):
    """Trend series for a finding, computed once per investigation (the price
    index walks every month pair, which is slow enough to notice on each rerun)."""
    cache = st.session_state.trend_cache
    if ev.id not in cache:
        cache[ev.id] = driver_trend(ev, orders, marketing)
    return cache[ev.id]


def evidence_card(ev, pattern=None, trend=None):
    d = ev.details
    with st.container(border=True):
        which = tag(metric_note(ev)) if metric_note(ev) else ""
        st.markdown(
            f"{badge(ev.strength)} {tag(ev.evidence_type)} {tag(label(ev))}{which}",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{ev.hypothesis}**")

        bits = [f"{ev.sample_size} orders in the period"]
        if d.get("p_value_text"):
            bits.append(f"p {d['p_value_text']}")
        if d.get("ci_low_pct") is not None:
            bits.append(f"95% interval {d['ci_low_pct']:+.1f}% to {d['ci_high_pct']:+.1f}%")
        if d.get("history_std_change_pct") is not None:
            bits.append(f"normal month-to-month swing about {d['history_std_change_pct']:.1f}%")
        st.caption(" · ".join(bits))
        if pattern:
            icon = {
                "consistent": "✅", "partial": "◐", "inconsistent": "⚠️",
                "concentrated": "✅", "unclear": "◐", "region_wide": "⚠️",
            }[pattern["status"]]
            st.markdown(f"{icon} **{pattern.get('title', 'Order pattern')}.** {pattern['text']}")
        if d.get("efficiency"):
            st.markdown(f"**Spend vs. return.** {d['efficiency']['text']}")

        for c in ev.caveats:
            st.markdown(f"- {c}")

        if trend is not None:
            with st.expander("Show the trend behind this finding"):
                fig = driver_figure(trend, ev.strength, palette=active_palette())
                st.plotly_chart(fig, width="stretch", key=f"trend_{ev.id}")
                st.caption(
                    "Both panels are indexed so the average of the earlier months is 100. The dashed "
                    "line is the comparison group the finding was measured against. Drawn from the "
                    "same order and marketing data as the finding; nothing here is modelled."
                )

        with st.expander("How this was computed"):
            prov = d.get("provenance", {})
            st.markdown(
                f"`{prov.get('tool')}({_fmt_args(prov.get('args', {}))})` "
                f"in `{prov.get('source')}`"
            )
            st.json({k: v for k, v in d.items() if k != "provenance"}, expanded=False)


def render_event(ev):
    """Render one orchestrator event on the progress screen."""
    kind, data = ev.kind, ev.data
    if kind == "start":
        engine = f"{data['display']} ({data['model']})" if data["mode"] == "live" else "offline scripted plan"
        st.caption(f"Engine: {engine}")
    elif kind == "plan":
        st.markdown(f"**Plan.** {data['text']}")
    elif kind == "tool_call":
        st.markdown(f"**Step {data['step']} · {step_label(data['tool'], data['args'])}**")
        st.caption(f"({_call_text(data['tool'], data['args'])})")
    elif kind == "evidence":
        rows = [{
            "Finding": label(e) + (f" · {metric_note(e)}" if metric_note(e) else ""),
            "Strength": e.strength,
            "Key figure (%)": e.value, "Orders": e.sample_size,
        } for e in data["items"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    elif kind == "narration":
        who = "written by the model, figures checked" if data["source"] == "llm" else "generated from the evidence"
        st.markdown(f"> {data['text']}")
        st.caption(f"Readout · {who}")
    elif kind == "guardrail":
        figures = ", ".join(data["unsupported"]) or "unsupported content"
        st.warning(
            f"Guardrail: a model-written passage was discarded because it cited figures "
            f"not found in the evidence ({figures}). Evidence-derived text is shown instead."
        )
    elif kind in ("coverage", "note"):
        st.caption(data["text"])


# --------------------------------------------------------------- screens ---

def sidebar(client):
    with st.sidebar:
        st.markdown("### KyuYaar")
        st.caption("AI that investigates before it recommends.")
        st.caption(f"Data: {active_dataset()['label']}")
        st.divider()
        options = ["Offline (scripted plan)"]
        if client is not None:
            options.insert(0, f"Live LLM · {client.display} ({client.model})")
        st.radio("Investigation engine", options, key="engine")
        if client is None:
            st.caption("No `GEMINI_API_KEY`, `FEATHERLESS_API_KEY` or `ANTHROPIC_API_KEY` found, so only the offline plan is available.")
        st.selectbox(
            "Chart colours", list(PALETTES), key="palette_name",
            index=list(PALETTES).index(st.session_state.get("palette_name", DEFAULT_PALETTE)),
        )
        st.caption("Cosmetic only — strong/moderate/weak keep their own colours everywhere.")
        with st.expander("What do the strength labels mean?"):
            st.markdown(
                "**Strong / moderate / weak** rate how well the data supports a finding. "
                "They combine effect size, statistical certainty and sample size. "
                "They are not a measure of business importance, and none of them proves cause. "
                "For orders versus order value, they rate how clearly a figure moved compared with "
                "how much it normally swings from month to month."
            )
        with st.expander("Design rules"):
            st.markdown(
                "- The model plans and explains; it never does the math.\n"
                "- Every figure comes from a deterministic calculation.\n"
                "- Figures in model-written text are checked against the evidence.\n"
                "- Weak evidence is shown as weak.\n"
                "- Options are yours to choose; nothing is decided for you."
            )


def nav(suffix=""):
    inv, running = st.session_state.inv, st.session_state.pending is not None
    reachable = {
        "command": True,
        "progress": running or inv is not None,
        "evidence": inv is not None,
        "decision": inv is not None,
    }
    cols = st.columns(len(SCREENS))
    for col, (key, text) in zip(cols, SCREENS):
        col.button(
            text, key=f"nav_{key}{suffix}", on_click=go_to, args=(key,), width="stretch",
            disabled=not reachable[key],
            type="primary" if st.session_state.screen == key else "secondary",
        )
    st.divider()


def data_picker():
    ds = active_dataset()
    with st.expander(f"Data: {ds['label']}", expanded=not st.session_state.get("data_chosen", False)):
        source = st.radio("Data source", ["Demo scenario", "Upload your own CSVs"], key="data_source", horizontal=True)
        if source == "Demo scenario":
            names = list(SCENARIOS)
            current = ds["name"] if ds["kind"] == "scenario" else "default"
            st.selectbox(
                "Scenario", names, index=names.index(current), key="scenario_choice",
                format_func=lambda n: SCENARIOS[n].title, on_change=use_scenario,
            )
            sc = SCENARIOS[st.session_state.get("scenario_choice", current)]
            st.caption(sc.description)
            st.caption(
                "These are generated datasets with a known answer, so the investigation can be checked "
                "against what was injected. Scenarios with no injected cause should end in "
                "\"the data does not support a cause\"."
            )
        else:
            st.caption(
                "Upload all four tables as CSV. Required columns: **orders** order_date, customer_id, "
                "product_id, quantity, revenue, channel · **products** product_id, category, cost · "
                "**customers** customer_id, region · **marketing** date, channel, region, spend. "
                "Nothing is changed silently: unusable rows are counted and reported."
            )
            cols = st.columns(4)
            for col, table in zip(cols, TABLES):
                col.file_uploader(f"{table}.csv", type=["csv"], key=f"upload_{table}")
            st.checkbox("Leave out the latest month if it is incomplete", key="upload_trim")
            st.button("Validate and use this data", on_click=apply_upload)
            result = st.session_state.upload_result
            if result is not None:
                for msg in result.errors:
                    st.error(msg)
                for msg in result.warnings:
                    st.warning(msg)
                if result.ok:
                    sm = result.summary
                    st.success(
                        f"Using your data: {sm['orders']:,} orders over {sm['months']} months "
                        f"({sm['first_order']} to {sm['last_order']})."
                    )


def screen_command(orders):
    st.title("KyuYaar")
    st.markdown("A small-business investigator: it checks the numbers, tests the likely causes, and hands you options instead of a guess.")
    data_picker()

    series = monthly_revenue(orders)
    latest, earlier = float(series.iloc[-1]), float(series.iloc[:-1].mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders in dataset", f"{len(orders):,}")
    c2.metric("Months of history", f"{len(series)}")
    c3.metric("Latest month revenue", f"{latest:,.0f}")
    c4.metric("vs. earlier-month average", f"{(latest / earlier - 1) * 100:+.1f}%")
    st.markdown('<p class="stage-note">Quick look only. The investigation below computes its own, traceable figures.</p>',
                unsafe_allow_html=True)
    st.plotly_chart(revenue_chart(orders, palette=active_palette()), width="stretch")

    st.subheader("What do you want to know?")
    st.text_area("Business question", key="question", height=90, label_visibility="collapsed")
    st.caption("This version runs one kind of investigation: why a revenue change happened, using order, product, customer and marketing data.")
    st.button("Investigate →", type="primary", on_click=start_investigation)
    if st.session_state.get("question_error"):
        st.warning(st.session_state.question_error)


def screen_progress(toolkit, client, nav_slot):
    st.header("Investigation")
    question = st.session_state.pending
    if question is not None:
        use_live = client is not None and st.session_state.get("engine", "").startswith("Live")
        pace = float(os.environ.get("KYUYAAR_PACE", "0.35"))
        status = st.status("Investigating…", expanded=True)
        final = None
        with status:
            for ev in investigate(question, toolkit, client=client if use_live else None):
                if ev.kind == "done":
                    final = ev.data["investigation"]
                    continue
                if ev.kind == "summary":
                    continue
                render_event(ev)
                if pace and ev.kind in ("narration", "tool_call"):
                    time.sleep(pace)
            status.update(label="Investigation complete", state="complete", expanded=True)
        st.session_state.pending = None
        st.session_state.inv = final
        st.session_state.decisions = build_options(final.evidence)
        with nav_slot.container():
            nav(suffix="_done")
        st.success("The evidence chain is ready.")
        st.button("See the evidence →", type="primary", on_click=go_to, args=("evidence",))
        return

    inv = st.session_state.inv
    if inv is None:
        st.info("Start an investigation from the command center.")
        return
    st.caption(f"Question: {inv.question}")
    for step in inv.steps:
        st.markdown(f"**Step {step['step']} · {step_label(step['tool'], step['args'])}**")
        st.caption(f"({_call_text(step['tool'], step['args'])})")
        if step["narration"]:
            st.markdown(f"> {step['narration']}")
    for note in inv.notes:
        st.caption(note)
    for block in inv.guardrail_blocks:
        st.warning("A model-written passage was discarded by the numeric guardrail: " + ", ".join(block["unsupported"]))
    st.button("See the evidence →", type="primary", on_click=go_to, args=("evidence",))


def trends_section(orders):
    st.subheader("Trends by metric")
    a, b = st.columns(2)
    metric = a.radio("Metric", list(METRICS), format_func=METRICS.get, horizontal=True, key="trend_metric")
    group = b.radio(
        "Split by", ["overall", *GROUPS], horizontal=True, key="trend_group",
        format_func=lambda g: "Overall" if g == "overall" else GROUPS[g],
    )
    frame = metric_by_group(orders, metric, None if group == "overall" else group)
    title = METRICS[metric] + " by month" + ("" if group == "overall" else f", by {group}")
    fig = metric_figure(frame, title, money=metric != "orders", palette=active_palette())
    st.plotly_chart(fig, width="stretch", key="trend_metric_chart")
    st.caption("The red ring marks the latest month, the one under investigation.")


def queue_followup(question=None):
    text = question if question is not None else st.session_state.get("followup_text", "")
    if text and text.strip():
        st.session_state.pending_followup = text.strip()


def followup_section(client):
    inv = st.session_state.inv
    st.subheader("Ask about these findings")
    st.caption(
        "Answers come only from the evidence above. Nothing new is computed, and a question "
        "the investigation did not test is answered as not tested."
    )
    top = next((e for e in inv.evidence if e.evidence_type == "statistical" and e.strength != "weak"), None)
    examples = ["Are orders smaller or fewer?", "How sure are you?"]
    if top is not None:
        examples.insert(0, f"What does the evidence say about {top.segment.split('=', 1)[1]}?")
    cols = st.columns(len(examples))
    for i, (col, text) in enumerate(zip(cols, examples)):
        col.button(text, key=f"example_{i}", on_click=queue_followup, args=(text,), width="stretch")
    with st.form("followup_form", clear_on_submit=True, border=False):
        st.text_input(
            "Your question", key="followup_text", max_chars=MAX_QUESTION_CHARS,
            placeholder="For example: is the loss in one channel or across the region?",
        )
        st.form_submit_button("Ask", on_click=queue_followup)

    pending = st.session_state.pending_followup
    if pending is not None:
        st.session_state.pending_followup = None
        use_live = client is not None and st.session_state.get("engine", "").startswith("Live")
        with st.spinner("Checking the evidence…"):
            st.session_state.followups.append(
                answer_question(pending, inv, client=client if use_live else None))

    by_id = {e.id: e for e in inv.evidence}
    for qa in reversed(st.session_state.followups):
        with st.container(border=True):
            st.markdown(f"**{qa.question}**")
            for note in qa.notes:
                st.warning(note)
            st.markdown(qa.text)
            who = "written by the model, figures checked" if qa.source == "llm" else "generated from the evidence"
            st.caption(f"Answer · {who}")
            if qa.evidence_ids:
                st.caption("Evidence used: " + " · ".join(
                    f"`{i}` ({by_id[i].strength})" for i in qa.evidence_ids if i in by_id))
            elif not qa.covered:
                st.caption("Not answerable from this investigation's evidence.")


def screen_evidence(orders, marketing, client=None):
    inv = st.session_state.inv
    st.header("Evidence")
    st.info(inv.summary)
    st.caption(
        "Summary written by the model and checked against the evidence."
        if inv.summary_source == "llm"
        else "Summary generated directly from the evidence objects."
    )

    left, right = st.columns(2)
    left.plotly_chart(revenue_chart(orders, palette=active_palette()), width="stretch")
    fig = causes_chart(inv.evidence)
    if fig is not None:
        right.plotly_chart(fig, width="stretch")
    trends_section(orders)

    groups = [
        ("What changed", "observation"),
        ("Fewer orders, or smaller orders?", "decomposition"),
        ("Where the change is concentrated", "association"),
        ("Candidate causes tested", "statistical"),
        ("Marketing by channel", "channel"),
    ]
    order = RANK
    for title, etype in groups:
        items = sorted(
            [e for e in inv.evidence if e.evidence_type == etype],
            # The overall split leads its group; segments follow by strength.
            key=lambda e: (e.segment is not None and etype == "decomposition", order[e.strength], -abs(e.value or 0)),
        )
        if not items:
            continue
        st.subheader(title)
        notable = [e for e in items if e.strength != "weak"]
        weak = [e for e in items if e.strength == "weak"]
        for e in notable:
            if etype == "statistical":
                pattern = signature_check(e, inv.evidence)
            elif etype == "channel" and e.details.get("channel_pattern"):
                pattern = {**e.details["channel_pattern"], "title": "Channel pattern"}
            else:
                pattern = None
            trend = cached_driver_trend(e, orders, marketing) if etype == "statistical" else None
            evidence_card(e, pattern, trend)
        if weak:
            heading = "Weak evidence (not supported as an explanation)" if notable or etype != "observation" else "Weak evidence"
            if etype == "decomposition":
                heading = "No clear change (not distinguishable from ordinary variation)"
            with st.expander(heading):
                for e in weak:
                    evidence_card(e)

    followup_section(client)

    st.divider()
    report = build_report(inv, st.session_state.decisions, None, "", None, st.session_state.followups)
    st.download_button(
        "Download investigation report (.md)", report, file_name="kyuyaar_investigation_report.md",
        mime="text/markdown",
    )
    st.button("Continue to the decision →", type="primary", on_click=go_to, args=("decision",))


def save_decision():
    """Runs as a button callback, before the page is redrawn, so the button is
    already disabled when the person sees it and one click makes one record."""
    ss = st.session_state
    if ss.chosen is None or ss.saved_record is not None:
        return
    scenario = ss.get("_scenarios", {}).get(ss.chosen)
    record = DecisionLog().append(record_from_decision(ss.inv, ss.decisions, ss.chosen, ss.note, scenario))
    ss.saved_record = record.record_id


def choose(option_id):
    if st.session_state.chosen != option_id:
        st.session_state.saved_record = None
    st.session_state.chosen = option_id


def option_assumptions(opt):
    """The person's current assumptions for one option, read from the widgets."""
    horizon = st.session_state.get("horizon_months", DEFAULT_HORIZON_MONTHS)
    lag = min(st.session_state.get("lag_months", DEFAULT_LAG_MONTHS), horizon)
    share = st.session_state.get(f"share_{opt.id}", round(DEFAULT_RECOVERY_SHARE * 100)) / 100
    test = st.session_state.get(f"test_{opt.id}", round(DEFAULT_TEST_SHARE * 100)) / 100
    return Assumptions(recovery_share=share, lag_months=lag, horizon_months=horizon, test_share=test)


def time_frame_controls():
    with st.container(border=True):
        st.markdown("**Time frame for the projections**")
        a, b = st.columns(2)
        a.slider("Months to look ahead", 1, 12, DEFAULT_HORIZON_MONTHS, key="horizon_months")
        b.slider("Months before the effect starts", 0, 6, DEFAULT_LAG_MONTHS, key="lag_months")
        st.caption(
            "Projections are arithmetic on the evidence and on the assumptions you set here and "
            "on each option. The starting values are placeholders, not estimates: the data cannot "
            "say how much of a drop comes back."
        )


def scenario_waterfall(sc):
    """Baseline -> revenue recovered -> cost of the option -> gross profit
    change. Built only from sc.revenue_total and sc.gross_profit_total, the
    same two numbers already shown in the metrics above and the math table
    below — the middle step is just their difference, so the bars land
    exactly on the gross-profit total with no new figure invented."""
    cost = sc.gross_profit_total - sc.revenue_total
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Baseline", "Revenue recovered", "Cost of the option", "Gross profit change"],
        y=[0, sc.revenue_total, cost, 0],
        text=["0", f"{sc.revenue_total:+,.0f}", f"{cost:+,.0f}", f"{sc.gross_profit_total:+,.0f}"],
        textposition="outside",
        connector=dict(line=dict(color="#9ca3af")),
        increasing=dict(marker=dict(color="#0f766e")),
        decreasing=dict(marker=dict(color="#b45309")),
        totals=dict(marker=dict(color="#2563eb")),
        hovertemplate="%{x}: %{y:+,.0f}<extra></extra>",
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    return fig


def scenario_panel(opt, inv, orders):
    """Sliders, headline figures and the step-by-step math for one option."""
    if opt.kind in ("act", "test") and opt.id != "sequence_fixes":
        a, b = st.columns(2)
        a.slider(
            "Share of the revenue at stake this wins back", 0, 100,
            round(DEFAULT_RECOVERY_SHARE * 100), step=5, format="%d%%", key=f"share_{opt.id}",
        )
        if opt.kind == "test":
            b.slider(
                "Share of the segment treated in the test", 5, 100,
                round(DEFAULT_TEST_SHARE * 100), step=5, format="%d%%", key=f"test_{opt.id}",
            )

    sc = run_scenario(opt, inv.evidence, orders, option_assumptions(opt))
    if not sc.projectable:
        st.info(sc.summary)
        return sc

    if opt.kind != "hold":
        m1, m2, m3 = st.columns(3)
        rng = (f"Across the 95% interval on the measured drop: {sc.revenue_range[0]:,.0f} "
               f"to {sc.revenue_range[1]:,.0f}") if sc.revenue_range else None
        m1.metric(f"Revenue recovered, {sc.assumptions.horizon_months} mo", f"{sc.revenue_total:,.0f}", help=rng)
        if sc.gross_profit_total is not None:
            prng = (f"Across the 95% interval on the measured drop: {sc.gross_profit_range[0]:+,.0f} "
                    f"to {sc.gross_profit_range[1]:+,.0f}") if sc.gross_profit_range else None
            m2.metric("Gross profit change", f"{sc.gross_profit_total:+,.0f}", help=prng)
        if sc.break_even_share is not None:
            reachable = sc.break_even_share <= 1
            m3.metric(
                "Break-even share won back",
                f"{sc.break_even_share:.0%}" if reachable else "Above 100%",
                help=("Share of the revenue at stake that must come back for gross profit to be unchanged."
                      if reachable else
                      "Gross profit does not break even within this horizon even if all of the revenue "
                      "at stake comes back."),
            )
        if sc.gross_profit_total is not None:
            st.plotly_chart(scenario_waterfall(sc), width="stretch", key=f"waterfall_{opt.id}")
    st.caption(sc.summary)
    label_text = "What waiting leaves open" if opt.kind == "hold" else "Show the math"
    with st.expander(label_text):
        if sc.steps:
            st.markdown(
                "| Step | Value | How |\n|---|---:|---|\n"
                + "\n".join(f"| {x.label} | {x.shown} | {x.formula} |" for x in sc.steps)
            )
        st.markdown("**Not modelled**\n" + "\n".join(f"- {x}" for x in sc.not_modelled))
    return sc


def screen_decision(orders):
    inv, decision_set = st.session_state.inv, st.session_state.decisions
    st.header("Decision")
    st.markdown("These are options, not instructions. Each one says what it assumes and what it risks. The call is yours.")

    if decision_set.insufficient:
        st.warning("The evidence does not single out a cause, so no action is proposed beyond gathering more data.")
    else:
        time_frame_controls()

    banner_slot = st.empty()   # filled after the loop, once every scenario is known

    scenarios = {}
    for opt in decision_set.options:
        chosen = st.session_state.chosen == opt.id
        with st.container(border=True):
            conf = badge(opt.confidence) if opt.confidence in STRENGTH_COLOR else ""
            st.markdown(
                f"### {opt.title} {tag(KIND_LABEL[opt.kind])} {conf}",
                unsafe_allow_html=True,
            )
            st.write(opt.rationale)
            st.markdown(f"**Expected impact.** {opt.impact}")
            a, r = st.columns(2)
            a.markdown("**Assumes**\n" + "\n".join(f"- {x}" for x in opt.assumptions))
            r.markdown("**Risks**\n" + "\n".join(f"- {x}" for x in opt.risks))
            if not decision_set.insufficient:
                st.divider()
                st.markdown("**Projected impact**")
                scenarios[opt.id] = scenario_panel(opt, inv, orders)
            if chosen:
                st.success("You chose this option.")
            else:
                st.button("Choose this option", key=f"choose_{opt.id}",
                          on_click=choose, args=(opt.id,))

    rec = None
    if not decision_set.insufficient:
        rec = recommend(decision_set, inv.evidence, scenarios)
        with banner_slot.container():
            (st.success if rec.has_pick else st.info)(rec.sentence)
            for note in rec.notes:
                st.caption(note)
            st.caption(rec.closing)

    if decision_set.not_supported:
        with st.expander("Tested and not supported by the data"):
            for x in decision_set.not_supported:
                st.markdown(f"- {x}")
    with st.expander("Still unresolved", expanded=True):
        for x in decision_set.unresolved:
            st.markdown(f"- {x}")

    st.divider()
    st.text_area("Note for the record (optional)", key="note", height=80)
    st.session_state["_scenarios"] = scenarios      # read by save_decision()
    memo = build_report(inv, decision_set, st.session_state.chosen, st.session_state.note,
                        scenarios, st.session_state.followups, rec)
    chosen = st.session_state.chosen
    left, right = st.columns(2)
    left.download_button(
        "Download decision memo (.md)", memo, file_name="kyuyaar_decision_memo.md",
        mime="text/markdown", disabled=chosen is None, width="stretch",
    )
    log = DecisionLog()
    saved = st.session_state.saved_record
    right.button(
        "Save to decision log", disabled=chosen is None or saved is not None,
        on_click=save_decision, width="stretch",
    )
    if chosen is None:
        st.caption("Choose an option to enable the memo and the decision log.")
    if saved:
        st.success(f"Saved to the decision log as {saved}, awaiting an outcome.")
    records = log.all()
    if records:
        with st.expander(f"Decision log ({len(records)} saved)"):
            st.dataframe(pd.DataFrame([{
                "Saved": r.recorded_at, "Option": r.option_title, "Data through": r.data_period,
                "Status": r.status.replace("_", " "),
            } for r in records]), hide_index=True, width="stretch")
            st.caption(
                "Stored in a local file on this machine. Each record keeps its evidence, assumptions "
                "and projection, and an empty outcome. Outcomes are entered by a person once the real "
                "months exist; nothing here estimates them."
            )


# ------------------------------------------------------------------ main ---

def main():
    init_state()
    ds = active_dataset()
    orders, marketing = ds["orders"], ds["marketing"]
    toolkit = Toolkit(orders, marketing)
    client = make_client()
    sidebar(client)
    nav_slot = st.empty()
    with nav_slot.container():
        nav()

    screen = st.session_state.screen
    if st.session_state.get("_shown_screen") != screen:
        st.session_state["_shown_screen"] = screen
        scroll_to_top()
    if screen == "command":
        screen_command(orders)
    elif screen == "progress":
        screen_progress(toolkit, client, nav_slot)
    elif screen == "evidence" and st.session_state.inv:
        screen_evidence(orders, marketing, client)
    elif screen == "decision" and st.session_state.inv:
        screen_decision(orders)
    else:
        screen_command(orders)


main()
