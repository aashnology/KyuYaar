"""Plotly figures for the evidence screen. They only draw series from trends.py."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from palette import get_palette

# Fixed regardless of the chosen colour preset: judges reading the evidence
# screen need strong/moderate/weak to mean the same colour everywhere.
STRENGTH_COLOR = {"strong": "#0f766e", "moderate": "#b45309", "weak": "#6b7280"}


def metric_figure(frame, title, money=False, palette=None):
    """Monthly metric, one line per column. The latest month is marked, since it
    is the month the investigation is about."""
    pal = get_palette(palette)
    series_colors, latest_color = pal["series"], pal["latest"]
    fig = go.Figure()
    for i, col in enumerate(frame.columns):
        color = series_colors[i % len(series_colors)]
        fig.add_trace(go.Scatter(
            x=list(frame.index), y=list(frame[col]), mode="lines+markers", name=str(col),
            line=dict(color=color, width=2), marker=dict(size=5),
            hovertemplate=f"{col} %{{x}}: %{{y:,.{0 if money else 1}f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[frame.index[-1]] * len(frame.columns), y=[frame.iloc[-1][c] for c in frame.columns],
        mode="markers", marker=dict(size=11, color=latest_color, symbol="circle-open", line=dict(width=2)),
        name="latest month", hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=50, b=10), title=title,
        showlegend=len(frame.columns) > 1, yaxis_title=None, xaxis_title=None,
    )
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


def driver_figure(trend, strength, palette=None):
    """Driver on top, orders below, each against the comparison segments the
    finding was measured against. 100 = the average of all earlier months."""
    grey = get_palette(palette)["comparison"]
    color = STRENGTH_COLOR[strength]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
        subplot_titles=(f"{trend.driver_label}: {trend.segment}",
                        f"Orders: {trend.segment}"),
    )
    comparison = "comparison: " + ", ".join(trend.comparison)
    rows = [
        (1, trend.driver_segment, trend.driver_comparison),
        (2, trend.orders_segment, trend.orders_comparison),
    ]
    for row, seg, ctl in rows:
        fig.add_trace(go.Scatter(
            x=trend.months, y=ctl, mode="lines", name=comparison, legendgroup="ctl",
            showlegend=row == 1, line=dict(color=grey, width=2, dash="dash"),
            hovertemplate="comparison %{x}: %{y:.1f}<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=trend.months, y=seg, mode="lines+markers", name=trend.segment, legendgroup="seg",
            showlegend=row == 1, line=dict(color=color, width=3), marker=dict(size=6),
            hovertemplate=f"{trend.segment} %{{x}}: %{{y:.1f}}<extra></extra>",
        ), row=row, col=1)
        fig.add_hline(y=100, line=dict(color=grey, width=1, dash="dot"), row=row, col=1)
    fig.update_layout(
        height=470, margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=-0.08), yaxis_title=None, yaxis2_title=None,
    )
    fig.update_xaxes(type="category", tickangle=-45)
    return fig
