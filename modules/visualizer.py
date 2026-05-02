"""
modules/visualizer.py
All Plotly chart factories for AeroLogix AI dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Brand palette ──────────────────────────────────────────────────────────────
PRIMARY   = "#00D4FF"
SECONDARY = "#FF6B35"
SUCCESS   = "#00E676"
WARNING   = "#FFD600"
DANGER    = "#FF1744"
BG        = "#0A0E1A"
CARD_BG   = "#111827"
TEXT      = "#E2E8F0"
GRID      = "#1E293B"

AIRLINE_COLORS = px.colors.qualitative.Bold

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Space Mono, monospace"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor=GRID),
)


def _apply_layout(fig, title=""):
    fig.update_layout(**_LAYOUT, title=dict(text=title, font=dict(color=PRIMARY, size=15)))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 1. Delay by Airline — Horizontal bar
# ──────────────────────────────────────────────────────────────────────────────
def chart_delay_by_airline(df_agg: pd.DataFrame) -> go.Figure:
    if df_agg.empty:
        return go.Figure()
    fig = px.bar(
        df_agg.sort_values("Avg_Delay"),
        x="Avg_Delay", y="Airline", orientation="h",
        color="Avg_Delay",
        color_continuous_scale=["#00E676", "#FFD600", "#FF1744"],
        labels={"Avg_Delay": "Avg Delay (min)"},
        text="Avg_Delay",
    )
    fig.update_traces(texttemplate="%{text:.1f}m", textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply_layout(fig, "✈️  Average Delay by Airline")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Delay heatmap by Hour of Day
# ──────────────────────────────────────────────────────────────────────────────
def chart_delay_heatmap(df_hour: pd.DataFrame) -> go.Figure:
    if df_hour.empty:
        return go.Figure()
    fig = go.Figure(go.Bar(
        x=df_hour["Hour_of_Day"],
        y=df_hour["Avg_Delay"],
        marker=dict(
            color=df_hour["Avg_Delay"],
            colorscale=[[0, "#00D4FF"], [0.5, "#FFD600"], [1, "#FF1744"]],
            showscale=True,
            colorbar=dict(title="Avg Delay<br>(min)", tickfont=dict(color=TEXT)),
        ),
        text=df_hour["Flights"],
        texttemplate="<b>%{text} flt</b>",
        textposition="outside",
    ))
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, title="Hour of Day (UTC)")
    fig.update_yaxes(title="Avg Delay (min)")
    return _apply_layout(fig, "🕐  Flight Delay by Hour of Day")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Gate Congestion — Bubble chart
# ──────────────────────────────────────────────────────────────────────────────
def chart_gate_congestion(df_gate: pd.DataFrame) -> go.Figure:
    if df_gate.empty:
        return go.Figure()
    fig = px.scatter(
        df_gate.head(20),
        x="Flights", y="Avg_Delay",
        size="Congestion_Score", color="Congestion_Score",
        hover_name="Gate_Number",
        color_continuous_scale=["#00E676", "#FFD600", "#FF1744"],
        labels={"Flights": "Number of Flights", "Avg_Delay": "Avg Delay (min)"},
        size_max=60,
        text="Gate_Number",
    )
    fig.update_traces(textposition="top center")
    fig.update_coloraxes(colorbar=dict(title="Congestion<br>Score", tickfont=dict(color=TEXT)))
    return _apply_layout(fig, "🚦  Gate Congestion Map")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Cargo Weight by Airline — Treemap
# ──────────────────────────────────────────────────────────────────────────────
def chart_cargo_treemap(df_cargo: pd.DataFrame) -> go.Figure:
    if df_cargo.empty:
        return go.Figure()
    fig = px.treemap(
        df_cargo,
        path=["Airline"],
        values="Total_Cargo_kg",
        color="Avg_Cargo_kg",
        color_continuous_scale=["#0A0E1A", "#00D4FF", "#FF6B35"],
        hover_data={"Total_Baggage": True, "Avg_Baggage": True},
    )
    fig.update_coloraxes(colorbar=dict(title="Avg Cargo<br>(kg)", tickfont=dict(color=TEXT)))
    fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:,.0f} kg")
    return _apply_layout(fig, "📦  Cargo Weight Distribution by Airline")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Flight Status Pie
# ──────────────────────────────────────────────────────────────────────────────
def chart_status_pie(status_series: pd.Series) -> go.Figure:
    if status_series.empty:
        return go.Figure()
    colors = {
        "On Time": SUCCESS, "Delayed": DANGER,
        "Cancelled": WARNING, "Diverted": PRIMARY,
    }
    palette = [colors.get(s, SECONDARY) for s in status_series.index]
    fig = go.Figure(go.Pie(
        labels=status_series.index,
        values=status_series.values,
        hole=0.55,
        marker=dict(colors=palette, line=dict(color=BG, width=2)),
        textinfo="percent+label",
        textfont=dict(color=TEXT, size=12),
    ))
    fig.add_annotation(
        text=f"<b>{status_series.sum()}</b><br>Flights",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=TEXT),
    )
    return _apply_layout(fig, "📊  Flight Status Distribution")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Top Routes — Bar
# ──────────────────────────────────────────────────────────────────────────────
def chart_top_routes(df_routes: pd.DataFrame) -> go.Figure:
    if df_routes.empty:
        return go.Figure()
    fig = px.bar(
        df_routes.sort_values("Flights"),
        x="Flights", y="Route", orientation="h",
        color="Avg_Delay",
        color_continuous_scale=["#00E676", "#FFD600", "#FF1744"],
        text="Flights",
    )
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(colorbar=dict(title="Avg Delay<br>(min)", tickfont=dict(color=TEXT)))
    return _apply_layout(fig, "🗺️  Top Routes by Flight Volume")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Baggage vs Delay — Scatter
# ──────────────────────────────────────────────────────────────────────────────
def chart_baggage_vs_delay(df: pd.DataFrame) -> go.Figure:
    if "Baggage_Count" not in df.columns or "Delay_Minutes" not in df.columns:
        return go.Figure()
    hover_cols = {k: True for k in ["Airline", "Gate_Number", "Flight_ID"] if k in df.columns}
    try:
        fig = px.scatter(
            df,
            x="Baggage_Count", y="Delay_Minutes",
            color="Airline" if "Airline" in df.columns else None,
            color_discrete_sequence=AIRLINE_COLORS,
            opacity=0.75,
            hover_data=hover_cols,
            trendline="ols",
            labels={"Baggage_Count": "Baggage Count", "Delay_Minutes": "Delay (min)"},
        )
    except Exception:
        # statsmodels not available — render without trendline
        fig = px.scatter(
            df,
            x="Baggage_Count", y="Delay_Minutes",
            color="Airline" if "Airline" in df.columns else None,
            color_discrete_sequence=AIRLINE_COLORS,
            opacity=0.75,
            hover_data=hover_cols,
            labels={"Baggage_Count": "Baggage Count", "Delay_Minutes": "Delay (min)"},
        )
    return _apply_layout(fig, "🧳  Baggage Count vs Delay Minutes")


# ──────────────────────────────────────────────────────────────────────────────
# 8. KPI Gauge — single metric
# ──────────────────────────────────────────────────────────────────────────────
def chart_delay_gauge(delay_rate: float) -> go.Figure:
    color = SUCCESS if delay_rate < 20 else (WARNING if delay_rate < 40 else DANGER)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=delay_rate,
        delta={"reference": 20, "increasing": {"color": DANGER}, "decreasing": {"color": SUCCESS}},
        title={"text": "Delay Rate (%)", "font": {"color": TEXT, "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT},
            "bar": {"color": color},
            "bgcolor": GRID,
            "bordercolor": GRID,
            "steps": [
                {"range": [0, 20], "color": "#0D2B20"},
                {"range": [20, 40], "color": "#2B2A0D"},
                {"range": [40, 100], "color": "#2B0D0D"},
            ],
            "threshold": {"line": {"color": WARNING, "width": 3}, "value": 20},
        },
        number={"suffix": "%", "font": {"color": color, "size": 32}},
    ))
    gauge_layout = {**_LAYOUT, "height": 200, "margin": dict(l=30, r=30, t=30, b=10)}
    fig.update_layout(**gauge_layout)
    return fig