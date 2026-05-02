from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import re

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except Exception:
    Client = None
    TWILIO_AVAILABLE = False

load_dotenv()

st.set_page_config(
    page_title="AeroLogix AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.data_loader import load_file, load_sample_data
from modules.data_cleaner import clean_data
from modules.eda import build_full_eda
from modules.visualizer import (
    chart_delay_by_airline,
    chart_delay_heatmap,
    chart_gate_congestion,
    chart_cargo_treemap,
    chart_status_pie,
    chart_top_routes,
    chart_baggage_vs_delay,
    chart_delay_gauge,
)
from modules.recommender import generate_recommendations
from modules.report_generator import generate_report
from modules.pdf_report import generate_pdf_report
from modules.memory import (
    init_db,
    create_session,
    save_message,
    save_insight,
    get_recent_messages,
    get_recent_sessions,
)
from modules.llm_agent import AeroLogixAgent
from modules.ml_predictor import train_delay_model

try:
    from modules.agent_graph import run_aerologix_agent, get_recent_memory
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    run_aerologix_agent = None
    get_recent_memory = None

init_db()

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --c-bg: #050814;
    --c-bg2: #08111f;
    --c-card: rgba(17, 24, 39, 0.86);
    --c-card2: rgba(22, 30, 46, 0.92);
    --c-border: rgba(148, 163, 184, 0.14);
    --c-border2: rgba(0, 212, 255, 0.28);
    --c-text: #e2e8f0;
    --c-muted: #94a3b8;
    --c-primary: #00d4ff;
    --c-secondary: #ff6b35;
    --c-success: #00e676;
    --c-warning: #ffd600;
    --c-danger: #ff1744;
    --c-accent: #7c3aed;
    --font-main: 'Space Grotesk', sans-serif;
    --font-mono: 'Space Mono', monospace;
    --shadow-glow: 0 0 30px rgba(0, 212, 255, 0.12);
    --shadow-card: 0 18px 45px rgba(0, 0, 0, 0.28);
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 20% 10%, rgba(0, 212, 255, 0.12), transparent 26%),
        radial-gradient(circle at 85% 20%, rgba(124, 58, 237, 0.11), transparent 28%),
        radial-gradient(circle at 50% 90%, rgba(255, 107, 53, 0.08), transparent 26%),
        var(--c-bg) !important;
    color: var(--c-text) !important;
    font-family: var(--font-main) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.035) 1px, transparent 1px);
    background-size: 42px 42px;
    pointer-events: none;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,.85), rgba(0,0,0,.25));
}

[data-testid="stHeader"] {
    background: rgba(5, 8, 20, 0.72) !important;
    backdrop-filter: blur(14px);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8, 17, 31, 0.98), rgba(5, 8, 20, 0.98)) !important;
    border-right: 1px solid var(--c-border) !important;
}

[data-testid="stSidebar"] * {
    font-family: var(--font-main) !important;
}

.block-container {
    padding-top: 1.1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1520px !important;
}

h1, h2, h3, h4, h5, h6, p, label, span, div {
    font-family: var(--font-main);
}

#MainMenu, footer { visibility: hidden; }

/* Top navigation */
.top-nav {
    min-height: 64px;
    padding: 0 22px;
    background: linear-gradient(135deg, rgba(13, 21, 38, 0.92), rgba(15, 23, 42, 0.76));
    border: 1px solid var(--c-border);
    border-radius: 18px;
    backdrop-filter: blur(16px);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
    box-shadow: var(--shadow-glow), var(--shadow-card);
    position: relative;
    overflow: hidden;
}

.top-nav::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,.08), transparent);
    transform: translateX(-100%);
    animation: nav-shine 7s infinite;
}

@keyframes nav-shine {
    0%, 65% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.nav-logo {
    display: flex;
    align-items: center;
    gap: 11px;
    font-family: var(--font-mono);
    color: var(--c-primary);
    font-weight: 800;
    letter-spacing: 0.9px;
    position: relative;
    z-index: 1;
}

.nav-logo-icon {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(0,212,255,.22), rgba(124,58,237,.16));
    border: 1px solid rgba(0,212,255,0.38);
    display: grid;
    place-items: center;
    box-shadow: 0 0 22px rgba(0,212,255,.18);
}

.status-dot {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--c-muted);
    font-family: var(--font-mono);
    font-size: 12px;
    position: relative;
    z-index: 1;
}

.status-dot::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--c-success);
    box-shadow: 0 0 16px rgba(0,230,118,0.85);
    animation: pulse-dot 1.8s infinite;
}

@keyframes pulse-dot {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: .55; transform: scale(.72); }
}

/* Hero */
.hero {
    background:
        linear-gradient(135deg, rgba(0,212,255,0.10), rgba(124,58,237,0.08) 48%, rgba(255,107,53,0.06)),
        rgba(17,24,39,.74);
    border: 1px solid var(--c-border);
    border-radius: 20px;
    padding: 26px 30px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-card);
}

.hero::before {
    content: '';
    position: absolute;
    top: -85px;
    right: -80px;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,212,255,0.18), transparent 70%);
}

.hero::after {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--c-primary), var(--c-accent), var(--c-secondary));
}

.hero-title {
    font-size: 28px;
    font-weight: 800;
    color: var(--c-text);
    margin-bottom: 6px;
    letter-spacing: -0.03em;
}

.hero-sub {
    color: var(--c-muted);
    font-size: 14px;
}

.hero-tag {
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--c-primary);
    background: rgba(0,212,255,0.12);
    padding: 7px 13px;
    border-radius: 999px;
    border: 1px solid rgba(0,212,255,0.28);
    box-shadow: 0 0 18px rgba(0,212,255,.12);
    position: relative;
    z-index: 1;
}

/* Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}

.kpi-card, .chart-card, .rec-card, .panel-card {
    background: var(--c-card);
    border: 1px solid var(--c-border);
    border-radius: 18px;
    box-shadow: var(--shadow-card);
    backdrop-filter: blur(12px);
}

.kpi-card {
    padding: 18px 18px;
    height: 132px;
    min-height: 132px;
    position: relative;
    overflow: hidden;
    transition: all .22s ease;
}

.kpi-card:hover {
    border-color: var(--c-border2);
    transform: translateY(-4px);
    box-shadow: 0 0 34px rgba(0,212,255,0.16), var(--shadow-card);
}

.kpi-card::before {
    content: '';
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 3px;
    background: var(--kpi-color, var(--c-primary));
    opacity: .95;
}

.kpi-card::after {
    content: '';
    position: absolute;
    top: -60px;
    right: -40px;
    width: 130px;
    height: 130px;
    border-radius: 50%;
    background: rgba(0,212,255,.08);
    opacity: .7;
}

.kpi-label {
    color: var(--c-muted);
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: .7px;
    text-transform: uppercase;
    margin-bottom: 10px;
    white-space: nowrap;
}

.kpi-value {
    font-family: var(--font-mono);
    color: var(--kpi-color, var(--c-text));
    font-size: 30px;
    font-weight: 800;
    line-height: 1;
    white-space: nowrap;
}

.kpi-sub {
    color: var(--c-muted);
    font-size: 12px;
    margin-top: 9px;
    white-space: nowrap;
}

.kpi-icon {
    position: absolute;
    right: 16px;
    top: 16px;
    opacity: .55;
    font-size: 22px;
    z-index: 1;
}

.chart-card, .panel-card {
    padding: 20px 22px;
    margin-bottom: 18px;
}

.card-title {
    color: var(--c-muted);
    font-family: var(--font-mono);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .7px;
    margin-bottom: 15px;
}

.section-title {
    font-weight: 800;
    color: var(--c-text);
    display: flex;
    gap: 9px;
    align-items: center;
    margin-bottom: 14px;
    font-size: 18px;
}

.section-title::before {
    content: '';
    width: 4px;
    height: 18px;
    background: linear-gradient(180deg, var(--c-primary), var(--c-accent));
    border-radius: 999px;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.16), rgba(0,136,170,0.18)) !important;
    border: 1px solid rgba(0,212,255,0.38) !important;
    color: var(--c-primary) !important;
    border-radius: 12px !important;
    font-family: var(--font-main) !important;
    font-weight: 700 !important;
    transition: all .22s ease !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, var(--c-primary), #0088aa) !important;
    color: #00111a !important;
    box-shadow: 0 0 26px rgba(0,212,255,0.28) !important;
    transform: translateY(-2px);
}

/* Inputs and tables */
[data-testid="stFileUploader"] {
    background: rgba(0,212,255,0.035) !important;
    border: 2px dashed rgba(0,212,255,0.25) !important;
    border-radius: 18px !important;
    padding: 14px !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--c-border) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

/* Radio nav */
[data-testid="stRadio"] > div {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

[data-testid="stRadio"] label {
    background: var(--c-card);
    border: 1px solid var(--c-border);
    border-radius: 12px;
    padding: 7px 11px;
    transition: all .22s ease;
}

[data-testid="stRadio"] label:hover {
    border-color: rgba(0,212,255,0.35);
    background: var(--c-card2);
    transform: translateY(-1px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    padding-bottom: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: var(--c-card) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 14px !important;
    padding: 10px 15px !important;
    color: var(--c-muted) !important;
    transition: all .22s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--c-primary) !important;
    border-color: rgba(0,212,255,.35) !important;
    transform: translateY(-2px);
    box-shadow: 0 0 18px rgba(0,212,255,.10);
}

.stTabs [aria-selected="true"] {
    color: #00111a !important;
    background: linear-gradient(135deg, var(--c-primary), #0088aa) !important;
    font-weight: 800 !important;
    box-shadow: 0 0 22px rgba(0,212,255,.20);
}

.stTabs [aria-selected="true"] p {
    color: #00111a !important;
    font-weight: 800 !important;
}

/* Recommendation cards */
.rec-card {
    padding: 16px 17px;
    border-left: 4px solid var(--rec-color, var(--c-primary));
    margin-bottom: 13px;
    transition: all .22s ease;
}

.rec-card:hover {
    transform: translateY(-2px);
    border-color: var(--c-border2);
}

.rec-title {
    font-weight: 800;
    margin-bottom: 6px;
    color: var(--c-text);
}

.rec-detail {
    color: #a8b5c8;
    font-size: 13px;
    line-height: 1.55;
}

.rec-impact {
    color: var(--c-success);
    font-size: 12px;
    margin-top: 8px;
}

/* Chat */
.chat-ai, .chat-user {
    border-radius: 16px;
    padding: 14px 16px;
    margin: 10px 0;
    border: 1px solid var(--c-border);
    box-shadow: var(--shadow-card);
}

.chat-ai {
    background: rgba(255,255,255,0.045);
    border-left: 4px solid var(--c-secondary);
}

.chat-user {
    background: rgba(0,212,255,0.09);
    border-left: 4px solid var(--c-primary);
}

.small-muted {
    color: var(--c-muted);
    font-size: 12px;
}

/* Prevent broken icon-name leakage */
.kpi-icon, .nav-logo-icon {
    font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif !important;
    max-width: 38px;
    max-height: 38px;
    overflow: hidden;
    white-space: nowrap;
}

/* Scrollbar */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,.04); }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,.30); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,.55); }

@media (max-width: 950px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .hero { flex-direction: column; align-items: flex-start; gap: 12px; }
    .hero-title { font-size: 22px; }
}
@media (max-width: 600px) {
    .kpi-grid { grid-template-columns: 1fr; }
}

/* Sidebar equal buttons + active navigation polish */
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 48px !important;
    border-radius: 14px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    margin: 6px 0 8px 0 !important;
    background: rgba(17,24,39,0.75) !important;
    border: 1px solid rgba(148,163,184,0.16) !important;
    color: #cbd5e1 !important;
    transition: all .22s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    color: #00d4ff !important;
    border-color: rgba(0,212,255,.40) !important;
    background: rgba(22,30,46,.95) !important;
    transform: translateX(4px);
    box-shadow: 0 0 24px rgba(0,212,255,.16) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4ff, #0088aa) !important;
    color: #00111a !important;
    border-color: rgba(0,212,255,.70) !important;
}
.sidebar-mini-footer {
    text-align: center;
    color: rgba(226,232,240,.45);
    font-size: 11px;
    margin-top: 12px;
    padding: 12px 6px;
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 12px;
    background: rgba(17,24,39,.45);
}


/* Login and Passenger Alert UI */
.login-card { max-width: 480px; margin: 6vh auto 0 auto; padding: 28px; background: rgba(17,24,39,.88); border: 1px solid rgba(0,212,255,.24); border-radius: 22px; box-shadow: 0 0 40px rgba(0,212,255,.14), 0 18px 45px rgba(0,0,0,.35); }
.login-title { font-size: 28px; font-weight: 800; color: #00d4ff; margin-bottom: 6px; }
.login-sub { color: #94a3b8; font-size: 14px; margin-bottom: 18px; }
.flight-card { background: rgba(17,24,39,.82); border: 1px solid rgba(148,163,184,.16); border-left: 4px solid #00d4ff; border-radius: 18px; padding: 18px; margin: 14px 0; }
.flight-card h3 { margin: 0 0 12px 0; color: #e2e8f0; }
.flight-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.flight-field { background: rgba(255,255,255,.045); border: 1px solid rgba(148,163,184,.12); border-radius: 12px; padding: 10px; }
.flight-label { color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
.flight-value { color: #e2e8f0; font-weight: 700; margin-top: 3px; }
@media (max-width: 700px) { .flight-grid { grid-template-columns: 1fr; } }

</style>

""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "raw_df": None,
        "df": None,
        "cleaning_report": "",
        "eda": None,
        "recommendations": [],
        "ai_summary": "",
        "report_md": "",
        "session_id": None,
        "lg_session_id": str(uuid.uuid4()),
        "chat_history": [],
        "filename": "",
        "analysis_done": False,
        "ml_model": None,
        "ml_accuracy": None,
        "ml_importance": None,
        "ml_report": None,
        "page": "Dashboard",
        "logged_in": False,
        "user_role": None,
        "username": "",
        "passenger_selected_flight": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


@st.cache_resource(show_spinner=False)
def build_agent():
    try:
        agent = AeroLogixAgent(provider=os.getenv("LLM_PROVIDER", "auto"))
        return agent, agent.llm, agent.model_name, None
    except Exception as exc:
        return None, None, None, str(exc)


agent_obj, raw_llm, llm_name, llm_error = build_agent()


def fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def kpi_card(label: str, value: str, sub: str, icon: str, color: str) -> None:
    icon_map = {
        "flight": "✈️",
        "schedule": "⏱️",
        "timer": "⌛",
        "inventory": "📦",
        "luggage": "🧳",
    }
    icon_str = str(icon)
    icon_display = icon_map.get(icon_str, icon_str if len(icon_str) <= 3 else "✦")

    st.markdown(
        f"""
<div class="kpi-card" style="--kpi-color:{color}">
  <div class="kpi-icon">{icon_display}</div>
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, tag: str = "AI-POWERED") -> None:
    st.markdown(
        f"""
<div class="hero">
  <div>
    <div class="hero-title">{title}</div>
    <div class="hero-sub">{subtitle}</div>
  </div>
  <div class="hero-tag">{tag}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def top_nav() -> None:
    status = "LIVE DATA" if st.session_state.analysis_done else "READY"
    st.markdown(
        f"""
<div class="top-nav">
  <div class="nav-logo"><div class="nav-logo-icon">✈</div> AEROLOGIX AI</div>
  <div class="status-dot">{status}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def nav_button(label: str, page_name: str, icon: str) -> None:
    """Premium sidebar navigation button."""
    active = st.session_state.page == page_name
    prefix = "●" if active else "○"
    if st.button(f"{prefix} {icon} {label}", key=f"nav_{page_name}", use_container_width=True):
        st.session_state.page = page_name
        st.rerun()



def prepare_agent_summary(eda: dict, df: pd.DataFrame) -> dict:
    priority_delayed = pd.DataFrame()
    if "Delay_Minutes" in df.columns:
        cols = [
            c
            for c in [
                "Flight_ID",
                "Airline",
                "Origin",
                "Destination",
                "Departure_Time",
                "Arrival_Time",
                "Delay_Minutes",
                "Gate_Number",
                "Flight_Status",
            ]
            if c in df.columns
        ]
        priority_delayed = df[df["Delay_Minutes"] > 0][cols].copy()
        if not priority_delayed.empty:
            priority_delayed = priority_delayed.sort_values("Delay_Minutes", ascending=False).head(10)
    return {
        "delay_by_airline": eda.get("delay_by_airline", pd.DataFrame()),
        "delay_by_hour": eda.get("delay_by_hour", pd.DataFrame()),
        "gate_congestion": eda.get("gate_congestion", pd.DataFrame()),
        "top_routes": eda.get("top_routes", pd.DataFrame()),
        "cargo": eda.get("cargo", pd.DataFrame()),
        "priority_delayed_flights": priority_delayed,
        "summary": eda.get("summary", {}),
    }


def load_and_analyse(df_raw: pd.DataFrame, filename: str) -> None:
    df_clean, cleaning_report = clean_data(df_raw)
    eda = build_full_eda(df_clean)
    recs = generate_recommendations(eda)

    session_id = create_session(filename, len(df_clean), eda["summary"])

    ai_summary = ""
    if agent_obj:
        agent_obj.set_data_context(df_clean, eda, cleaning_report.to_markdown())
        ai_summary = agent_obj.explain_patterns(eda)
        save_insight(session_id, "executive_summary", ai_summary)

    report_md = generate_report(
        df_clean,
        eda,
        cleaning_report.to_markdown(),
        recs,
        ai_summary or "AI summary unavailable. Add an API key to generate LLM insights.",
        filename=filename,
    )

    st.session_state.raw_df = df_raw
    st.session_state.df = df_clean
    st.session_state.cleaning_report = cleaning_report.to_markdown()
    st.session_state.eda = eda
    st.session_state.recommendations = recs
    st.session_state.ai_summary = ai_summary
    st.session_state.report_md = report_md
    st.session_state.session_id = session_id
    st.session_state.filename = filename
    st.session_state.analysis_done = True
    st.session_state.chat_history = []
    st.session_state.ml_model = None
    st.session_state.ml_accuracy = None
    st.session_state.ml_importance = None
    st.session_state.ml_report = None


def chart_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    if importance_df is None or importance_df.empty:
        return go.Figure()
    plot_df = importance_df.sort_values("Importance", ascending=True).tail(10)
    fig = go.Figure(
        go.Bar(
            x=plot_df["Importance"],
            y=plot_df["Feature"],
            orientation="h",
            marker=dict(color="#00d4ff"),
            text=plot_df["Importance"].round(3),
            textposition="outside",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Space Mono, monospace"),
        margin=dict(l=30, r=30, t=20, b=30),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
        height=360,
    )
    return fig



# -----------------------------------------------------------------------------
# Login, passenger lookup, and WhatsApp helpers
# -----------------------------------------------------------------------------
def login_screen() -> None:
    """
    Demo login for assignment/portfolio:
    - Any non-empty username and password can log in.
    - User chooses Admin or Passenger role.
    - Admin sees full dashboard.
    - Passenger sees only flight-status tools.
    """
    st.markdown(
        """
<div class="login-card">
  <div class="login-title">✈ AeroLogix AI Login</div>
  <div class="login-sub">Select your role and continue with any username and password.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1.25, 1])
    with c2:
        role = st.radio(
            "Login as",
            ["Admin", "Passenger"],
            horizontal=True,
            key="login_role_choice",
        )

        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("Login", type="primary", use_container_width=True):
            username_clean = username.strip()
            password_clean = password.strip()

            if not username_clean:
                st.error("Please enter a username.")
            elif not password_clean:
                st.error("Please enter a password.")
            else:
                st.session_state.logged_in = True
                st.session_state.username = username_clean
                st.session_state.user_role = role.lower()
                st.success(f"Login successful as {role}")
                st.rerun()


def logout() -> None:
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.username = ""
    st.session_state.chat_history = []
    st.rerun()


def normalize_flight_id(value: str) -> str:
    return str(value).strip().upper()


def find_flight(df: pd.DataFrame, flight_id: str):
    if df is None or df.empty or "Flight_ID" not in df.columns:
        return None
    fid = normalize_flight_id(flight_id)
    matches = df[df["Flight_ID"].astype(str).str.upper().str.strip() == fid]
    if matches.empty:
        return None
    return matches.iloc[0]


def passenger_advice(row) -> str:
    delay = float(row.get("Delay_Minutes", 0) or 0)
    gate = row.get("Gate_Number", "N/A")
    if delay >= 60:
        return f"Your flight has a major delay. Stay near Gate {gate}, monitor announcements, and contact airport support if you need assistance."
    if delay > 0:
        return f"Your flight is delayed. Please stay near Gate {gate} and check updates regularly."
    return f"Your flight is currently on time. Please arrive at Gate {gate} as scheduled."


def build_flight_message(row) -> str:
    route = f"{row.get('Origin', 'N/A')} -> {row.get('Destination', 'N/A')}"
    delay = float(row.get("Delay_Minutes", 0) or 0)
    delay_text = f"{int(delay)} minutes" if delay > 0 else "No delay"
    status = row.get("Flight_Status", "On Time" if delay <= 0 else "Delayed")
    return (
        "AeroLogix Flight Update\n\n"
        f"Flight ID: {row.get('Flight_ID', 'N/A')}\n"
        f"Airline: {row.get('Airline', 'N/A')}\n"
        f"Route: {route}\n"
        f"Departure: {row.get('Departure_Time', 'N/A')}\n"
        f"Arrival: {row.get('Arrival_Time', 'N/A')}\n"
        f"Gate: {row.get('Gate_Number', 'N/A')}\n"
        f"Delay: {delay_text}\n"
        f"Status: {status}\n\n"
        f"Passenger Advice:\n{passenger_advice(row)}"
    )


def normalize_whatsapp_number(number: str) -> str:
    """
    Accept both +947XXXXXXXX and whatsapp:+947XXXXXXXX.
    Twilio requires whatsapp:+countrycode format.
    """
    number = str(number or "").strip().replace(" ", "")
    if not number:
        return ""
    if number.startswith("whatsapp:+"):
        return number
    if number.startswith("+"):
        return f"whatsapp:{number}"
    if number.startswith("94"):
        return f"whatsapp:+{number}"
    if number.startswith("0") and len(number) >= 10:
        # Sri Lanka local mobile format, e.g. 0703394005 -> +94703394005
        return f"whatsapp:+94{number[1:]}"
    return number


def is_valid_whatsapp_number(number: str) -> bool:
    number = normalize_whatsapp_number(number)
    return bool(re.fullmatch(r"whatsapp:\+\d{10,15}", number))


def clean_twilio_error(exc: Exception) -> str:
    """Remove terminal color codes/noisy output from Twilio exceptions."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
    text = text.replace("[31m", "").replace("[0m", "").strip()

    if "could not find a Channel with the specified From address" in text:
        return (
            "Twilio could not find the WhatsApp sender. "
            "Set TWILIO_WHATSAPP_FROM=whatsapp:+14155238886 in .env, "
            "then restart Streamlit."
        )

    if "63007" in text:
        return (
            "Twilio error 63007: invalid WhatsApp sender. "
            "Use the Sandbox sender: whatsapp:+14155238886"
        )

    if "Authenticate" in text or "Authentication" in text or "20003" in text:
        return "Twilio authentication failed. Check TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."

    return text


def send_whatsapp_alert(to_number: str, message: str) -> tuple[bool, str]:
    """
    Send a WhatsApp alert using Twilio Sandbox.

    .env must use:
    TWILIO_WHATSAPP_FROM=whatsapp:+14155238886

    The passenger number can be entered as:
    +94703394005
    or
    whatsapp:+94703394005
    """
    if not TWILIO_AVAILABLE:
        return False, "Twilio package is not installed. Run: pip install twilio"

    sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()

    # IMPORTANT: For Twilio Sandbox, FROM must be the official sandbox sender.
    # Do not put your personal phone number here.
    from_number = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886").strip()
    if from_number in {"", "whatsapp:+13203993544", "+13203993544"}:
        from_number = "whatsapp:+14155238886"
    if from_number.startswith("+"):
        from_number = "whatsapp:" + from_number

    to_number = normalize_whatsapp_number(to_number)

    if not sid or not token:
        return False, "Twilio credentials are missing in .env"

    if from_number != "whatsapp:+14155238886":
        return (
            False,
            "Wrong TWILIO_WHATSAPP_FROM value. For Sandbox use: whatsapp:+14155238886"
        )

    if not is_valid_whatsapp_number(to_number):
        return False, "Use WhatsApp format like whatsapp:+947XXXXXXXX or +947XXXXXXXX"

    try:
        client = Client(sid, token)
        sent = client.messages.create(
            body=message,
            from_=from_number,
            to=to_number,
        )
        return True, f"WhatsApp alert sent successfully. SID: {sent.sid}"
    except Exception as exc:
        return False, f"WhatsApp send failed: {clean_twilio_error(exc)}"



def render_twilio_config_hint() -> None:
    from_number = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886").strip()
    if from_number != "whatsapp:+14155238886":
        st.warning(
            "Twilio Sandbox sender should be TWILIO_WHATSAPP_FROM=whatsapp:+14155238886. "
            "Update .env and restart Streamlit."
        )

def render_flight_card(row) -> None:
    delay = float(row.get("Delay_Minutes", 0) or 0)
    status_icon = "Delayed" if delay > 0 else "On Time"
    fields = [
        ("Flight ID", row.get("Flight_ID", "N/A")),
        ("Airline", row.get("Airline", "N/A")),
        ("Route", f"{row.get('Origin', 'N/A')} -> {row.get('Destination', 'N/A')}"),
        ("Departure", row.get("Departure_Time", "N/A")),
        ("Arrival", row.get("Arrival_Time", "N/A")),
        ("Gate", row.get("Gate_Number", "N/A")),
        ("Delay", f"{int(delay)} min"),
        ("Status", status_icon),
        ("Baggage", row.get("Baggage_Count", "N/A")),
    ]
    html = '<div class="flight-card"><h3>Passenger Flight Details</h3><div class="flight-grid">'
    for label, value in fields:
        html += f'<div class="flight-field"><div class="flight-label">{label}</div><div class="flight-value">{value}</div></div>'
    html += f'</div><br><b>Advice:</b> {passenger_advice(row)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def passenger_page() -> None:
    top_nav()
    hero(
        "Passenger Flight Status",
        "Choose your Flight ID and receive passenger-friendly delay updates.",
        "PASSENGER",
    )

    if not st.session_state.analysis_done:
        with st.spinner("Loading sample airport data for passenger lookup..."):
            raw = load_sample_data()
            load_and_analyse(raw, "sample_airport_data.csv")

    df_local = st.session_state.df

    if df_local is None or df_local.empty or "Flight_ID" not in df_local.columns:
        st.error("No flight data available.")
        if st.button("Logout", use_container_width=True):
            logout()
        return

    flight_options = sorted(df_local["Flight_ID"].dropna().astype(str).unique())

    st.markdown('<div class="panel-card"><div class="section-title">Find Your Flight</div>', unsafe_allow_html=True)

    flight_id = st.selectbox(
        "Choose Flight ID",
        flight_options,
        index=0,
        key="passenger_flight_select",
    )

    if st.button("View Flight Status", type="primary", use_container_width=True):
        st.session_state.passenger_selected_flight = flight_id

    if not st.session_state.get("passenger_selected_flight") and flight_options:
        st.session_state.passenger_selected_flight = flight_options[0]

    selected_id = st.session_state.get("passenger_selected_flight", "")
    if selected_id:
        row = find_flight(df_local, selected_id)
        if row is None:
            st.error("Flight ID not found. Please choose another Flight ID.")
        else:
            render_flight_card(row)
            msg = build_flight_message(row)
            st.text_area("Message Preview", msg, height=230)
            render_twilio_config_hint()

            number = st.text_input(
                "Passenger WhatsApp Number",
                placeholder="+947XXXXXXXX or whatsapp:+947XXXXXXXX",
            )
            number = normalize_whatsapp_number(number)

            if st.button("Send WhatsApp Update", use_container_width=True):
                ok, info = send_whatsapp_alert(number, msg)
                if ok:
                    st.success(info)
                else:
                    st.error(info)

    st.markdown("</div>", unsafe_allow_html=True)

    st.info("Passenger access is limited to flight status only.")

    if st.button("Logout", use_container_width=True):
        logout()


def passenger_alerts_tab(df_local: pd.DataFrame) -> None:
    st.markdown('<div class="panel-card"><div class="section-title">Passenger WhatsApp Alerts</div>', unsafe_allow_html=True)
    flight_id = st.text_input("Flight ID for passenger alert", placeholder="Example: FL009")
    if flight_id:
        row = find_flight(df_local, flight_id)
        if row is None:
            st.warning("No matching flight found.")
        else:
            render_flight_card(row)
            msg = build_flight_message(row)
            st.text_area("WhatsApp Message Preview", msg, height=230)
            render_twilio_config_hint()
            number = st.text_input("Send to WhatsApp number", placeholder="+947XXXXXXXX or whatsapp:+947XXXXXXXX")
            number = normalize_whatsapp_number(number)
            if st.button("Send Passenger WhatsApp Alert", type="primary", use_container_width=True):
                ok, info = send_whatsapp_alert(number, msg)
                st.success(info) if ok else st.error(info)
    st.markdown('</div>', unsafe_allow_html=True)


if not st.session_state.logged_in:
    login_screen()
    st.stop()

if st.session_state.user_role == "passenger":
    passenger_page()
    st.stop()


with st.sidebar:
    st.markdown("### ✈ AeroLogix AI")
    st.caption("Airport Operations Intelligence")
    st.caption(f"Logged in as: {st.session_state.username} ({st.session_state.user_role})")
    if st.button("Logout", use_container_width=True):
        logout()
    st.divider()

    st.markdown(
        "<div class='sidebar-mini-footer'>Premium airport intelligence dashboard</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Data Source")
    use_sample = st.toggle("Use sample dataset", value=True)
    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader("Upload CSV / XLSX / XLS", type=["csv", "xlsx", "xls"])

    if st.button("Load & Analyse", type="primary", use_container_width=True):
        with st.spinner("Loading and analysing airport data..."):
            try:
                if use_sample:
                    raw = load_sample_data()
                    fname = "sample_airport_data.csv"
                    err = ""
                elif uploaded_file:
                    raw, err = load_file(uploaded_file)
                    fname = uploaded_file.name
                else:
                    raw, err, fname = None, "Upload a file or enable sample dataset.", ""

                if err:
                    st.error(err)
                elif raw is not None:
                    load_and_analyse(raw, fname)
                    st.success(f"Loaded {len(st.session_state.df):,} flights")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    st.divider()
    st.markdown("### Navigation")
    nav_button("Dashboard", "Dashboard", "📊")
    nav_button("Analytics", "Analytics", "📈")
    nav_button("ML Predictor", "ML Predictor", "🤖")
    nav_button("Reports", "Reports", "📄")
    nav_button("AI Assistant", "AI Assistant", "💬")
    nav_button("Passenger Alerts", "Passenger Alerts", "📲")

    if st.session_state.analysis_done:
        s = st.session_state.eda["summary"]
        st.divider()
        st.markdown("### Quick Stats")
        st.metric("Flights", fmt_int(s.get("total_flights", 0)))
        st.metric("Delay Rate", f"{s.get('delay_rate_pct', 0)}%")
        st.metric("Avg Delay", f"{s.get('avg_delay_minutes', 0)} min")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


top_nav()

if not st.session_state.analysis_done:
    hero(
        "Airport Operations Intelligence",
        "Upload airport logistics data to analyse delays, cargo, gates, routes, ML risk and AI recommendations.",
        "READY",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="panel-card"><div class="section-title">Auto Analysis</div><p class="small-muted">Clean datasets, compute KPIs, visualise delays and find bottlenecks.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel-card"><div class="section-title">AI Q&A</div><p class="small-muted">Ask questions in plain English using LangChain or LangGraph workflow.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="panel-card"><div class="section-title">Reports</div><p class="small-muted">Generate professional PDF reports for your assignment and portfolio.</p></div>', unsafe_allow_html=True)
    st.info("Use the sidebar to load the sample dataset or upload your own CSV/XLSX file.")
    st.stop()

# Aliases
df = st.session_state.df
eda = st.session_state.eda
summary = eda["summary"]
recs = st.session_state.recommendations
page = st.session_state.page

if page == "Dashboard":
    hero(
        "Airport Operations Intelligence",
        f"Dataset: {st.session_state.filename} | Analyze delays, cargo, gates and airline performance in real time",
        "AI-POWERED",
    )

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Flights", fmt_int(summary.get("total_flights", 0)), "Loaded flight records", "✈️", "#00d4ff")
    with c2:
        kpi_card("Delay Rate", f"{summary.get('delay_rate_pct', 0)}%", "Target: below 20%", "⏱️", "#ff1744")
    with c3:
        kpi_card("Avg Delay", f"{summary.get('avg_delay_minutes', 0)} min", "Among delayed flights", "⌛", "#ffd600")
    with c4:
        cargo_tons = summary.get("total_cargo_kg", 0) / 1000
        kpi_card("Cargo Handled", f"{cargo_tons:,.1f}T", f"{fmt_int(summary.get('total_baggage_pieces', 0))} bags", "📦", "#00e676")
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="chart-card"><div class="card-title">Flight Status Overview</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_status_pie(eda["status_dist"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="chart-card"><div class="card-title">Delay Rate Gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_delay_gauge(summary.get("delay_rate_pct", 0)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="chart-card"><div class="card-title">Delay by Hour</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_delay_heatmap(eda["delay_by_hour"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="chart-card"><div class="card-title">AI Recommendations</div>', unsafe_allow_html=True)
        for rec in recs[:3]:
            color = {"Critical": "#ff1744", "High": "#ff6b35", "Medium": "#ffd600", "Low": "#00e676"}.get(rec.priority, "#00d4ff")
            st.markdown(
                f"""
<div class="rec-card" style="--rec-color:{color}">
  <div class="rec-title">{rec.priority}: {rec.title}</div>
  <div class="rec-detail">{rec.detail}</div>
  <div class="rec-impact">Expected benefit: {rec.impact}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="card-title">Airline Performance Table</div>', unsafe_allow_html=True)
    st.dataframe(eda["delay_by_airline"], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


elif page == "Analytics":
    hero("Deep Analytics", "Gate congestion, top routes, cargo, baggage and delay correlations", "EDA ENGINE")

    tab1, tab2, tab3, tab4 = st.tabs(["Gate Congestion", "Routes", "Cargo & Baggage", "Delay Details"])
    with tab1:
        st.plotly_chart(chart_gate_congestion(eda["gate_congestion"]), use_container_width=True)
        st.dataframe(eda["gate_congestion"], use_container_width=True, hide_index=True)
    with tab2:
        st.plotly_chart(chart_top_routes(eda["top_routes"]), use_container_width=True)
        st.dataframe(eda["top_routes"], use_container_width=True, hide_index=True)
    with tab3:
        st.plotly_chart(chart_cargo_treemap(eda["cargo"]), use_container_width=True)
        if "Baggage_Count" in df.columns and "Delay_Minutes" in df.columns:
            st.plotly_chart(chart_baggage_vs_delay(df), use_container_width=True)
        st.dataframe(eda["cargo"], use_container_width=True, hide_index=True)
    with tab4:
        st.plotly_chart(chart_delay_by_airline(eda["delay_by_airline"]), use_container_width=True)
        st.plotly_chart(chart_delay_heatmap(eda["delay_by_hour"]), use_container_width=True)

elif page == "ML Predictor":
    hero("ML Delay Predictor", "Train a Random Forest model and test flight delay risk", "SKLEARN")

    if st.button("Train Delay Model", type="primary"):
        with st.spinner("Training Random Forest model..."):
            model, accuracy, importance, report, err = train_delay_model(df)
        if err:
            st.error(err)
        else:
            st.session_state.ml_model = model
            st.session_state.ml_accuracy = accuracy
            st.session_state.ml_importance = importance
            st.session_state.ml_report = report
            st.success(f"Model trained successfully. Accuracy: {accuracy * 100:.1f}%")

    if st.session_state.ml_accuracy is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="panel-card"><div class="card-title">Model Performance</div>', unsafe_allow_html=True)
            st.metric("Accuracy", f"{st.session_state.ml_accuracy * 100:.1f}%")
            if st.session_state.ml_report:
                st.code(st.session_state.ml_report)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card"><div class="card-title">Feature Importance</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_feature_importance(st.session_state.ml_importance), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-card"><div class="section-title">Live Prediction Form</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            airline = st.selectbox("Airline", sorted(df["Airline"].dropna().unique()) if "Airline" in df.columns else [""])
            origin = st.selectbox("Origin", sorted(df["Origin"].dropna().unique()) if "Origin" in df.columns else [""])
        with c2:
            destination = st.selectbox("Destination", sorted(df["Destination"].dropna().unique()) if "Destination" in df.columns else [""])
            gate = st.selectbox("Gate", sorted(df["Gate_Number"].dropna().unique()) if "Gate_Number" in df.columns else [""])
        with c3:
            hour = st.number_input("Departure Hour", min_value=0, max_value=23, value=14)
            cargo = st.number_input("Cargo Weight", min_value=0.0, value=float(df["Cargo_Weight"].mean()) if "Cargo_Weight" in df.columns else 1000.0)
            baggage = st.number_input("Baggage Count", min_value=0.0, value=float(df["Baggage_Count"].mean()) if "Baggage_Count" in df.columns else 100.0)

        if st.button("Predict Delay Risk"):
            input_df = pd.DataFrame([
                {
                    "Airline": airline,
                    "Origin": origin,
                    "Destination": destination,
                    "Gate_Number": gate,
                    "Departure_Hour": int(hour),
                    "Cargo_Weight": float(cargo),
                    "Baggage_Count": float(baggage),
                }
            ])
            try:
                pred = st.session_state.ml_model.predict(input_df)[0]
                proba = None
                if hasattr(st.session_state.ml_model, "predict_proba"):
                    proba = st.session_state.ml_model.predict_proba(input_df)[0][1]
                if pred == 1:
                    st.error(f"High delay risk detected{f' ({proba*100:.1f}%)' if proba is not None else ''}.")
                else:
                    st.success(f"Likely on time{f' ({(1-proba)*100:.1f}% confidence)' if proba is not None else ''}.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Click Train Delay Model to enable prediction and feature importance.")


elif page == "Reports":
    hero("Reports & Memory", "Build PDF reports and review previous analysis sessions", "REPORTLAB")

    st.markdown('<div class="panel-card"><div class="section-title">Report Builder</div>', unsafe_allow_html=True)
    include_preview = st.checkbox("Show Markdown preview", value=True)
    if include_preview:
        st.markdown(st.session_state.report_md)

    pdf_buffer = generate_pdf_report(st.session_state.report_md)
    st.download_button(
        "Download Full PDF Report",
        data=pdf_buffer,
        file_name="aerologix_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download Cleaned CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="aerologix_cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-card"><div class="section-title">Session Memory</div>', unsafe_allow_html=True)
    sessions = get_recent_sessions(limit=8)
    if sessions:
        st.dataframe(pd.DataFrame(sessions), use_container_width=True, hide_index=True)
    else:
        st.info("No saved sessions yet.")
    st.markdown('</div>', unsafe_allow_html=True)


elif page == "Passenger Alerts":
    hero("Passenger Alerts", "Search a flight, preview the message, and send WhatsApp updates to passengers.", "TWILIO")
    passenger_alerts_tab(df)

elif page == "AI Assistant":
    hero("AeroLogix AI Agent", "Ask natural-language questions with LangGraph workflow and SQLite memory", "LANGGRAPH")

    quick_questions = [
        "What are the peak delay hours and how should we staff?",
        "Which airline has the worst delay performance?",
        "Which gates are most congested?",
        "Which routes have the highest delay risk?",
        "Give passenger-friendly delay recommendations.",
    ]
    st.markdown('<div class="panel-card"><div class="section-title">Quick Questions</div>', unsafe_allow_html=True)
    cols = st.columns(len(quick_questions))
    for idx, q in enumerate(quick_questions):
        with cols[idx]:
            if st.button(q[:28] + "...", key=f"qq_{idx}", use_container_width=True):
                st.session_state.pending_question = q
    st.markdown('</div>', unsafe_allow_html=True)

    for message in st.session_state.chat_history:
        role_class = "chat-user" if message["role"] == "user" else "chat-ai"
        icon = "User" if message["role"] == "user" else "AeroLogix AI"
        st.markdown(f'<div class="{role_class}"><b>{icon}</b><br>{message["content"]}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Ask about delays, gates, cargo, routes or recommendations...")
    if not user_input and st.session_state.get("pending_question"):
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        save_message(st.session_state.session_id, "user", user_input)

        with st.spinner("AeroLogix AI is reasoning over your data..."):
            if raw_llm and LANGGRAPH_AVAILABLE:
                result = run_aerologix_agent(
                    question=user_input,
                    llm=raw_llm,
                    df_summary=prepare_agent_summary(eda, df),
                    session_id=st.session_state.lg_session_id,
                )
                answer = result.get("final_response", "No response generated.")
            elif agent_obj:
                answer = agent_obj.answer_question(user_input, [])
            else:
                answer = "LLM is not connected. Add GROQ_API_KEY or GOOGLE_API_KEY to your .env file."

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        save_message(st.session_state.session_id, "assistant", answer)
        st.rerun()

    st.markdown('<div class="panel-card"><div class="section-title">Visible Memory</div>', unsafe_allow_html=True)
    recent = get_recent_messages(st.session_state.session_id, limit=8)
    if recent:
        for row in recent:
            st.markdown(f"**{row['role'].title()}** · `{row['created_at'][:19]}`  ")
            st.caption(row["content"][:350] + ("..." if len(row["content"]) > 350 else ""))
            st.divider()
    else:
        st.info("No chat memory yet.")
    st.markdown('</div>', unsafe_allow_html=True)
