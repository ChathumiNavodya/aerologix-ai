"""
AeroLogix AI — LangGraph Agent Workflow
modules/agent_graph.py

Wraps the airport logistics AI assistant in a structured multi-node
StateGraph so every answer is classified, grounded in real data,
and stored in SQLite memory before being returned to Streamlit.
"""

from __future__ import annotations

import json
import sqlite3
import textwrap
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import END, StateGraph

# ── LangChain (already in your requirements) ────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

# ── Standard data libs ──────────────────────────────────────────────────────
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# 1.  Shared Graph State
# ════════════════════════════════════════════════════════════════════════════

class AeroLogixState(TypedDict):
    """All fields shared across every node in the workflow."""
    question: str
    category: str                       # classified question type
    df_summary: Dict[str, Any]          # pre-computed EDA summaries
    relevant_context: str               # subset of data fetched for this Q
    reasoning: str                      # LLM chain-of-thought
    recommendation: str                 # structured recommendation block
    final_response: str                 # clean numbered answer for Streamlit
    session_id: str
    error: Optional[str]


# ════════════════════════════════════════════════════════════════════════════
# 2.  Category constants
# ════════════════════════════════════════════════════════════════════════════

CATEGORIES = [
    "delay_analysis",
    "airline_analysis",
    "gate_analysis",
    "route_analysis",
    "cargo_baggage_analysis",
    "passenger_alert",
    "recommendation",
    "general_question",
]

# Maps category → which EDA table(s) to pull
CATEGORY_TABLE_MAP: Dict[str, List[str]] = {
    "delay_analysis":        ["delay_by_airline", "delay_by_hour"],
    "airline_analysis":      ["delay_by_airline"],
    "gate_analysis":         ["gate_congestion"],
    "route_analysis":        ["top_routes"],
    "cargo_baggage_analysis":["cargo"],
    "passenger_alert":       ["priority_delayed_flights", "delay_by_hour"],
    "recommendation":        ["delay_by_airline", "gate_congestion", "top_routes"],
    "general_question":      [],
}


# ════════════════════════════════════════════════════════════════════════════
# 3.  SQLite memory helper
# ════════════════════════════════════════════════════════════════════════════

def _init_db(db_path: str = "aerologix_memory.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_memory (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            question    TEXT    NOT NULL,
            category    TEXT    NOT NULL,
            answer      TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def _save_to_db(
    session_id: str,
    question: str,
    category: str,
    answer: str,
    db_path: str = "aerologix_memory.db",
) -> None:
    conn = _init_db(db_path)
    conn.execute(
        "INSERT INTO conversation_memory (session_id, question, category, answer, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, question, category, answer, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_memory(session_id: str, limit: int = 5, db_path: str = "aerologix_memory.db") -> List[Dict]:
    """Retrieve the most recent interactions for a session (used by Streamlit)."""
    try:
        conn = _init_db(db_path)
        rows = conn.execute(
            "SELECT question, category, answer, timestamp FROM conversation_memory "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        conn.close()
        return [
            {"question": r[0], "category": r[1], "answer": r[2], "timestamp": r[3]}
            for r in rows
        ]
    except Exception:
        return []


# ════════════════════════════════════════════════════════════════════════════
# 4.  Node implementations
# ════════════════════════════════════════════════════════════════════════════

def make_llm_call(llm: BaseChatModel, system: str, user: str) -> str:
    """Thin wrapper — returns the text content of an LLM response."""
    try:
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        return response.content.strip()
    except Exception as exc:
        return f"[LLM error: {exc}]"


# ── Node 1: classify_question ────────────────────────────────────────────────

def classify_question(state: AeroLogixState, llm: BaseChatModel) -> AeroLogixState:
    """
    Detect the intent of the user's question and assign a category.
    Returns the state with `category` set.
    """
    system = textwrap.dedent(f"""
        You are an intent classifier for an airport logistics AI assistant.
        Classify the user's question into EXACTLY ONE of these categories:
        {json.dumps(CATEGORIES, indent=2)}

        Rules:
        - Reply with ONLY the category string, nothing else.
        - If the question mentions delays → "delay_analysis"
        - If the question mentions a specific airline → "airline_analysis"
        - If the question mentions gates or terminals → "gate_analysis"
        - If the question mentions routes or destinations → "route_analysis"
        - If the question mentions cargo, baggage, or luggage → "cargo_baggage_analysis"
        - If the question asks about alerts or notifications for passengers → "passenger_alert"
        - If the question asks for improvements or suggestions → "recommendation"
        - Otherwise → "general_question"
    """)
    raw = make_llm_call(llm, system, state["question"])
    # Sanitise: pick the first matching category token
    category = "general_question"
    for cat in CATEGORIES:
        if cat in raw.lower():
            category = cat
            break

    return {**state, "category": category}


# ── Node 2: extract_data_context ─────────────────────────────────────────────

def extract_data_context(state: AeroLogixState) -> AeroLogixState:
    """
    Pull the relevant EDA tables from df_summary based on the category
    and format them as a readable string for downstream nodes.
    """
    tables_needed = CATEGORY_TABLE_MAP.get(state["category"], [])
    df_summary = state.get("df_summary", {})
    chunks: List[str] = []

    for table_key in tables_needed:
        data = df_summary.get(table_key)
        if data is None:
            continue
        if isinstance(data, pd.DataFrame):
            chunks.append(f"[{table_key}]\n{data.to_string(index=False)}")
        elif isinstance(data, dict):
            chunks.append(f"[{table_key}]\n{json.dumps(data, indent=2, default=str)}")
        elif isinstance(data, str):
            chunks.append(f"[{table_key}]\n{data}")
        else:
            try:
                chunks.append(f"[{table_key}]\n{str(data)}")
            except Exception:
                pass

    context = "\n\n".join(chunks) if chunks else "No specific data table found for this query."
    return {**state, "relevant_context": context}


# ── Node 3: reason_over_data ─────────────────────────────────────────────────

def reason_over_data(state: AeroLogixState, llm: BaseChatModel) -> AeroLogixState:
    """
    Use the LLM to interpret the data evidence and produce a reasoning narrative.
    """
    system = textwrap.dedent("""
        You are an expert airport operations analyst.
        You will be given a user question and relevant data tables.
        Produce a concise, evidence-based analysis (3-5 sentences).
        - Reference specific numbers from the data.
        - Highlight the most important pattern or anomaly.
        - Do NOT make up figures that are not in the data.
        - Write in plain language suitable for airport staff or passengers.
    """)
    user = textwrap.dedent(f"""
        Question: {state['question']}
        Category: {state['category']}

        Data Context:
        {state['relevant_context']}

        Provide your reasoning:
    """)
    reasoning = make_llm_call(llm, system, user)
    return {**state, "reasoning": reasoning}


# ── Node 4: generate_recommendation ──────────────────────────────────────────

def generate_recommendation(state: AeroLogixState, llm: BaseChatModel) -> AeroLogixState:
    """
    Produce a structured recommendation block if the category warrants it.
    For general_question this is skipped (returns a placeholder).
    """
    if state["category"] == "general_question":
        return {**state, "recommendation": ""}

    system = textwrap.dedent("""
        You are an airport operations advisor.
        Based on the analysis, produce ONE structured recommendation in this exact format:

        Issue:
        <one sentence describing the core problem>

        Evidence:
        <key data points that confirm the issue>

        Action:
        <specific, actionable step airport management should take>

        Expected Benefit:
        <measurable outcome if the action is implemented>

        Be concise and specific. Do not add any extra text outside the four fields.
    """)
    user = textwrap.dedent(f"""
        Question: {state['question']}
        Category: {state['category']}
        Reasoning: {state['reasoning']}
        Data Context:
        {state['relevant_context']}
    """)
    recommendation = make_llm_call(llm, system, user)
    return {**state, "recommendation": recommendation}


# ── Node 5: save_memory ───────────────────────────────────────────────────────

def save_memory(state: AeroLogixState) -> AeroLogixState:
    """Persist the Q&A pair to SQLite."""
    try:
        _save_to_db(
            session_id=state["session_id"],
            question=state["question"],
            category=state["category"],
            answer=state.get("final_response", ""),
        )
    except Exception as exc:
        # Non-fatal — log but don't crash the workflow
        print(f"[AeroLogix] Memory save failed: {exc}")
    return state


# ── Node 6: final_answer ──────────────────────────────────────────────────────

def final_answer(state: AeroLogixState, llm: BaseChatModel) -> AeroLogixState:
    """
    Assemble the reasoning + recommendation into a clean numbered answer
    ready for display in Streamlit.
    """
    rec_block = (
        f"\n\n📋 **Recommendation**\n{state['recommendation']}"
        if state.get("recommendation")
        else ""
    )

    system = textwrap.dedent("""
        You are AeroLogix AI, the airport operations assistant.
        Rewrite the provided analysis into a polished, numbered response for
        airport staff or passengers. Requirements:
        - Start with a one-sentence direct answer.
        - Use numbered points (1. 2. 3. …) for supporting evidence.
        - Keep each point concise (≤ 2 sentences).
        - End with a brief summary line.
        - Use plain language — no jargon.
        - Do NOT add made-up data. Only use what is in the analysis.
    """)
    user = textwrap.dedent(f"""
        Question: {state['question']}

        Analysis:
        {state['reasoning']}
        {rec_block}
    """)
    final = make_llm_call(llm, system, user)
    return {**state, "final_response": final}


# ════════════════════════════════════════════════════════════════════════════
# 5.  Graph builder
# ════════════════════════════════════════════════════════════════════════════

def build_aerologix_graph(llm: BaseChatModel) -> Any:
    """
    Construct and compile the AeroLogix StateGraph.

    Parameters
    ----------
    llm : BaseChatModel
        Any LangChain-compatible chat model (Groq, Gemini, OpenAI, …).

    Returns
    -------
    CompiledGraph
        Call `.invoke(initial_state)` to run the full workflow.
    """

    # ── Bind the LLM into each node that needs it via closures ──────────────
    def _classify(state):      return classify_question(state, llm)
    def _extract(state):       return extract_data_context(state)
    def _reason(state):        return reason_over_data(state, llm)
    def _recommend(state):     return generate_recommendation(state, llm)
    def _save(state):          return save_memory(state)
    def _finalize(state):      return final_answer(state, llm)

    graph = StateGraph(AeroLogixState)

    # ── Add nodes ────────────────────────────────────────────────────────────
    graph.add_node("classify_question",      _classify)
    graph.add_node("extract_data_context",   _extract)
    graph.add_node("reason_over_data",       _reason)
    graph.add_node("generate_recommendation",_recommend)
    graph.add_node("final_answer",           _finalize)
    graph.add_node("save_memory",            _save)

    # ── Wire edges (linear pipeline) ─────────────────────────────────────────
    graph.set_entry_point("classify_question")
    graph.add_edge("classify_question",       "extract_data_context")
    graph.add_edge("extract_data_context",    "reason_over_data")
    graph.add_edge("reason_over_data",        "generate_recommendation")
    graph.add_edge("generate_recommendation", "final_answer")
    graph.add_edge("final_answer",            "save_memory")
    graph.add_edge("save_memory",             END)

    return graph.compile()


# ════════════════════════════════════════════════════════════════════════════
# 6.  Public helper — called from app.py
# ════════════════════════════════════════════════════════════════════════════

def run_aerologix_agent(
    question: str,
    llm: BaseChatModel,
    df_summary: Dict[str, Any],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Entry point for Streamlit.

    Parameters
    ----------
    question   : User's raw question string.
    llm        : Initialised LangChain chat model.
    df_summary : Dict of pre-computed EDA DataFrames / dicts keyed by table name.
                 Expected keys (any subset):
                   "delay_by_airline", "delay_by_hour", "gate_congestion",
                   "top_routes", "cargo", "priority_delayed_flights"
    session_id : Optional session identifier; auto-generated if not provided.

    Returns
    -------
    dict with keys:
        final_response   – the polished answer to show in the UI
        category         – detected question type
        recommendation   – structured recommendation block (may be empty)
        session_id       – session identifier
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    initial_state: AeroLogixState = {
        "question":          question,
        "category":          "",
        "df_summary":        df_summary,
        "relevant_context":  "",
        "reasoning":         "",
        "recommendation":    "",
        "final_response":    "",
        "session_id":        session_id,
        "error":             None,
    }

    compiled_graph = build_aerologix_graph(llm)

    try:
        result: AeroLogixState = compiled_graph.invoke(initial_state)
    except Exception as exc:
        return {
            "final_response":  f"⚠️ Agent error: {exc}",
            "category":        "general_question",
            "recommendation":  "",
            "session_id":      session_id,
        }

    return {
        "final_response":  result.get("final_response", "No response generated."),
        "category":        result.get("category", "general_question"),
        "recommendation":  result.get("recommendation", ""),
        "session_id":      session_id,
    }