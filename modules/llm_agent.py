"""
modules/llm_agent.py
LangChain-based AI agent logic using Groq (primary) or Google Gemini (fallback).
Handles planning, reasoning, memory-aware Q&A, and clear point-form answers over airport data.
"""

import os
import json
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


SYSTEM_PROMPT = """
You are AeroLogix AI, an expert airport logistics analytics agent.

Your job is to answer airport logistics questions using ONLY the structured dataset context
provided to you. You analyze flight delays, passenger delay risk, airline performance,
gate congestion, cargo, baggage, route issues, and staffing needs.

STRICT ANSWER RULES:
1. ALWAYS answer in clear numbered points.
2. ALWAYS use headings.
3. ALWAYS include exact evidence from the dataset when relevant.
4. NEVER say "data is unavailable" if the value exists in the structured context.
5. NEVER give vague or general-only answers.
6. If the user asks about delays, include delay rate, average delay, max delay, or delayed flight count when relevant.
7. If the user asks about peak hours, use the DELAY BY HOUR TABLE.
8. If the user asks about airlines, use the AIRLINE DELAY TABLE.
9. If the user asks about gates, use the GATE CONGESTION TABLE.
10. If the user asks about routes, use the TOP ROUTES TABLE.
11. If the user asks about passengers, explain the operational impact in passenger-friendly language.
12. Every recommendation must include:
    - Issue
    - Evidence from data
    - Action
    - Expected benefit

DEFAULT FORMAT:
### [Clear Analysis Title]

1. Key finding
Evidence: [specific number/table value]
Explanation: [what it means]

2. Key finding
Evidence: [specific number/table value]
Explanation: [what it means]

3. Key finding
Evidence: [specific number/table value]
Explanation: [what it means]

### Recommendations
1. Action: ...
   Expected benefit: ...

2. Action: ...
   Expected benefit: ...

Use urgency icons where useful:
🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low.
"""


def _build_llm(provider: str = "auto"):
    """Build the LLM client based on available API keys."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GOOGLE_API_KEY", "").strip()

    placeholders = {
        "",
        "your_groq_api_key_here",
        "your_google_api_key_here",
        "your_gemini_api_key_here",
    }

    groq_valid = groq_key not in placeholders
    gemini_valid = gemini_key not in placeholders

    if provider == "groq" or (provider == "auto" and groq_valid):
        if not groq_valid:
            raise ValueError("GROQ_API_KEY is missing or not set in your .env file.")
        return ChatGroq(
            api_key=groq_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=2048,
        ), "Groq (LLaMA 3.3 70B)"

    if provider == "gemini" or (provider == "auto" and gemini_valid):
        if not gemini_valid:
            raise ValueError("GOOGLE_API_KEY is missing or not set in your .env file.")
        return ChatGoogleGenerativeAI(
            google_api_key=gemini_key,
            model="gemini-2.0-flash",
            temperature=0.2,
            max_output_tokens=2048,
        ), "Google Gemini 2.0 Flash"

    raise ValueError("No LLM API key found. Set GROQ_API_KEY or GOOGLE_API_KEY in your .env file.")


class AeroLogixAgent:
    """Autonomous airport logistics AI agent with planning, reasoning, and memory."""

    def __init__(self, provider: str = "auto"):
        self.llm, self.model_name = _build_llm(provider)
        self._context: dict = {}

    def _table_text(self, table: pd.DataFrame, max_rows: int = 15) -> str:
        """Convert a DataFrame into compact text for LLM context."""
        if table is None or table.empty:
            return "No rows calculated."
        return table.head(max_rows).to_string(index=False)

    def _build_delay_by_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create delay-by-hour table directly from Departure_Time and Delay_Minutes."""
        required = {"Departure_Time", "Delay_Minutes"}
        if not required.issubset(set(df.columns)):
            return pd.DataFrame()

        temp = df.copy()
        temp["Departure_Time"] = pd.to_datetime(temp["Departure_Time"], errors="coerce")
        temp["Hour"] = temp["Departure_Time"].dt.hour
        temp["Is_Delayed"] = temp["Delay_Minutes"] > 0
        temp = temp.dropna(subset=["Hour"])

        if temp.empty:
            return pd.DataFrame()

        hourly = (
            temp.groupby("Hour")
            .agg(
                Flights=("Flight_ID", "count") if "Flight_ID" in temp.columns else ("Delay_Minutes", "count"),
                Delayed_Flights=("Is_Delayed", "sum"),
                Avg_Delay=("Delay_Minutes", "mean"),
                Max_Delay=("Delay_Minutes", "max"),
            )
            .reset_index()
        )

        hourly["Delay_Rate_%"] = (hourly["Delayed_Flights"] / hourly["Flights"] * 100).round(1)
        hourly["Avg_Delay"] = hourly["Avg_Delay"].round(1)
        hourly = hourly.sort_values(["Avg_Delay", "Delay_Rate_%"], ascending=False)
        return hourly

    def _build_priority_flights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build passenger support priority flights using highest delay."""
        if "Delay_Minutes" not in df.columns:
            return pd.DataFrame()

        cols = [
            c for c in [
                "Flight_ID", "Airline", "Origin", "Destination",
                "Departure_Time", "Arrival_Time", "Delay_Minutes",
                "Baggage_Count", "Gate_Number", "Flight_Status"
            ]
            if c in df.columns
        ]

        priority = df[df["Delay_Minutes"] > 0][cols].copy()
        if priority.empty:
            return pd.DataFrame()

        priority = priority.sort_values("Delay_Minutes", ascending=False).head(10)
        return priority

    def set_data_context(self, df: pd.DataFrame, eda_results: dict, cleaning_report_md: str):
        """Load rich structured dataset context so the agent can answer with exact values."""
        summary = eda_results.get("summary", {})
        delay_df: pd.DataFrame = eda_results.get("delay_by_airline", pd.DataFrame())
        gate_df: pd.DataFrame = eda_results.get("gate_congestion", pd.DataFrame())
        route_df: pd.DataFrame = eda_results.get("top_routes", pd.DataFrame())
        cargo_df: pd.DataFrame = eda_results.get("cargo", pd.DataFrame())
        status_df: pd.DataFrame = eda_results.get("status_dist", pd.DataFrame())

        delay_by_hour_df = eda_results.get("delay_by_hour", pd.DataFrame())
        if delay_by_hour_df is None or delay_by_hour_df.empty:
            delay_by_hour_df = self._build_delay_by_hour(df)

        priority_flights_df = self._build_priority_flights(df)

        delayed_flights = int((df["Delay_Minutes"] > 0).sum()) if "Delay_Minutes" in df.columns else 0
        on_time_flights = int((df["Delay_Minutes"] <= 0).sum()) if "Delay_Minutes" in df.columns else 0
        avg_delay_all = round(float(df["Delay_Minutes"].mean()), 1) if "Delay_Minutes" in df.columns else 0
        avg_delay_delayed = (
            round(float(df.loc[df["Delay_Minutes"] > 0, "Delay_Minutes"].mean()), 1)
            if "Delay_Minutes" in df.columns and delayed_flights > 0
            else 0
        )

        context_parts = [
            "IMPORTANT: Use ONLY this structured context. Do not say a value is unavailable if it is listed here.",
            "",
            "CORE KPI SUMMARY",
            f"Total flights: {summary.get('total_flights', len(df))}",
            f"Delayed flights: {delayed_flights}",
            f"On-time flights: {on_time_flights}",
            f"Delay rate: {summary.get('delay_rate_pct', round((delayed_flights / len(df) * 100), 1) if len(df) else 0)}%",
            f"Average delay across all flights: {avg_delay_all} minutes",
            f"Average delay among delayed flights: {avg_delay_delayed} minutes",
            f"Maximum delay: {summary.get('max_delay_minutes', df['Delay_Minutes'].max() if 'Delay_Minutes' in df.columns else 0)} minutes",
            f"Total cargo: {summary.get('total_cargo_kg', df['Cargo_Weight'].sum() if 'Cargo_Weight' in df.columns else 0):,} kg",
            f"Total baggage: {summary.get('total_baggage_pieces', df['Baggage_Count'].sum() if 'Baggage_Count' in df.columns else 0):,} pieces",
            f"Number of airlines: {summary.get('airlines_count', df['Airline'].nunique() if 'Airline' in df.columns else 0)}",
            f"Gates used: {summary.get('gates_used', df['Gate_Number'].nunique() if 'Gate_Number' in df.columns else 0)}",
            "",
            "AIRLINE DELAY TABLE",
            self._table_text(delay_df),
            "",
            "DELAY BY HOUR TABLE",
            self._table_text(delay_by_hour_df),
            "",
            "PASSENGER SUPPORT PRIORITY FLIGHTS",
            self._table_text(priority_flights_df),
            "",
            "GATE CONGESTION TABLE",
            self._table_text(gate_df),
            "",
            "TOP ROUTES TABLE",
            self._table_text(route_df),
            "",
            "CARGO AND BAGGAGE TABLE",
            self._table_text(cargo_df),
            "",
            "FLIGHT STATUS DISTRIBUTION",
            self._table_text(status_df),
            "",
            "DATA CLEANING SUMMARY",
            cleaning_report_md,
        ]

        self._context = {
            "summary_text": "\n".join(context_parts),
            "cleaning_report": cleaning_report_md,
            "columns": list(df.columns),
        }

    def _build_system_with_context(self) -> str:
        parts = [SYSTEM_PROMPT]
        if self._context:
            parts.append("\n\n--- CURRENT DATASET CONTEXT ---")
            parts.append(self._context.get("summary_text", ""))
            parts.append(f"Available columns: {', '.join(self._context.get('columns', []))}")
        return "\n".join(parts)

    def plan_analysis(self, filename: str, row_count: int) -> str:
        """Generate an analysis plan after dataset upload."""
        prompt = f"""
A new airport logistics dataset '{filename}' with {row_count} rows has been uploaded.

Create a structured AI analysis plan in 6-8 numbered points.

Use this format:
### AI Analysis Plan

1. Step name
Purpose: ...
Output: ...

2. Step name
Purpose: ...
Output: ...
"""
        return self._invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])

    def explain_patterns(self, eda_results: dict) -> str:
        """Generate natural-language reasoning about EDA findings."""
        summary = eda_results.get("summary", {})
        prompt = f"""
Create an executive summary using clear numbered points.

Dataset metrics:
{json.dumps(summary, indent=2)}

Use this format:
### AI Executive Summary

1. Overall operational health
Evidence: ...
Explanation: ...

2. Delay performance
Evidence: ...
Explanation: ...

3. Cargo and baggage status
Evidence: ...
Explanation: ...

4. Gate utilisation
Evidence: ...
Explanation: ...

### Immediate Action Items
1. ...
2. ...
3. ...
"""
        return self._invoke([SystemMessage(content=self._build_system_with_context()), HumanMessage(content=prompt)])

    def answer_question(self, question: str, conversation_history: list[dict]) -> str:
        """Answer a natural language question using structured dataset context and conversation history."""
        messages = [SystemMessage(content=self._build_system_with_context())]

        for msg in conversation_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            else:
                messages.append(AIMessage(content=msg.get("content", "")))

        final_question = f"""
User question:
{question}

Answer requirements:
- Use clear headings.
- Use numbered points.
- Use exact values from the structured dataset context.
- Do not say data is unavailable if it appears in the context.
- For recommendations, use: Issue, Evidence from data, Action, Expected benefit.
- For passenger questions, explain passenger impact clearly.
- For peak delay/staffing questions, use the DELAY BY HOUR TABLE.
"""
        messages.append(HumanMessage(content=final_question))
        return self._invoke(messages)

    def generate_insight(self, category: str, data_snippet: str) -> str:
        """Generate a focused insight for a specific category."""
        prompt = f"""
Based on this {category} data for an airport:

{data_snippet}

Answer in this format:

### {category.title()} Insight

1. Key finding
Evidence: ...
Explanation: ...

2. Risk or opportunity
Evidence: ...
Explanation: ...

### Recommendation
Action: ...
Expected benefit: ...
"""
        return self._invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])

    def _invoke(self, messages) -> str:
        """Safe LLM invocation with error handling."""
        try:
            result = self.llm.invoke(messages)
            return result.content
        except Exception as e:
            return f"⚠️ Agent error: {str(e)}. Please check your API key configuration."
