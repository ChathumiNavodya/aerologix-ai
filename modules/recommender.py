"""
modules/recommender.py
Rule-based + LLM-enhanced recommendation generation for airport logistics.
"""

import pandas as pd
from dataclasses import dataclass


PRIORITY_LEVELS = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}


@dataclass
class Recommendation:
    priority: str
    category: str
    title: str
    detail: str
    impact: str

    def to_markdown(self) -> str:
        icon = PRIORITY_LEVELS.get(self.priority, "⚪")
        return (
            f"### {icon} [{self.priority}] {self.title}\n"
            f"**Category:** {self.category}  \n"
            f"**Detail:** {self.detail}  \n"
            f"**Expected Impact:** {self.impact}"
        )


def generate_recommendations(eda_results: dict) -> list[Recommendation]:
    """
    Analyse EDA results and return a prioritised list of Recommendations.
    """
    recs: list[Recommendation] = []
    summary = eda_results.get("summary", {})
    delay_rate = summary.get("delay_rate_pct", 0)
    avg_delay = summary.get("avg_delay_minutes", 0)

    # ── Delay rate thresholds ──────────────────────────────────────────────────
    if delay_rate >= 50:
        recs.append(Recommendation(
            priority="Critical",
            category="Delay Management",
            title="Severe Delay Crisis — Immediate Action Required",
            detail=(
                f"{delay_rate:.1f}% of flights are delayed (avg {avg_delay:.0f} min). "
                "Deploy emergency ground teams, contact airlines for slot reallocation, "
                "and activate passenger communication protocols."
            ),
            impact="Could reduce delay-affected passengers by ~40% within 24 hours.",
        ))
    elif delay_rate >= 30:
        recs.append(Recommendation(
            priority="High",
            category="Delay Management",
            title="High Delay Rate — Operational Review Needed",
            detail=(
                f"Delay rate of {delay_rate:.1f}% exceeds the 25% industry benchmark. "
                "Review turnaround SLAs with ground handlers and identify root causes."
            ),
            impact="Targeted root-cause fixes can reduce delays by 30–50%.",
        ))
    elif delay_rate >= 20:
        recs.append(Recommendation(
            priority="Medium",
            category="Delay Management",
            title="Moderate Delays — Proactive Monitoring Advised",
            detail=(
                f"Delay rate {delay_rate:.1f}%. Introduce real-time delay dashboards "
                "for ground operations supervisors."
            ),
            impact="Early detection can prevent delay cascade by ~20%.",
        ))

    # ── Gate Congestion ────────────────────────────────────────────────────────
    gate_df: pd.DataFrame = eda_results.get("gate_congestion", pd.DataFrame())
    if not gate_df.empty:
        top_gate = gate_df.iloc[0]
        if top_gate["Congestion_Score"] > 50:
            recs.append(Recommendation(
                priority="High",
                category="Gate Management",
                title=f"Gate {top_gate['Gate_Number']} Critically Congested",
                detail=(
                    f"Gate {top_gate['Gate_Number']} has {int(top_gate['Flights'])} flights "
                    f"with avg delay {top_gate['Avg_Delay']:.0f} min. "
                    "Immediately reassign 2–3 flights to adjacent underutilised gates."
                ),
                impact="Reduces gate turnaround time by ~25% and passenger crowding.",
            ))
        congested_gates = gate_df[gate_df["Congestion_Score"] > 30]
        if len(congested_gates) > 2:
            recs.append(Recommendation(
                priority="Medium",
                category="Gate Management",
                title="Multiple Gates Showing High Congestion",
                detail=(
                    f"{len(congested_gates)} gates have high congestion scores. "
                    "Review gate allocation plan and redistribute low-traffic aircraft."
                ),
                impact="Balanced gate loads can cut average taxi time by 8–12 min.",
            ))

    # ── Peak-Hour Staffing ─────────────────────────────────────────────────────
    hour_df: pd.DataFrame = eda_results.get("delay_by_hour", pd.DataFrame())
    if not hour_df.empty:
        peak = hour_df.sort_values("Avg_Delay", ascending=False).head(3)
        hours_str = ", ".join(f"{int(h):02d}:00" for h in peak["Hour_of_Day"])
        recs.append(Recommendation(
            priority="Medium",
            category="Staffing",
            title="Increase Ground Staff During Peak Delay Hours",
            detail=(
                f"Highest average delays occur at {hours_str}. "
                "Deploy additional baggage handlers, fuelling crews, and gate agents "
                "during these windows."
            ),
            impact="Additional staff during peak hours reduces avg delay by 15–20 min.",
        ))

    # ── Cargo & Baggage ────────────────────────────────────────────────────────
    cargo_df: pd.DataFrame = eda_results.get("cargo", pd.DataFrame())
    if not cargo_df.empty and "Total_Cargo_kg" in cargo_df.columns:
        top_cargo_airline = cargo_df.iloc[0]["Airline"]
        avg_cargo = cargo_df["Avg_Cargo_kg"].mean()
        if avg_cargo > 12000:
            recs.append(Recommendation(
                priority="Medium",
                category="Cargo Operations",
                title="High Average Cargo Volume — Review Loading Procedures",
                detail=(
                    f"Average cargo load is {avg_cargo:,.0f} kg per flight. "
                    f"{top_cargo_airline} carries the most. "
                    "Upgrade cargo handling equipment and pre-stage pallets "
                    "to reduce aircraft turnaround times."
                ),
                impact="Faster cargo loading can shave 10–15 min off turnaround.",
            ))

    if summary.get("total_baggage_pieces", 0) > 0:
        total_bag = summary.get("total_baggage_pieces", 0)
        total_flt = summary.get("total_flights", 1)
        avg_bag = total_bag / max(total_flt, 1)
        if avg_bag > 250:
            recs.append(Recommendation(
                priority="Medium",
                category="Baggage Handling",
                title="Above-Average Baggage Volume Detected",
                detail=(
                    f"Average {avg_bag:.0f} bags per flight exceeds the 200-bag benchmark. "
                    "Open additional baggage carousels and assign dedicated sorters "
                    "for high-volume flights."
                ),
                impact="Reduces baggage claim wait times and mishandling incidents.",
            ))

    # ── Airline-specific delay ─────────────────────────────────────────────────
    airline_df: pd.DataFrame = eda_results.get("delay_by_airline", pd.DataFrame())
    if not airline_df.empty:
        worst = airline_df.iloc[0]
        if worst["Avg_Delay"] > 45:
            recs.append(Recommendation(
                priority="High",
                category="Airline Coordination",
                title=f"Coordinate Urgent Turnaround Review with {worst['Airline']}",
                detail=(
                    f"{worst['Airline']} averages {worst['Avg_Delay']:.0f} min delay "
                    f"across {int(worst['Total_Flights'])} flights "
                    f"({worst['Delay_Rate_%']:.0f}% delay rate). "
                    "Schedule a joint operations review and establish shared KPIs."
                ),
                impact="Joint SLA agreement typically cuts airline-specific delays by 30%.",
            ))

    # ── Scheduling ────────────────────────────────────────────────────────────
    recs.append(Recommendation(
        priority="Low",
        category="Scheduling",
        title="Stagger Departure Schedules to Reduce Peak-Hour Congestion",
        detail=(
            "Coordinate with airlines to redistribute departures away from peak congestion "
            "windows. A ±15-minute slot adjustment for 20% of flights can significantly "
            "relieve ground-side bottlenecks."
        ),
        impact="Smoothing traffic peaks reduces runway queuing by ~18%.",
    ))

    return recs


def recommendations_to_markdown(recs: list[Recommendation]) -> str:
    sections = ["# 🎯 AeroLogix AI — Operational Recommendations\n"]
    for priority in ["Critical", "High", "Medium", "Low"]:
        filtered = [r for r in recs if r.priority == priority]
        if filtered:
            icon = PRIORITY_LEVELS[priority]
            sections.append(f"## {icon} {priority} Priority\n")
            for r in filtered:
                sections.append(r.to_markdown() + "\n")
    return "\n".join(sections)