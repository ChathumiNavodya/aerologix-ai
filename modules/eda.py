"""
modules/eda.py
Exploratory Data Analysis — computes all key metrics for the dashboard.
"""

import pandas as pd
import numpy as np


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """High-level summary KPIs."""
    total = len(df)
    delayed = int(df["Is_Delayed"].sum()) if "Is_Delayed" in df.columns else 0
    on_time = total - delayed
    delay_rate = round(delayed / total * 100, 1) if total else 0

    avg_delay = 0.0
    max_delay = 0.0
    if "Delay_Minutes" in df.columns:
        delayed_only = df[df["Delay_Minutes"] > 0]["Delay_Minutes"]
        avg_delay = round(float(delayed_only.mean()), 1) if len(delayed_only) else 0.0
        max_delay = round(float(df["Delay_Minutes"].max()), 1)

    total_cargo = int(df["Cargo_Weight"].sum()) if "Cargo_Weight" in df.columns else 0
    total_baggage = int(df["Baggage_Count"].sum()) if "Baggage_Count" in df.columns else 0

    return {
        "total_flights": total,
        "delayed_flights": delayed,
        "on_time_flights": on_time,
        "delay_rate_pct": delay_rate,
        "avg_delay_minutes": avg_delay,
        "max_delay_minutes": max_delay,
        "total_cargo_kg": total_cargo,
        "total_baggage_pieces": total_baggage,
        "airlines_count": int(df["Airline"].nunique()) if "Airline" in df.columns else 0,
        "gates_used": int(df["Gate_Number"].nunique()) if "Gate_Number" in df.columns else 0,
    }


def delay_by_airline(df: pd.DataFrame) -> pd.DataFrame:
    if "Airline" not in df.columns or "Delay_Minutes" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("Airline").agg(
        Total_Flights=("Flight_ID", "count"),
        Avg_Delay=("Delay_Minutes", "mean"),
        Max_Delay=("Delay_Minutes", "max"),
        Delayed_Flights=("Is_Delayed", "sum"),
    ).reset_index()
    grp["Delay_Rate_%"] = (grp["Delayed_Flights"] / grp["Total_Flights"] * 100).round(1)
    grp["Avg_Delay"] = grp["Avg_Delay"].round(1)
    return grp.sort_values("Avg_Delay", ascending=False)


def delay_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    if "Hour_of_Day" not in df.columns or "Delay_Minutes" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("Hour_of_Day").agg(
        Flights=("Flight_ID", "count"),
        Avg_Delay=("Delay_Minutes", "mean"),
        Total_Delay=("Delay_Minutes", "sum"),
    ).reset_index()
    grp["Avg_Delay"] = grp["Avg_Delay"].round(1)
    return grp.sort_values("Hour_of_Day")


def gate_congestion(df: pd.DataFrame) -> pd.DataFrame:
    if "Gate_Number" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("Gate_Number").agg(
        Flights=("Flight_ID", "count"),
        Avg_Delay=("Delay_Minutes", "mean"),
        Total_Baggage=("Baggage_Count", "sum"),
        Total_Cargo=("Cargo_Weight", "sum"),
    ).reset_index()
    grp["Avg_Delay"] = grp["Avg_Delay"].round(1)
    grp["Congestion_Score"] = (
        grp["Flights"] * 0.5 + grp["Avg_Delay"] * 0.3 + grp["Total_Baggage"] / 100 * 0.2
    ).round(2)
    return grp.sort_values("Congestion_Score", ascending=False)


def cargo_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if "Airline" not in df.columns or "Cargo_Weight" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("Airline").agg(
        Total_Cargo_kg=("Cargo_Weight", "sum"),
        Avg_Cargo_kg=("Cargo_Weight", "mean"),
        Total_Baggage=("Baggage_Count", "sum"),
        Avg_Baggage=("Baggage_Count", "mean"),
    ).reset_index()
    grp["Avg_Cargo_kg"] = grp["Avg_Cargo_kg"].round(0)
    grp["Avg_Baggage"] = grp["Avg_Baggage"].round(0)
    return grp.sort_values("Total_Cargo_kg", ascending=False)


def flight_status_distribution(df: pd.DataFrame) -> pd.Series:
    if "Flight_Status" not in df.columns:
        return pd.Series()
    return df["Flight_Status"].value_counts()


def top_routes(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if "Origin" not in df.columns or "Destination" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["Route"] = df["Origin"] + " → " + df["Destination"]
    grp = df.groupby("Route").agg(
        Flights=("Flight_ID", "count"),
        Avg_Delay=("Delay_Minutes", "mean"),
        Avg_Baggage=("Baggage_Count", "mean"),
    ).reset_index()
    grp["Avg_Delay"] = grp["Avg_Delay"].round(1)
    grp["Avg_Baggage"] = grp["Avg_Baggage"].round(0)
    return grp.sort_values("Flights", ascending=False).head(top_n)


def build_full_eda(df: pd.DataFrame) -> dict:
    """Run all EDA functions and return a single results dict."""
    return {
        "summary": compute_summary_stats(df),
        "delay_by_airline": delay_by_airline(df),
        "delay_by_hour": delay_by_hour(df),
        "gate_congestion": gate_congestion(df),
        "cargo": cargo_analysis(df),
        "status_dist": flight_status_distribution(df),
        "top_routes": top_routes(df),
    }
