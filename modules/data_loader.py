"""
modules/data_loader.py
Handles loading of CSV, XLSX, and XLS airport logistics datasets.
"""

import pandas as pd
import streamlit as st
from pathlib import Path


EXPECTED_COLUMNS = [
    "Flight_ID", "Airline", "Origin", "Destination",
    "Departure_Time", "Arrival_Time", "Delay_Minutes",
    "Cargo_Weight", "Baggage_Count", "Gate_Number", "Flight_Status"
]

OPTIONAL_COLUMNS = ["Delay_Minutes", "Cargo_Weight", "Baggage_Count"]


def load_file(uploaded_file) -> "tuple[pd.DataFrame | None, str]":  # requires Python 3.10+
    """
    Load an uploaded file into a DataFrame.
    Returns (dataframe, error_message). On success, error_message is empty.
    """
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif filename.endswith(".xls"):
            df = pd.read_excel(uploaded_file, engine="xlrd")
        else:
            return None, f"Unsupported file format: {uploaded_file.name}"

        df = _normalize_columns(df)
        validation_msg = _validate_schema(df)
        if validation_msg:
            return None, validation_msg

        return df, ""
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def load_sample_data() -> pd.DataFrame:
    """Load the bundled sample dataset for demonstration."""
    # Try several candidate locations so the app works regardless of how it's launched
    candidates = [
        Path(__file__).parent.parent / "data" / "sample_airport_data.csv",
        Path(__file__).parent.parent / "sample_airport_data.csv",
        Path(__file__).parent / "sample_airport_data.csv",
        Path("data") / "sample_airport_data.csv",
        Path("sample_airport_data.csv"),
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            return _normalize_columns(df)
    raise FileNotFoundError(
        "sample_airport_data.csv not found. Tried: " + ", ".join(str(p) for p in candidates)
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalise column names."""
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def _validate_schema(df: pd.DataFrame) -> str:
    """
    Validate that required columns exist.
    Returns an error string or empty string if OK.
    """
    required = [c for c in EXPECTED_COLUMNS if c not in OPTIONAL_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return (
            f"Dataset is missing required columns: {missing}. "
            f"Expected columns: {EXPECTED_COLUMNS}"
        )
    return ""


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a concise summary dict of the dataset for logging/display."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "airlines": df["Airline"].nunique() if "Airline" in df.columns else 0,
        "flights": len(df),
        "date_range": _get_date_range(df),
    }


def _get_date_range(df: pd.DataFrame) -> str:
    try:
        if "Departure_Time" in df.columns:
            dates = pd.to_datetime(df["Departure_Time"], errors="coerce").dropna()
            if len(dates):
                return f"{dates.min().date()} → {dates.max().date()}"
    except Exception:
        pass
    return "N/A"