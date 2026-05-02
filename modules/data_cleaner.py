"""
modules/data_cleaner.py
Handles data cleaning, type coercion, and missing-value imputation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CleaningReport:
    original_rows: int = 0
    final_rows: int = 0
    duplicates_removed: int = 0
    missing_filled: dict = field(default_factory=dict)
    type_corrections: list = field(default_factory=list)
    outliers_flagged: int = 0
    notes: list = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "## 🧹 Data Cleaning Report",
            f"- **Original rows:** {self.original_rows}",
            f"- **Rows after cleaning:** {self.final_rows}",
            f"- **Duplicates removed:** {self.duplicates_removed}",
        ]
        if self.missing_filled:
            lines.append("- **Missing values filled:**")
            for col, count in self.missing_filled.items():
                lines.append(f"  - `{col}`: {count} values imputed")
        if self.type_corrections:
            lines.append(f"- **Type corrections:** {', '.join(self.type_corrections)}")
        if self.outliers_flagged:
            lines.append(f"- **Delay outliers flagged:** {self.outliers_flagged} rows")
        for note in self.notes:
            lines.append(f"- ⚠️ {note}")
        return "\n".join(lines)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Full cleaning pipeline. Returns (cleaned_df, CleaningReport).
    """
    report = CleaningReport(original_rows=len(df))
    df = df.copy()

    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    report.duplicates_removed = before - len(df)

    # 2. Parse datetime columns
    for col in ["Departure_Time", "Arrival_Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if df[col].isna().any():
                report.type_corrections.append(f"{col} (some unparseable)")
            else:
                report.type_corrections.append(f"{col} → datetime")

    # 3. Numeric columns
    num_cols = {"Delay_Minutes": 0.0, "Cargo_Weight": 0.0, "Baggage_Count": 0}
    for col, default in num_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Count NaNs AFTER coercion (captures both original NaNs and newly coerced ones)
            after_coerce_na = int(df[col].isna().sum())
            if after_coerce_na:
                report.missing_filled[col] = after_coerce_na
            df[col] = df[col].fillna(default)

    # 4. String columns – strip whitespace, title-case
    str_cols = ["Airline", "Origin", "Destination", "Gate_Number", "Flight_Status", "Flight_ID"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 5. Flight_Status normalisation
    if "Flight_Status" in df.columns:
        mapping = {
            "on time": "On Time", "ontime": "On Time",
            "delayed": "Delayed", "cancelled": "Cancelled",
            "diverted": "Diverted",
        }
        df["Flight_Status"] = df["Flight_Status"].str.lower().map(
            lambda x: mapping.get(x, x.title())
        )

    # 6. Flag delay outliers (> 3 std above mean)
    if "Delay_Minutes" in df.columns:
        mean_d = df["Delay_Minutes"].mean()
        std_d = df["Delay_Minutes"].std()
        if std_d > 0:
            outlier_mask = df["Delay_Minutes"] > mean_d + 3 * std_d
            df["Delay_Outlier"] = outlier_mask
            report.outliers_flagged = int(outlier_mask.sum())
        else:
            df["Delay_Outlier"] = False

    # 7. Derive helper columns
    if "Departure_Time" in df.columns and df["Departure_Time"].notna().any():
        df["Hour_of_Day"] = df["Departure_Time"].dt.hour
        df["Day_of_Week"] = df["Departure_Time"].dt.day_name()

    if "Delay_Minutes" in df.columns:
        df["Is_Delayed"] = df["Delay_Minutes"] > 0

    report.final_rows = len(df)
    if report.duplicates_removed:
        report.notes.append(f"Removed {report.duplicates_removed} duplicate flight records.")
    if "Departure_Time" in df.columns and df["Departure_Time"].isna().any():
        report.notes.append("Some Departure_Time values could not be parsed and are NaT.")

    return df, report