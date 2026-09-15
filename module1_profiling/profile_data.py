"""
Module 1 — Advanced Data Profiling & Metadata Intelligence
CadetX AI Engineering — Automated Data Quality & Validation System

Profiles every table in the dataset before any cleaning or ML happens:
infers column types, maps missing values, detects mixed-type columns,
flags likely PII columns, and summarises distributions.

Output: reports/profiling_report.json
"""

import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("reports/profiling_report.json")

# Column name patterns that strongly suggest personally identifiable data.
PII_PATTERNS = [
    r"first.?name", r"last.?name", r"^name$", r"email", r"phone",
    r"address", r"street", r"city", r"ssn", r"date.?of.?birth", r"dob",
]


def infer_semantic_type(column_name: str, series: pd.Series) -> str:
    """Guess what kind of real-world thing a column holds, beyond its dtype."""
    name = column_name.lower()

    if re.search(r"email", name):
        return "email"
    if re.search(r"phone", name):
        return "phone"
    if re.search(r"date|_dt$", name) or "datetime" in name:
        return "date"
    if name.endswith("id") or name.endswith("_id"):
        return "identifier"
    if re.search(r"name", name):
        return "name"
    if re.search(r"amount|balance|rate|price", name):
        return "monetary"
    if re.search(r"street|city|address|country", name):
        return "address_component"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "text"


def is_likely_pii(column_name: str) -> bool:
    name = column_name.lower()
    return any(re.search(pat, name) for pat in PII_PATTERNS)


def detect_mixed_types(series: pd.Series) -> bool:
    """A column is 'mixed type' if, ignoring nulls, it contains more than
    one underlying Python type (e.g. some rows numeric, some rows text)."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    types_seen = non_null.map(type).unique()
    return len(types_seen) > 1


def profile_column(column_name: str, series: pd.Series, total_rows: int) -> dict:
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    col_profile = {
        "dtype": str(series.dtype),
        "semantic_type": infer_semantic_type(column_name, series),
        "is_likely_pii": is_likely_pii(column_name),
        "missing_count": missing_count,
        "missing_pct": round(100 * missing_count / total_rows, 2) if total_rows else 0,
        "unique_count": unique_count,
        "unique_pct": round(100 * unique_count / total_rows, 2) if total_rows else 0,
        "is_mixed_type": detect_mixed_types(series),
    }

    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if not non_null.empty:
            col_profile["stats"] = {
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": round(float(non_null.mean()), 4),
                "std": round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0,
            }

    return col_profile


def profile_table(name: str, df: pd.DataFrame) -> dict:
    total_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())

    columns = {
        col: profile_column(col, df[col], total_rows) for col in df.columns
    }

    pii_columns = [c for c, p in columns.items() if p["is_likely_pii"]]

    return {
        "table": name,
        "row_count": total_rows,
        "column_count": len(df.columns),
        "duplicate_row_count": duplicate_rows,
        "duplicate_row_pct": round(100 * duplicate_rows / total_rows, 2) if total_rows else 0,
        "flagged_pii_columns": pii_columns,
        "columns": columns,
    }


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Expected dataset at {DATA_DIR}/ — place the CSV files there first.")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}/")

    report = {"source_directory": str(DATA_DIR), "tables": {}}

    for csv_path in csv_files:
        table_name = csv_path.stem
        df = pd.read_csv(csv_path)
        report["tables"][table_name] = profile_table(table_name, df)
        print(f"Profiled {table_name}: {len(df)} rows, {len(df.columns)} columns")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nProfiling report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
