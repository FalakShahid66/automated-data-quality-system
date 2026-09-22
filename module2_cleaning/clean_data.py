"""
Module 2 — Advanced Data Cleaning & Transformation Pipeline
CadetX AI Engineering — Automated Data Quality & Validation System

Fixes what Module 1 found: missing values, duplicate rows, and
inconsistent formats. Scores data quality before and after cleaning
so the improvement is measurable.

Output: data/processed/<table>_cleaned.csv + reports/cleaning_log.json
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.impute import KNNImputer

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOG_PATH = Path("reports/cleaning_log.json")

# Columns that identify each table so ID-like fields are never imputed.
ID_LIKE_SUFFIXES = ("ID", "Id")


def quality_score(df: pd.DataFrame) -> float:
    """Simple 0-100 score: share of cells that are non-null and non-duplicate-row."""
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 100.0
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    duplicate_cells = duplicate_rows * df.shape[1]
    bad_cells = missing_cells + duplicate_cells
    return round(100 * (1 - bad_cells / total_cells), 2)


def is_id_column(col: str) -> bool:
    return col.endswith(ID_LIKE_SUFFIXES)


def clean_table(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before_score = quality_score(df)
    before_rows = len(df)
    actions = []

    # 1. Drop exact duplicate rows.
    dup_count = int(df.duplicated().sum())
    if dup_count:
        df = df.drop_duplicates().reset_index(drop=True)
        actions.append(f"Dropped {dup_count} exact duplicate row(s).")

    # 2. Impute missing values.
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing == 0 or is_id_column(col):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # ML-based imputation for numeric columns: KNN using other
            # numeric columns in the same table as context.
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) > 1:
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                actions.append(f"KNN-imputed {missing} missing value(s) in '{col}'.")
            else:
                median = df[col].median()
                df[col] = df[col].fillna(median)
                actions.append(f"Median-imputed {missing} missing value(s) in '{col}'.")
        else:
            # Text/categorical columns: fill with an explicit placeholder
            # rather than guessing a name or address that was never given.
            df[col] = df[col].fillna("UNKNOWN")
            actions.append(f"Filled {missing} missing value(s) in '{col}' with 'UNKNOWN'.")

    # 3. Normalise obvious text formatting issues (extra whitespace, casing
    # left as-is since names/cities are case-sensitive by nature).
    text_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in text_cols:
        stripped = df[col].astype(str).str.strip()
        if not stripped.equals(df[col].astype(str)):
            df[col] = stripped
            actions.append(f"Stripped leading/trailing whitespace in '{col}'.")

    after_score = quality_score(df)

    log_entry = {
        "table": name,
        "rows_before": before_rows,
        "rows_after": len(df),
        "duplicate_rows_removed": dup_count,
        "quality_score_before": before_score,
        "quality_score_after": after_score,
        "quality_delta": round(after_score - before_score, 2),
        "actions": actions,
    }
    return df, log_entry


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Expected dataset at {DATA_DIR}/ — run Module 1 setup first.")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}/")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaning_log = {"tables": {}}

    for csv_path in csv_files:
        table_name = csv_path.stem
        df = pd.read_csv(csv_path)
        cleaned_df, log_entry = clean_table(table_name, df)
        cleaning_log["tables"][table_name] = log_entry

        out_path = PROCESSED_DIR / f"{table_name}_cleaned.csv"
        cleaned_df.to_csv(out_path, index=False)
        print(
            f"Cleaned {table_name}: {log_entry['quality_score_before']} -> "
            f"{log_entry['quality_score_after']} "
            f"({len(log_entry['actions'])} action(s))"
        )

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(cleaning_log, f, indent=2, default=str)

    print(f"\nCleaned files written to {PROCESSED_DIR}/")
    print(f"Cleaning log written to {LOG_PATH}")


if __name__ == "__main__":
    main()
