"""
Data ingestion utilities for loading CSV/Excel into the SQLite database.

Adapted from Text to SQL/ingestion.py.
"""

import re
from io import BytesIO

import pandas as pd
from sqlalchemy import inspect

from database import get_engine

SQL_RESERVED_WORDS = {
    "select", "from", "where", "join", "inner", "left", "right", "outer",
    "on", "and", "or", "not", "in", "is", "null", "like", "between",
    "order", "by", "group", "having", "limit", "offset", "distinct",
    "insert", "update", "delete", "create", "drop", "alter", "table",
    "database", "index", "view", "trigger", "procedure", "function",
    "default", "check", "primary", "key", "foreign", "unique", "constraint",
    "column", "schema", "cascade", "restrict", "set", "no", "action",
}


def clean_name(name: str) -> str:
    """Clean a name to be SQL-safe."""
    cleaned = str(name).lower().strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "column"
    if cleaned in SQL_RESERVED_WORDS:
        cleaned = cleaned + "_col"
    return cleaned


def read_csv(file_path_or_bytes) -> tuple[pd.DataFrame, dict]:
    if isinstance(file_path_or_bytes, bytes):
        df = pd.read_csv(BytesIO(file_path_or_bytes))
    elif hasattr(file_path_or_bytes, "read"):
        df = pd.read_csv(file_path_or_bytes)
    else:
        df = pd.read_csv(file_path_or_bytes)

    mapping = {}
    rename = {}
    for col in df.columns:
        cleaned = clean_name(col)
        mapping[col] = cleaned
        rename[col] = cleaned
    df = df.rename(columns=rename)
    return df, mapping


def read_excel(file_path_or_bytes) -> dict:
    if isinstance(file_path_or_bytes, bytes):
        excel_file = pd.ExcelFile(BytesIO(file_path_or_bytes))
    elif hasattr(file_path_or_bytes, "read"):
        excel_file = pd.ExcelFile(file_path_or_bytes)
    else:
        excel_file = pd.ExcelFile(file_path_or_bytes)

    sheets = {}
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        table_name = clean_name(sheet_name)
        mapping = {}
        rename = {}
        for col in df.columns:
            cleaned = clean_name(col)
            mapping[col] = cleaned
            rename[col] = cleaned
        df = df.rename(columns=rename)
        sheets[table_name] = {
            "dataframe": df,
            "column_mapping": mapping,
            "original_sheet_name": sheet_name,
        }
    return sheets


def save_to_sql(df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> bool:
    try:
        engine = get_engine()
        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
        return True
    except Exception as e:
        print(f"Error saving to SQL: {e}")
        return False


def ingest_csv(file_path_or_bytes, table_name: str | None = None) -> dict:
    if table_name is None:
        table_name = "csv_data"
    table_name = clean_name(table_name)
    df, column_mapping = read_csv(file_path_or_bytes)
    success = save_to_sql(df, table_name, if_exists="replace")
    return {
        "success": success,
        "table_name": table_name,
        "rows": len(df),
        "columns": list(df.columns),
        "column_mapping": column_mapping,
    }


def ingest_excel(file_path_or_bytes) -> dict:
    sheets_data = read_excel(file_path_or_bytes)
    report = {"success": True, "tables": []}
    for table_name, info in sheets_data.items():
        df = info["dataframe"]
        success = save_to_sql(df, table_name, if_exists="replace")
        report["tables"].append({
            "original_sheet_name": info["original_sheet_name"],
            "sql_table_name": table_name,
            "rows": len(df),
            "columns": list(df.columns),
            "column_mapping": info["column_mapping"],
            "success": success,
        })
    return report
