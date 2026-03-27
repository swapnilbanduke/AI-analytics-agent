"""
SQL Agent — NL2SQL pipeline with schema inspection, generation,
validation, execution, and error recovery.

Adapted from Text to SQL/workflow.py.
"""

import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from config import build_chat_model, get_default_model
from database import get_sql_database, list_tables, run_query
from prompts import SQL_ANSWER_PROMPT, SQL_GENERATION_PROMPT

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "TRUNCATE", "CREATE", "REPLACE", "GRANT", "REVOKE",
    "DENY", "EXEC", "EXECUTE",
]

MAX_RESULT_ROWS = 100


def is_sql_safe(sql: str) -> tuple[bool, str | None]:
    """Validate SQL for safety — only SELECT queries allowed."""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed."
    for kw in FORBIDDEN_KEYWORDS:
        if kw in sql_upper:
            return False, f"Query contains forbidden operation: {kw}."
    if "--" in sql or "/*" in sql:
        return False, "SQL comments are not allowed."
    return True, None


def enforce_limit(sql: str) -> str:
    """Ensure SQL has a LIMIT clause, capped at MAX_RESULT_ROWS."""
    sql_upper = sql.upper()
    if "LIMIT" not in sql_upper:
        return f"{sql} LIMIT {MAX_RESULT_ROWS}"
    limit_match = re.search(r"LIMIT\s+(\d+)", sql_upper)
    if limit_match and int(limit_match.group(1)) > MAX_RESULT_ROWS:
        return re.sub(r"LIMIT\s+\d+", f"LIMIT {MAX_RESULT_ROWS}", sql, flags=re.IGNORECASE)
    return sql


def _clean_sql(raw: str) -> str:
    """Strip markdown code fences from LLM-generated SQL."""
    cleaned = re.sub(r"```sql\n?", "", raw)
    cleaned = re.sub(r"```\n?", "", cleaned)
    return cleaned.strip()


def _friendly_error(error: str) -> str:
    """Convert a technical error into a user-friendly message."""
    err = (error or "").lower()
    if any(t in err for t in ["api key", "authentication", "unauthorized", "401"]):
        return "API key issue — please check the provider and key in the sidebar."
    if "rate limit" in err or "quota" in err:
        return "Rate-limited by the AI provider. Please wait a moment and try again."
    if "no such table" in err:
        return "The database doesn't contain the needed table."
    if "no such column" in err or "unknown column" in err:
        return "The database doesn't contain the needed column."
    if "syntax" in err or "parse" in err:
        return "Could not generate a valid SQL query. Please try rephrasing the question."
    return "Could not generate a valid query for that question."


def run_sql_pipeline(
    question: str,
    provider: str = "openai",
    model_name: str = "",
    api_key: str | None = None,
    max_retries: int = 1,
) -> dict[str, Any]:
    """
    End-to-end NL2SQL: schema → generate → validate → execute → summarize.

    Returns dict with: answer, sql, result, error.
    """
    provider_key = (provider or "openai").lower()
    resolved_model = model_name or get_default_model(provider_key)

    # 1) Check tables exist
    tables = list_tables()
    if not tables:
        return {"answer": "No data uploaded yet. Upload a CSV or Excel file first.", "error": "no_tables"}

    # 2) Get schema
    try:
        db = get_sql_database()
        schema = db.get_table_info()
    except Exception as exc:
        return {"answer": _friendly_error(str(exc)), "error": str(exc)}

    # 3) Generate → Validate → Execute (with retry loop)
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            # Generate SQL
            retry_context = ""
            if last_error and attempt > 0:
                retry_context = (
                    f"Previous attempt failed with: {last_error}\n"
                    "Generate a different SQL query that avoids this error."
                )

            prompt = SQL_GENERATION_PROMPT.format(
                schema=schema,
                question=question,
                retry_context=retry_context,
            )

            llm = build_chat_model(provider_key, resolved_model, api_key)
            response = llm.invoke([HumanMessage(content=prompt)])
            generated_sql = _clean_sql(response.content)

            # Validate
            is_safe, error_msg = is_sql_safe(generated_sql)
            if not is_safe:
                last_error = error_msg
                continue

            validated_sql = enforce_limit(generated_sql)

            # Execute
            result = run_query(validated_sql)

            # Summarize
            result_str = str(result[:20]) if len(result) > 20 else str(result)
            if len(result) > 20:
                result_str += f"\n... ({len(result)} total rows)"

            answer_prompt = SQL_ANSWER_PROMPT.format(
                question=question,
                sql_query=validated_sql,
                sql_result=result_str,
            )
            answer_response = llm.invoke([HumanMessage(content=answer_prompt)])

            return {
                "answer": answer_response.content.strip(),
                "sql": validated_sql,
                "result": result,
                "error": None,
            }

        except Exception as exc:
            last_error = str(exc)
            continue

    return {"answer": _friendly_error(last_error or ""), "sql": None, "result": None, "error": last_error}


# ========================
# Layer 8 — Human-in-the-Loop
# ========================

REVIEW_KEYWORDS = ["JOIN", "SUBQUERY", "UNION", "HAVING", "EXISTS", "CASE"]


def classify_sql_sensitivity(sql: str) -> tuple[str, str]:
    """Classify SQL as 'safe', 'review', or 'blocked'.

    Returns (level, reason).
    """
    sql_upper = sql.strip().upper()

    # Blocked: any DML
    for kw in FORBIDDEN_KEYWORDS:
        if kw in sql_upper:
            return "blocked", f"Contains {kw} — destructive operations are not allowed."

    if not sql_upper.startswith("SELECT"):
        return "blocked", "Only SELECT queries are permitted."

    # Review: complex queries
    for kw in REVIEW_KEYWORDS:
        if kw in sql_upper:
            return "review", f"Complex query contains {kw} — please review before executing."

    # Count subqueries
    if sql_upper.count("SELECT") > 1:
        return "review", "Contains nested subquery — please review before executing."

    return "safe", "Simple SELECT query."
