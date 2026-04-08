"""
MCP (Model Context Protocol) server exposing the agent's tools.

Run standalone: python mcp_server.py
Test with: npx @modelcontextprotocol/inspector python mcp_server.py
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("AI Data Analyst")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Examples: "15 * 0.15", "(100 + 200) / 3", "2 ** 10"
    """
    import ast
    import operator

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"Unsupported: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def query_database(question: str, provider: str = "openai", model_name: str = "", api_key: str = "") -> str:
    """Query the uploaded database using natural language.

    Use for statistics, counts, averages, trends, rankings, or any
    question answerable from the loaded data tables.

    Args:
        question: Natural language question about the data.
        provider: LLM provider ("openai" or "anthropic").
        model_name: Model ID (e.g., "gpt-4.1-mini"). Empty = default.
        api_key: API key for the provider. Empty = use env var.
    """
    from sql_agent import run_sql_pipeline

    result = run_sql_pipeline(
        question=question,
        provider=provider or "openai",
        model_name=model_name,
        api_key=api_key or None,
    )
    answer = result.get("answer", "No answer.")
    sql = result.get("sql")
    if sql:
        return f"SQL: {sql}\n\n{answer}"
    return answer


@mcp.tool()
def search_documents(query: str, num_results: int = 4) -> str:
    """Search uploaded documents for relevant information.

    Use for questions about reports, policies, PDFs, or other
    uploaded documents.

    Args:
        query: Search query.
        num_results: Number of results to return (default 4).
    """
    from rag import search_documents as rag_search

    results = rag_search(query, k=num_results)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(results)


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for current information.

    Use for recent events, live data, current prices, news.

    Args:
        query: Search query.
    """
    from tools import web_search as ws
    return ws.invoke({"query": query})


@mcp.tool()
def list_data_tables() -> str:
    """List all tables available in the database."""
    from database import list_tables

    tables = list_tables()
    if not tables:
        return "No tables loaded. Upload a CSV or Excel file first."
    return "Available tables: " + ", ".join(tables)


@mcp.resource("schema://database")
def get_database_schema() -> str:
    """Return the current database schema (all tables and columns)."""
    from database import get_sql_database, list_tables

    tables = list_tables()
    if not tables:
        return "No tables loaded."
    db = get_sql_database()
    return db.get_table_info()


if __name__ == "__main__":
    mcp.run(transport="stdio")
