"""
LangChain tool definitions for the AI Data Analyst agent.

Layer 1: calculator + web_search
Layer 3: sql_query
Layer 4: document_search
"""

import ast
import operator
import os

from langchain_core.tools import tool


# ---------- Calculator ----------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    """Recursively evaluate an AST node using only arithmetic operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use for any arithmetic calculation.

    Examples: "15 * 0.15", "2847 * 0.15", "(100 + 200) / 3", "2 ** 10"
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


# ---------- Web Search ----------

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use for questions about
    recent events, live data, current prices, news, or anything that
    requires up-to-date information."""
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return "Web search is unavailable: TAVILY_API_KEY not set."

        search = TavilySearchResults(
            max_results=3,
            api_key=api_key,
        )
        results = search.invoke(query)

        if not results:
            return f"No results found for: {query}"

        formatted = []
        for r in results:
            title = r.get("url", "")
            content = r.get("content", "")
            formatted.append(f"Source: {title}\n{content}")

        return "\n\n---\n\n".join(formatted)
    except Exception as exc:
        return f"Web search error: {exc}"


# ---------- SQL Query (Layer 3) ----------

@tool
def sql_query(question: str) -> str:
    """Query the uploaded database using natural language. Use this for any
    question about uploaded data: statistics, counts, averages, trends,
    top/bottom rankings, filtering, grouping, or comparing values."""
    try:
        from sql_agent import run_sql_pipeline
        from config import resolve_api_key

        import streamlit as st
        provider = st.session_state.get("provider", "openai")
        model_name = st.session_state.get("model_name", "")

        # Multi-layer API key resolution to handle all Streamlit timing scenarios
        api_key_val = st.session_state.get("_resolved_api_key", "")
        if not api_key_val:
            api_key_val = st.session_state.get("api_key", "")
        if not api_key_val:
            api_key_val = resolve_api_key(provider, "")
        # Also ensure the env var is set for any downstream code
        if api_key_val and provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key_val
        elif api_key_val and provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = api_key_val

        result = run_sql_pipeline(
            question=question,
            provider=provider,
            model_name=model_name,
            api_key=api_key_val,
        )

        answer = result.get("answer", "No answer generated.")
        sql = result.get("sql")
        if sql:
            return f"**SQL:** `{sql}`\n\n{answer}"
        return answer
    except Exception as exc:
        return f"SQL query error: {exc}"


# ---------- Document Search (Layer 4) ----------

@tool
def document_search(query: str) -> str:
    """Search uploaded documents for relevant information. Use this for
    questions about reports, policies, PDFs, or any uploaded documents."""
    try:
        from rag import search_documents

        results = search_documents(query, k=4)
        if not results:
            return "No relevant information found in uploaded documents."
        return "\n\n---\n\n".join(results)
    except ImportError:
        return "Document search is not available yet."
    except Exception as exc:
        return f"Document search error: {exc}"


# ---------- Tool registry ----------

def get_base_tools() -> list:
    """Return the base tools (calculator + web search)."""
    return [calculator, web_search]


def get_all_tools(has_database: bool = False, has_documents: bool = False) -> list:
    """Return all available tools based on what's loaded."""
    tools = [calculator, web_search]
    if has_database:
        tools.append(sql_query)
    if has_documents:
        tools.append(document_search)
    return tools
