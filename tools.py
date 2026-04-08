"""
LangChain tool definitions for the AI Data Analyst agent.

Tools: web_search, sql_query, document_search
Utilities: fetch_web_results_structured, score_source_credibility
"""

import os
from urllib.parse import urlparse

from langchain_core.tools import tool


# ========================
# Source credibility scoring
# ========================

CREDIBILITY_TIERS = {
    "high": {
        "suffixes": [".gov", ".edu"],
        "exact": [
            "wikipedia.org", "reuters.com", "apnews.com",
            "bbc.com", "bbc.co.uk", "nature.com",
            "sciencedirect.com", "scholar.google.com",
            "who.int", "cdc.gov", "nih.gov",
        ],
        "score": 1.0,
    },
    "medium": {
        "suffixes": [".org"],
        "exact": [
            "nytimes.com", "washingtonpost.com", "theguardian.com",
            "bloomberg.com", "cnbc.com", "economist.com", "forbes.com",
            "techcrunch.com", "arstechnica.com", "wired.com",
        ],
        "score": 0.7,
    },
    "low": {
        "suffixes": [],
        "exact": [],
        "score": 0.4,
    },
}


def score_source_credibility(domain: str) -> tuple[str, float]:
    """Return (tier_name, score) for a domain. Pure Python, no LLM."""
    domain = domain.lower().strip()
    for tier_name in ("high", "medium"):
        tier = CREDIBILITY_TIERS[tier_name]
        for suffix in tier["suffixes"]:
            if domain.endswith(suffix):
                return tier_name, tier["score"]
        for exact in tier["exact"]:
            if domain == exact or domain.endswith("." + exact):
                return tier_name, tier["score"]
    return "low", CREDIBILITY_TIERS["low"]["score"]


def fetch_web_results_structured(query: str, max_results: int = 5) -> list[dict]:
    """Fetch web results with structured metadata for the validation pipeline.

    Returns list of dicts: [{url, domain, title, content}]
    Not a LangChain tool — called directly by graph nodes.
    """
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return []

        search = TavilySearchResults(max_results=max_results, api_key=api_key)
        results = search.invoke(query)

        if not results:
            return []

        structured = []
        for r in results:
            url = r.get("url", "")
            parsed = urlparse(url)
            domain = parsed.netloc.lower().removeprefix("www.")
            structured.append({
                "url": url,
                "domain": domain,
                "title": url,
                "content": r.get("content", ""),
            })
        return structured
    except Exception:
        return []


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
            max_results=5,
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
    """Return the base tools (web search)."""
    return [web_search]


def get_all_tools(has_database: bool = False, has_documents: bool = False) -> list:
    """Return all available tools based on what's loaded."""
    tools = [web_search]
    if has_database:
        tools.append(sql_query)
    if has_documents:
        tools.append(document_search)
    return tools
