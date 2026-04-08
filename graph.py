"""
LangGraph agent graph — the core execution engine.

Routes:
  - direct:     classify → agent_direct → END
  - sql/doc:    classify → agent ↔ tools → END
  - web_search: classify → web_retrieve → score_credibility → extract_claims
                → grounded_generate → verify_answer → accept/retry/uncertain
"""

import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config import build_chat_model
from prompts import (
    CLAIM_EXTRACTION_PROMPT,
    CLASSIFICATION_PROMPT,
    GROUNDED_GENERATION_PROMPT,
    QUERY_REFINEMENT_PROMPT,
    SYSTEM_PROMPT,
    VERIFICATION_PROMPT,
)
from state import AgentState
from tools import (
    fetch_web_results_structured,
    get_all_tools,
    score_source_credibility,
)


# ========================
# Rate-limit retry helper
# ========================

def _invoke_with_retry(llm, messages, max_retries=3):
    """Invoke LLM with exponential backoff on rate-limit (429) errors."""
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str
            if is_rate_limit and attempt < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            raise


# ========================
# Parse helpers
# ========================

def _parse_claims_response(text: str) -> tuple[list[dict], str]:
    """Parse the LLM's claim extraction response."""
    claims = []
    for match in re.finditer(r"-\s*\[Source\s*(\d+)\]\s*(.+)", text):
        claims.append({
            "source_index": int(match.group(1)),
            "claim_text": match.group(2).strip(),
        })

    consensus = "partial"
    consensus_match = re.search(r"CONSENSUS:\s*(agree|partial|conflict)", text, re.IGNORECASE)
    if consensus_match:
        consensus = consensus_match.group(1).lower()

    return claims, consensus


def _parse_verification_response(text: str) -> tuple[str, float]:
    """Parse the LLM's verification response."""
    label = "medium"
    score = 0.6

    match = re.search(r"CONFIDENCE:\s*(high|medium|low)", text, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        score = {"high": 0.9, "medium": 0.6, "low": 0.3}[label]

    return label, score


# ========================
# Shared nodes
# ========================

def classify_question(state: AgentState) -> dict:
    """Use LLM to classify the question into a route category."""
    question = state["question"]
    has_database = bool(state.get("tables"))
    has_documents = bool(state.get("has_documents"))

    # Include recent conversation context so follow-up questions classify correctly
    context_lines = []
    msgs = state.get("messages", [])
    # Take last few user/assistant turns (skip system message, skip current question)
    recent = [m for m in msgs if isinstance(m, (HumanMessage, AIMessage))]
    for m in recent[-6:-1]:  # last 3 exchanges before current question
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        snippet = m.content[:200]
        context_lines.append(f"{role}: {snippet}")
    context_str = "\n".join(context_lines) if context_lines else "None"

    prompt = CLASSIFICATION_PROMPT.format(
        question=question,
        has_database="Yes — tables: " + ", ".join(state.get("tables", [])) if has_database else "No",
        has_documents="Yes" if has_documents else "No",
    )
    # Append conversation context
    prompt += f"\n\nRecent conversation context:\n{context_str}"

    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))
    response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])
    route = response.content.strip().lower().strip('"').strip("'")

    valid_routes = {"web_search", "sql", "document", "direct"}
    if route not in valid_routes:
        route = "direct"

    return {"route": route, "current_step": "classified"}


def handle_error(state: AgentState) -> dict:
    """Convert errors into a friendly final answer."""
    error = state.get("error", "An unexpected error occurred.")
    return {
        "final_answer": f"I encountered an issue: {error}",
        "messages": [AIMessage(content=f"I encountered an issue: {error}")],
    }


# ========================
# Direct route node
# ========================

def agent_direct(state: AgentState) -> dict:
    """Answer directly without tools. For general knowledge questions."""
    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    response = _invoke_with_retry(llm, messages)
    return {"messages": [response], "current_step": "direct_answered"}


# ========================
# SQL/Document agent nodes (existing ReAct loop)
# ========================

ROUTE_INSTRUCTIONS = {
    "sql": "IMPORTANT: You MUST call the `sql_query` tool right now to answer this from the uploaded database.",
    "document": "IMPORTANT: You MUST call the `document_search` tool right now to answer this from the uploaded documents.",
}


def agent_node(state: AgentState) -> dict:
    """Call the LLM with the current messages and bound tools (sql/document routes)."""
    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))

    has_database = bool(state.get("tables"))
    has_documents = bool(state.get("has_documents"))
    tools = get_all_tools(has_database=has_database, has_documents=has_documents)

    llm_with_tools = llm.bind_tools(tools)

    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    # Inject route instruction only on the first agent call (before any tool use)
    route = state.get("route")
    has_tool_results = any(
        getattr(m, "type", None) == "tool" for m in state["messages"]
    )
    if route and route in ROUTE_INSTRUCTIONS and not has_tool_results:
        messages = list(messages) + [
            HumanMessage(content=ROUTE_INSTRUCTIONS[route])
        ]

    response = _invoke_with_retry(llm_with_tools, messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """After agent_node: continue to tools if there are tool calls, else end."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ========================
# Web search validation pipeline
# ========================

def web_retrieve(state: AgentState) -> dict:
    """Fetch 5 web results using Tavily. No LLM call."""
    query = state.get("refined_query") or state["question"]
    results = fetch_web_results_structured(query, max_results=5)
    attempts = state.get("search_attempts", 0) + 1

    if not results:
        return {
            "web_sources": [],
            "error": "No web results found.",
            "current_step": "web_retrieve_empty",
            "search_attempts": attempts,
        }
    return {
        "web_sources": results,
        "current_step": "web_retrieved",
        "search_attempts": attempts,
    }


def web_score_credibility(state: AgentState) -> dict:
    """Score and rank sources by domain credibility. No LLM call."""
    sources = list(state.get("web_sources", []))
    for source in sources:
        tier, score = score_source_credibility(source["domain"])
        source["credibility_tier"] = tier
        source["credibility_score"] = score
    sources.sort(key=lambda s: s["credibility_score"], reverse=True)
    return {"web_sources": sources, "current_step": "credibility_scored"}


def web_extract_claims(state: AgentState) -> dict:
    """LLM extracts claims from each source + consensus check."""
    sources = state.get("web_sources", [])
    if not sources:
        return {"extracted_claims": [], "consensus_summary": "none", "current_step": "no_claims"}

    numbered = []
    for i, s in enumerate(sources, 1):
        numbered.append(
            f"[Source {i}] ({s.get('credibility_tier', 'unknown')} credibility) {s['url']}\n{s['content']}"
        )

    prompt = CLAIM_EXTRACTION_PROMPT.format(
        question=state["question"],
        numbered_sources="\n\n".join(numbered),
    )

    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))
    response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])

    claims, consensus = _parse_claims_response(response.content)

    return {
        "extracted_claims": claims,
        "consensus_summary": consensus,
        "current_step": "claims_extracted",
    }


def web_grounded_generate(state: AgentState) -> dict:
    """Generate answer ONLY from extracted claims. Anti-hallucination gate."""
    claims = state.get("extracted_claims", [])
    if not claims:
        fallback = "I couldn't find reliable information from web sources to answer this question."
        return {
            "grounded_answer": fallback,
            "messages": [AIMessage(content=fallback)],
            "current_step": "grounded_empty",
        }

    claims_text = "\n".join(
        f"- {c['claim_text']} [Source {c['source_index']}]" for c in claims
    )
    cred_summary = "\n".join(
        f"Source {i + 1}: {s.get('credibility_tier', 'unknown')} ({s['url']})"
        for i, s in enumerate(state.get("web_sources", []))
    )

    prompt = GROUNDED_GENERATION_PROMPT.format(
        question=state["question"],
        claims_text=claims_text,
        credibility_summary=cred_summary,
        consensus=state.get("consensus_summary", "unknown"),
    )

    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))
    response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])

    return {
        "grounded_answer": response.content.strip(),
        "current_step": "grounded_generated",
    }


def web_verify_answer(state: AgentState) -> dict:
    """Verify answer against sources. Assign confidence score."""
    answer = state.get("grounded_answer", "")
    sources = state.get("web_sources", [])

    numbered = []
    for i, s in enumerate(sources, 1):
        numbered.append(f"[Source {i}] {s['url']}\n{s['content']}")

    prompt = VERIFICATION_PROMPT.format(
        question=state["question"],
        answer=answer,
        numbered_sources="\n\n".join(numbered),
    )

    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))
    response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])

    confidence_label, confidence_score = _parse_verification_response(response.content)

    return {
        "confidence_label": confidence_label,
        "confidence_score": confidence_score,
        "current_step": "verified",
    }


def web_output_decision(state: AgentState) -> str:
    """Route after verification: accept, retry, or uncertain."""
    confidence = state.get("confidence_label", "medium")
    attempts = state.get("search_attempts", 1)

    if confidence in ("high", "medium"):
        return "accept"
    # confidence == "low"
    if attempts < 2:
        return "retry"
    return "uncertain"


def web_finalize_answer(state: AgentState) -> dict:
    """Package the grounded answer as the final answer with source citations."""
    answer = state.get("grounded_answer", "")
    sources = state.get("web_sources", [])

    source_lines = [f"- [{s['url']}]({s['url']})" for s in sources[:5]]
    source_footer = "\n".join(source_lines)

    full_answer = f"{answer}\n\n**Sources:**\n{source_footer}"

    return {
        "final_answer": full_answer,
        "messages": [AIMessage(content=full_answer)],
        "current_step": "web_complete",
    }


def web_finalize_uncertain(state: AgentState) -> dict:
    """Package a low-confidence answer with uncertainty caveat."""
    answer = state.get("grounded_answer", "")
    sources = state.get("web_sources", [])

    source_lines = [f"- [{s['url']}]({s['url']})" for s in sources[:5]]
    source_footer = "\n".join(source_lines)

    caveat = (
        "I found some relevant information but could not verify it with high confidence. "
        "The sources available were limited or partially conflicting."
    )

    full_answer = (
        f"**Note: Limited confidence in this answer.**\n\n{answer}\n\n"
        f"---\n*{caveat}*\n\n**Sources:**\n{source_footer}"
    )

    return {
        "final_answer": full_answer,
        "messages": [AIMessage(content=full_answer)],
        "current_step": "web_uncertain",
    }


def web_refine_query(state: AgentState) -> dict:
    """Refine search query for retry. One LLM call."""
    prompt = QUERY_REFINEMENT_PROMPT.format(
        question=state["question"],
        original_query=state["question"],
        problem_description=f"Confidence was low. Consensus: {state.get('consensus_summary', 'unknown')}",
    )

    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))
    response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])

    return {
        "refined_query": response.content.strip(),
        "current_step": "query_refined",
    }


# ========================
# Routing functions
# ========================

def route_after_classify(state: AgentState) -> str:
    """Branch to route-specific subflows after classification."""
    route = state.get("route", "direct")
    if route == "web_search":
        return "web_retrieve"
    if route == "direct":
        return "agent_direct"
    # sql and document both use the existing agent+tools loop
    return "agent"


# ========================
# Graph builder
# ========================

def build_agent_graph(has_database: bool = False, has_documents: bool = False):
    """
    Build the main agent graph with route-specific subflows.

    Routes:
      direct:     classify → agent_direct → END
      sql/doc:    classify → agent ↔ tools → END
      web_search: classify → web pipeline (retrieve → score → extract → generate → verify) → END
    """
    tools = get_all_tools(has_database=has_database, has_documents=has_documents)
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    # --- Shared nodes ---
    workflow.add_node("classify", classify_question)
    workflow.add_node("handle_error", handle_error)

    # --- Direct route ---
    workflow.add_node("agent_direct", agent_direct)

    # --- SQL/Document route (existing ReAct loop) ---
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # --- Web search validation pipeline ---
    workflow.add_node("web_retrieve", web_retrieve)
    workflow.add_node("score_credibility", web_score_credibility)
    workflow.add_node("extract_claims", web_extract_claims)
    workflow.add_node("grounded_generate", web_grounded_generate)
    workflow.add_node("verify_answer", web_verify_answer)
    workflow.add_node("finalize_web_answer", web_finalize_answer)
    workflow.add_node("finalize_uncertain", web_finalize_uncertain)
    workflow.add_node("refine_query", web_refine_query)

    # --- Entry ---
    workflow.set_entry_point("classify")

    # --- After classify: branch by route ---
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "agent_direct": "agent_direct",
            "agent": "agent",
            "web_retrieve": "web_retrieve",
        },
    )

    # --- Direct route → END ---
    workflow.add_edge("agent_direct", END)

    # --- SQL/Document agent loop ---
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    # --- Web pipeline: linear chain ---
    workflow.add_edge("web_retrieve", "score_credibility")
    workflow.add_edge("score_credibility", "extract_claims")
    workflow.add_edge("extract_claims", "grounded_generate")
    workflow.add_edge("grounded_generate", "verify_answer")

    # --- After verification: conditional branch ---
    workflow.add_conditional_edges(
        "verify_answer",
        web_output_decision,
        {
            "accept": "finalize_web_answer",
            "retry": "refine_query",
            "uncertain": "finalize_uncertain",
        },
    )

    # --- Terminal web nodes → END ---
    workflow.add_edge("finalize_web_answer", END)
    workflow.add_edge("finalize_uncertain", END)

    # --- Retry loop ---
    workflow.add_edge("refine_query", "web_retrieve")

    # --- Error handling ---
    workflow.add_edge("handle_error", END)

    return workflow.compile()


# ========================
# Entry point
# ========================

def process_question(
    question: str,
    provider: str = "openai",
    model_name: str = "",
    api_key: str | None = None,
    tables: list[str] | None = None,
    has_documents: bool = False,
    chat_history: list[dict] | None = None,
) -> dict:
    """Process a user question through the agent graph and return the result."""
    from config import get_default_model

    provider_key = (provider or "openai").lower()
    resolved_model = model_name or get_default_model(provider_key)
    tables = tables or []

    # Build message list with conversation history for context
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in (chat_history or []):
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=question))

    initial_state = {
        "messages": messages,
        "question": question,
        "provider": provider_key,
        "model_name": resolved_model,
        "api_key": api_key,
        "route": None,
        "current_step": None,
        "tool_outputs": [],
        "final_answer": None,
        "error": None,
        "tables": tables,
        "has_database": bool(tables),
        "has_documents": has_documents,
        # Web retrieval validation pipeline
        "web_sources": [],
        "extracted_claims": [],
        "consensus_summary": None,
        "grounded_answer": None,
        "confidence_score": None,
        "confidence_label": None,
        "search_attempts": 0,
        "refined_query": None,
    }

    graph = build_agent_graph(
        has_database=bool(tables),
        has_documents=has_documents,
    )
    final_state = graph.invoke(initial_state, {"recursion_limit": 15})

    # Extract the last AI message as the answer
    answer = final_state.get("final_answer")
    if not answer:
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break

    return {
        "answer": answer or "I couldn't generate an answer.",
        "route": final_state.get("route"),
        "error": final_state.get("error"),
        "messages": final_state["messages"],
        "confidence_label": final_state.get("confidence_label"),
        "confidence_score": final_state.get("confidence_score"),
        "sources": [s.get("url") for s in final_state.get("web_sources", [])],
    }
