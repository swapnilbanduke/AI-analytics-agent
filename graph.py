"""
LangGraph agent graph — the core execution engine.

Layer 1: ReAct agent with tool calling (calculator + web search).
Layer 2: Classification node with conditional routing.
Layer 3: SQL tool integration.
Layer 4: Document search tool integration.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config import build_chat_model
from prompts import CLASSIFICATION_PROMPT, SYSTEM_PROMPT
from state import AgentState
from tools import get_all_tools


# ========================
# Node functions
# ========================

def classify_question(state: AgentState) -> dict:
    """Use LLM to classify the question into a route category."""
    question = state["question"]
    has_database = bool(state.get("tables"))
    has_documents = bool(state.get("has_documents"))

    prompt = CLASSIFICATION_PROMPT.format(
        question=question,
        has_database="Yes — tables: " + ", ".join(state.get("tables", [])) if has_database else "No",
        has_documents="Yes" if has_documents else "No",
    )

    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))
    response = llm.invoke([HumanMessage(content=prompt)])
    route = response.content.strip().lower().strip('"').strip("'")

    valid_routes = {"calculation", "web_search", "sql", "document", "direct"}
    if route not in valid_routes:
        route = "direct"

    return {"route": route, "current_step": "classified"}


ROUTE_INSTRUCTIONS = {
    "calculation": "IMPORTANT: You MUST call the `calculator` tool right now to answer this. Do NOT calculate in your head.",
    "web_search": "IMPORTANT: You MUST call the `web_search` tool right now to answer this. Do NOT answer from your own knowledge — use the tool.",
    "sql": "IMPORTANT: You MUST call the `sql_query` tool right now to answer this from the uploaded database.",
    "document": "IMPORTANT: You MUST call the `document_search` tool right now to answer this from the uploaded documents.",
}


def agent_node(state: AgentState) -> dict:
    """Call the LLM with the current messages and bound tools."""
    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))

    has_database = bool(state.get("tables"))
    has_documents = bool(state.get("has_documents"))
    tools = get_all_tools(has_database=has_database, has_documents=has_documents)

    llm_with_tools = llm.bind_tools(tools)

    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    # Inject route instruction so the LLM knows which tool to use
    route = state.get("route")
    if route and route in ROUTE_INSTRUCTIONS:
        messages = list(messages) + [
            HumanMessage(content=ROUTE_INSTRUCTIONS[route])
        ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def handle_error(state: AgentState) -> dict:
    """Convert errors into a friendly final answer."""
    error = state.get("error", "An unexpected error occurred.")
    return {
        "final_answer": f"I encountered an issue: {error}",
        "messages": [AIMessage(content=f"I encountered an issue: {error}")],
    }


# ========================
# Routing functions
# ========================

def should_continue(state: AgentState) -> str:
    """After agent_node: continue to tools if there are tool calls, else end."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def route_after_classify(state: AgentState) -> str:
    """Route based on classification result — all routes go to agent node
    which has all tools bound dynamically."""
    return "agent"


# ========================
# Graph builder
# ========================

def build_agent_graph(has_database: bool = False, has_documents: bool = False):
    """
    Build the main agent graph.

    Flow: [classify] → [agent] ↔ [tools] → END

    The agent node binds tools dynamically based on available data sources.
    """
    tools = get_all_tools(has_database=has_database, has_documents=has_documents)
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("classify", classify_question)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("handle_error", handle_error)

    # Entry point
    workflow.set_entry_point("classify")

    # Classify always routes to agent (agent has all tools)
    workflow.add_edge("classify", "agent")

    # Agent ↔ Tools loop
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    # Error handling
    workflow.add_edge("handle_error", END)

    return workflow.compile()


def process_question(
    question: str,
    provider: str = "openai",
    model_name: str = "",
    api_key: str | None = None,
    tables: list[str] | None = None,
    has_documents: bool = False,
) -> dict:
    """Process a user question through the agent graph and return the result."""
    from config import get_default_model

    provider_key = (provider or "openai").lower()
    resolved_model = model_name or get_default_model(provider_key)
    tables = tables or []

    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
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
    }

    graph = build_agent_graph(
        has_database=bool(tables),
        has_documents=has_documents,
    )
    final_state = graph.invoke(initial_state)

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
    }
