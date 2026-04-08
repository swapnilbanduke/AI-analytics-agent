"""
Typed state definitions for the LangGraph agent.
"""

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Core state passed through every node in the agent graph."""

    # Chat messages (LangChain BaseMessage objects)
    messages: Annotated[list, add_messages]

    # Current user question
    question: str

    # LLM configuration
    provider: str
    model_name: str
    api_key: Optional[str]

    # Routing
    route: Optional[str]                 # "web_search", "sql", "document", "direct"
    current_step: Optional[str]

    # Tool outputs
    tool_outputs: list[dict[str, Any]]

    # Final answer
    final_answer: Optional[str]

    # Error handling
    error: Optional[str]

    # SQL context (Layer 3)
    tables: list[str]
    has_database: bool

    # Document context (Layer 4)
    has_documents: bool

    # Web retrieval validation pipeline
    web_sources: list[dict[str, Any]]
    extracted_claims: list[dict[str, Any]]
    consensus_summary: Optional[str]
    grounded_answer: Optional[str]
    confidence_score: Optional[float]
    confidence_label: Optional[str]
    search_attempts: int
    refined_query: Optional[str]

    # Multi-agent (Layer 6)
    plan: Optional[dict]
    sql_agent_output: Optional[str]
    rag_agent_output: Optional[str]
    web_agent_output: Optional[str]

    # Memory + self-correction (Layer 7)
    chat_history: list[dict[str, str]]
    quality_score: Optional[float]
    reflection: Optional[str]
    retry_count: int

    # Human-in-the-loop (Layer 8)
    requires_approval: bool
    human_approved: Optional[bool]
    pending_sql: Optional[str]
