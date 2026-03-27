"""
Multi-agent architecture with supervisor pattern.

Agents: Planner → SQL Agent / RAG Agent / Web Agent → Synthesizer
"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage

from config import build_chat_model
from prompts import PLANNING_PROMPT, SYNTHESIS_PROMPT


def plan_action(state: dict) -> dict:
    """Supervisor: decide which sub-agents to invoke."""
    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))

    prompt = PLANNING_PROMPT.format(
        question=state["question"],
        has_database="Yes — tables: " + ", ".join(state.get("tables", [])) if state.get("tables") else "No",
        has_documents="Yes" if state.get("has_documents") else "No",
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # Parse JSON from response
    try:
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
        else:
            plan = {"agents": ["direct"], "reasoning": "Could not parse plan."}
    except json.JSONDecodeError:
        plan = {"agents": ["direct"], "reasoning": content}

    return {
        "plan": plan,
        "current_step": "planned",
    }


def run_sql_sub_agent(state: dict) -> dict:
    """SQL specialist: generates and executes SQL queries."""
    from sql_agent import run_sql_pipeline

    result = run_sql_pipeline(
        question=state["question"],
        provider=state["provider"],
        model_name=state["model_name"],
        api_key=state.get("api_key"),
    )
    return {
        "sql_agent_output": result.get("answer", "SQL agent returned no result."),
        "current_step": "sql_done",
    }


def run_rag_sub_agent(state: dict) -> dict:
    """Document specialist: retrieves and reasons over documents."""
    from langchain_core.messages import HumanMessage

    from config import build_chat_model
    from prompts import RAG_ANSWER_PROMPT
    from rag import search_documents

    chunks = search_documents(state["question"], k=4)
    if not chunks:
        return {
            "rag_agent_output": "No relevant information found in uploaded documents.",
            "current_step": "rag_done",
        }

    context = "\n\n---\n\n".join(chunks)
    prompt = RAG_ANSWER_PROMPT.format(
        context=context,
        question=state["question"],
    )

    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))
    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "rag_agent_output": response.content.strip(),
        "current_step": "rag_done",
    }


def run_web_sub_agent(state: dict) -> dict:
    """Web specialist: searches for current information."""
    from tools import web_search

    result = web_search.invoke({"query": state["question"]})
    return {
        "web_agent_output": result,
        "current_step": "web_done",
    }


def synthesize_results(state: dict) -> dict:
    """Combine outputs from all sub-agents into a unified answer."""
    plan = state.get("plan", {})
    agents_used = plan.get("agents", ["direct"])

    # Collect outputs
    outputs = []
    if "sql" in agents_used and state.get("sql_agent_output"):
        outputs.append(f"**Database Analysis:**\n{state['sql_agent_output']}")
    if "rag" in agents_used and state.get("rag_agent_output"):
        outputs.append(f"**Document Search:**\n{state['rag_agent_output']}")
    if "web" in agents_used and state.get("web_agent_output"):
        outputs.append(f"**Web Search:**\n{state['web_agent_output']}")

    if not outputs:
        return {"current_step": "synthesized"}

    if len(outputs) == 1:
        return {
            "final_answer": outputs[0],
            "messages": [AIMessage(content=outputs[0])],
            "current_step": "synthesized",
        }

    # Multiple agents — use LLM to synthesize
    agent_outputs_text = "\n\n".join(outputs)
    prompt = SYNTHESIS_PROMPT.format(
        question=state["question"],
        agent_outputs=agent_outputs_text,
    )

    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "current_step": "synthesized",
    }


def route_plan(state: dict) -> list[str]:
    """Route based on the planner's output — returns list of next nodes."""
    plan = state.get("plan", {})
    agents = plan.get("agents", ["direct"])

    nodes = []
    if "sql" in agents:
        nodes.append("sql_agent")
    if "rag" in agents:
        nodes.append("rag_agent")
    if "web" in agents:
        nodes.append("web_agent")

    if not nodes:
        return ["synthesize"]
    return nodes
