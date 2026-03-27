"""
Conversation memory and Reflexion self-correction.

- ConversationMemory: stores turns, manages context window, summarizes old messages.
- evaluate_answer_quality: LLM scores its own answer and triggers retry if poor.
"""

import re

from langchain_core.messages import HumanMessage

from config import build_chat_model
from prompts import QUALITY_EVAL_PROMPT


class ConversationMemory:
    """Manages conversation history with context window management."""

    def __init__(self, max_turns: int = 20):
        self.turns: list[dict] = []
        self.max_turns = max_turns

    def add_turn(self, question: str, answer: str, metadata: dict | None = None):
        self.turns.append({
            "question": question,
            "answer": answer,
            "metadata": metadata or {},
        })
        # Trim to max
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_context_messages(self) -> list[dict]:
        """Return recent turns as message dicts for LLM context."""
        messages = []
        for turn in self.turns[-10:]:  # last 10 turns
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        return messages

    def get_summary(self) -> str:
        """Generate a compact summary of the conversation so far."""
        if not self.turns:
            return "No previous conversation."
        lines = []
        for i, turn in enumerate(self.turns[-5:], 1):
            lines.append(f"Q{i}: {turn['question'][:100]}")
            lines.append(f"A{i}: {turn['answer'][:150]}")
        return "\n".join(lines)

    def clear(self):
        self.turns = []


def evaluate_answer_quality(state: dict) -> dict:
    """LLM scores its own answer on quality. Returns updated state with score."""
    answer = state.get("final_answer", "")
    if not answer:
        return {"quality_score": 0.0, "reflection": "No answer to evaluate."}

    prompt = QUALITY_EVAL_PROMPT.format(
        question=state["question"],
        answer=answer,
        tool_outputs=str(state.get("tool_outputs", []))[:500],
    )

    llm = build_chat_model(state["provider"], state["model_name"], state.get("api_key"))
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # Parse score
    score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", content)
    score = float(score_match.group(1)) if score_match else 5.0

    retry_count = state.get("retry_count", 0)

    # If quality is low and we haven't retried too much, trigger a retry
    if score < 6 and retry_count < 2:
        return {
            "quality_score": score,
            "reflection": content,
            "retry_count": retry_count + 1,
            "final_answer": None,  # Clear to trigger retry
            "current_step": "needs_retry",
        }

    return {
        "quality_score": score,
        "reflection": content,
        "current_step": "quality_ok",
    }


def should_retry(state: dict) -> str:
    """Check if the answer needs a retry based on quality evaluation."""
    if state.get("current_step") == "needs_retry":
        return "retry"
    return "done"
