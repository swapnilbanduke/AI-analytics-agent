"""
Multi-provider model catalog and configuration.

Adapted from Text to SQL/model_catalog.py — supports OpenAI + Anthropic
with environment variable fallback for API keys.
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ========================
# LangSmith tracing (Layer 9)
# ========================

LANGSMITH_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if LANGSMITH_TRACING:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "ai-data-analyst"))


def enable_langsmith(api_key: str, project: str = "ai-data-analyst"):
    """Enable LangSmith tracing at runtime (called from sidebar)."""
    global LANGSMITH_TRACING
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    LANGSMITH_TRACING = True


def disable_langsmith():
    """Disable LangSmith tracing at runtime."""
    global LANGSMITH_TRACING
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ.pop("LANGCHAIN_API_KEY", None)
    LANGSMITH_TRACING = False


DEFAULT_PROVIDER = "openai"

PROVIDER_CATALOG = {
    "openai": {
        "label": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "api_key_label": "OpenAI API key",
        "api_key_help": "Used for GPT models such as GPT-4o and GPT-4.1.",
        "models": [
            {"value": "gpt-4.1-nano", "label": "GPT-4.1 nano (cheapest)"},
            {"value": "gpt-4.1-mini", "label": "GPT-4.1 mini"},
            {"value": "gpt-4.1", "label": "GPT-4.1"},
            {"value": "gpt-4o-mini", "label": "GPT-4o mini"},
            {"value": "gpt-4o", "label": "GPT-4o"},
            {"value": "gpt-4o-audio-preview", "label": "GPT-4o audio preview"},
            {"value": "gpt-4.5-preview", "label": "GPT-4.5 preview"},
            {"value": "o4-mini", "label": "o4-mini (reasoning)"},
            {"value": "o3", "label": "o3 (reasoning)"},
            {"value": "o3-mini", "label": "o3-mini (reasoning)"},
            {"value": "o1", "label": "o1 (reasoning)"},
            {"value": "o1-mini", "label": "o1-mini (reasoning)"},
            {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo (legacy, cheap)"},
        ],
        "default_model": "gpt-4.1-nano",
    },
    "anthropic": {
        "label": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "api_key_label": "Anthropic API key",
        "api_key_help": "Used for Claude models such as Sonnet and Opus.",
        "models": [
            {"value": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4"},
            {"value": "claude-opus-4-20250514", "label": "Claude Opus 4"},
            {"value": "claude-3-7-sonnet-20250219", "label": "Claude Sonnet 3.7"},
            {"value": "claude-3-5-haiku-20241022", "label": "Claude Haiku 3.5"},
        ],
        "default_model": "claude-sonnet-4-20250514",
    },
}


def get_provider_ids() -> list[str]:
    return list(PROVIDER_CATALOG.keys())


def get_provider_config(provider: str) -> dict:
    provider_key = (provider or "").lower()
    if provider_key not in PROVIDER_CATALOG:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDER_CATALOG[provider_key]


def get_model_options(provider: str) -> list[dict]:
    return get_provider_config(provider)["models"]


def get_model_values(provider: str) -> list[str]:
    return [m["value"] for m in get_model_options(provider)]


def get_model_label(provider: str, model_value: str) -> str:
    for m in get_model_options(provider):
        if m["value"] == model_value:
            return m["label"]
    return model_value


def get_default_model(provider: str) -> str:
    return get_provider_config(provider)["default_model"]


def resolve_model(provider: str, selected_model: str, custom_model: str = "") -> str:
    custom_value = (custom_model or "").strip()
    if custom_value:
        return custom_value
    if selected_model:
        return selected_model
    return get_default_model(provider)


def resolve_api_key(provider: str, session_value: str = "") -> str:
    config = get_provider_config(provider)
    return (session_value or "").strip() or os.getenv(config["env_var"], "")


def build_chat_model(provider: str, model_name: str, api_key: str | None = None, **kwargs):
    """Create a chat model instance for the chosen provider."""
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    provider_key = (provider or "").lower()
    resolved_model = model_name or get_default_model(provider_key)
    resolved_key = resolve_api_key(provider_key, api_key or "")

    if not resolved_key:
        raise ValueError("Missing API key for selected provider.")

    if provider_key == "openai":
        return ChatOpenAI(
            model=resolved_model,
            temperature=0,
            timeout=60,
            api_key=resolved_key,
            **kwargs,
        )

    if provider_key == "anthropic":
        return ChatAnthropic(
            model_name=resolved_model,
            temperature=0,
            timeout=60,
            max_retries=2,
            api_key=resolved_key,
            **kwargs,
        )

    raise ValueError(f"Unsupported provider: {provider}")
