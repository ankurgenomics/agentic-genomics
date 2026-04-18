"""Provider-agnostic LLM factory.

The MVP uses Anthropic's Claude via ``langchain-anthropic``. We wrap model
construction in a single function so swapping providers (OpenAI, local
models via Ollama, etc.) only requires editing this file.

Why a thin wrapper and not heavy abstractions?
- Keeps the agent code readable.
- Avoids forcing a lowest-common-denominator API when Claude-specific
  features (e.g. large context, cache control) are useful.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

DEFAULT_MODEL = os.getenv("AG_DEFAULT_MODEL", "claude-sonnet-4-5")


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for the agent's LLM."""

    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 4096
    provider: str = "anthropic"


def get_llm(config: LLMConfig | None = None) -> BaseChatModel:
    """Return a LangChain-compatible chat model.

    Temperature defaults to 0.0 because clinical-style reasoning should be
    reproducible. Raise temperature explicitly for brainstorming nodes if
    ever needed.
    """
    cfg = config or LLMConfig()

    if cfg.provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if cfg.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the openai extra: pip install -e '.[openai]'"
            ) from exc
        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    raise ValueError(f"Unknown LLM provider: {cfg.provider!r}")
