"""Structured logging with reasoning-trace support.

Agent decisions should be *auditable*. We use ``rich`` for pretty console
output and also emit a machine-readable trace list on the agent state so
downstream code (notebooks, reports) can render the full decision chain.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

_LEVEL = os.getenv("AG_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_path=False)],
)

logger = logging.getLogger("agentic_genomics")
console = Console()


@dataclass
class ReasoningStep:
    """One entry in an agent's audit trail."""

    node: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "summary": self.summary,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


def trace(node: str, summary: str, **evidence: Any) -> ReasoningStep:
    """Create a reasoning step and log it at INFO."""
    step = ReasoningStep(node=node, summary=summary, evidence=evidence)
    logger.info("[%s] %s", node, summary)
    return step
