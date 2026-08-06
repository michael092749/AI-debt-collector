"""Model clients. The only part of the system permitted to be non-deterministic.

Everything reachable from here talks; nothing reachable from here decides. A
client's entire job is to turn a conversation into either a tool call or a
sentence, and both are checked before they have any effect.
"""

from __future__ import annotations

from collector.llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    system_prompt,
)
from collector.llm.mock_client import MockLLMClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "MockLLMClient",
    "ToolCall",
    "system_prompt",
]
