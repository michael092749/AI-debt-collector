"""OpenRouter client — an alternate `LLMClient`, added alongside `AnthropicClient`.

Routes the same model (`anthropic/claude-sonnet-5`) through OpenRouter's
OpenAI-compatible `/chat/completions` endpoint rather than Anthropic's native
Messages API. Unlike the Anthropic client, this is not a drop-in: the request
and response shapes differ, so the mapping lives in `openai_shape.py` — shared
with `livekit_client.py`, which speaks the same format — rather than being a
reskin of `_to_anthropic`.

Two details of this model shape the mapping:

1. **`reasoning.effort` was meant to replace `thinking: {"type": "adaptive"}`,
   but empirically does nothing.** OpenRouter's unified `reasoning` parameter is
   passed via `extra_body` (it isn't a native field on the OpenAI SDK's
   `create()`) as the closest equivalent for a model routed through it. Live
   probing against `anthropic/claude-sonnet-5` found `reasoning_tokens: 0` in the
   response usage at every effort level from `"low"` through `"high"` — the
   parameter is not observably changing model behavior on this route. It is
   kept at `"low"` for cost, not because a higher setting was shown to help.
   The opening-disclosure failures this was originally suspected of causing
   turned out to be a regex coverage gap in `guardrails/disclosures.py`
   (`_AI_DISCLOSURE_RE` missing a third-person phrasing), not a reasoning or
   tool-calling reliability issue — the model disclosed correctly on nearly
   every generated candidate regardless of effort level.
2. **Refusal surfaces as `message.refusal`, not a `stop_reason`.** Mapped the
   same way as the Anthropic client: say nothing rather than something unvetted.
"""

from __future__ import annotations

import os
from typing import Any, cast

from collector.llm.base import LLMResponse, Message
from collector.llm.openai_shape import (
    load_env,
    to_llm_response,
    to_openai_messages,
    tool_definitions,
)

BASE_URL = "https://openrouter.ai/api/v1"
# Pinned to a dated revision, not the ``~deepseek/deepseek-v4-flash-latest``
# alias. The alias re-points to whatever ships next, which would change the
# model answering live calls with no deploy and no test run behind it — and
# every budget above this line (MAX_TOOL_ROUNDS, the regeneration strikes) is
# tuned per-model. A route change here is a re-certification, so it has to be
# something a commit can name.
MODEL = "deepseek/deepseek-v4-flash-0731"

# A spoken turn is one or two sentences. The cap is a backstop against a
# runaway generation, not a style control — matches the Anthropic client.
MAX_TOKENS = 1024

# See module docstring point 1: kept low for cost — probing found no
# observable effect on model behavior at any effort level.
EFFORT = "low"


class OpenRouterClient:
    """`LLMClient` backed by OpenRouter's OpenAI-compatible chat completions
    endpoint. Constructed lazily so importing this module never requires a
    key — only calling it does."""

    def __init__(
        self,
        *,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        effort: str = EFFORT,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        load_env()
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Run the scripted client instead "
                "(uv run collector-text) if you don't have one."
            )
        self._client = OpenAI(base_url=BASE_URL, api_key=key)
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
        self._tools: list[Any] = tool_definitions()

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        conversation = to_openai_messages(messages)
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=cast(Any, conversation),
            tools=self._tools,
            extra_body={"reasoning": {"effort": self._effort}},
        )
        return to_llm_response(response.choices[0].message)
