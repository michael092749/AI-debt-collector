"""Claude client — SPEC §3, step 7.

The same `LLMClient` the scripted client implements, backed by
``claude-sonnet-5``. Nothing above this file changes when you swap them: the
agent loop, the tools, and the guardrails are identical either way, which is
what makes the offline suite meaningful.

Three details of this model shape the mapping, and each would be a runtime
error or a silent behaviour change if got wrong:

1. **Mid-conversation `system` messages are not supported here.** The turn loop
   uses one to name a guardrail violation back to the model on a blocked turn.
   That arrives as a user turn instead, tagged so it reads as an operator note
   rather than as something the consumer said.
2. **Sampling parameters are rejected.** No `temperature`, no `top_p`. Register
   is set in the prompt, which is where this system keeps it anyway.
3. **A tool result must follow the assistant turn that requested it.** The loop
   records only what was *spoken*, so the assistant tool-use turn is rebuilt
   here from the tool call carried on each tool message.
"""

from __future__ import annotations

import json
import os
from hashlib import sha256
from typing import Any, cast

from collector.llm.base import LLMResponse, Message, ToolCall
from collector.tools import TOOL_SCHEMAS

MODEL = "claude-sonnet-5"

# A spoken turn is one or two sentences. The cap is a backstop against a
# runaway generation, not a style control.
MAX_TOKENS = 1024

# Adaptive thinking at low effort. Thinking stays *on* deliberately: with it
# disabled this model reaches for tools less readily, and an agent that stops
# calling the engine is the one failure this architecture cannot tolerate.
EFFORT = "low"

JsonDict = dict[str, Any]


def _load_env() -> None:
    """Best-effort .env load. Absent library or file is not an error — the key
    may just as well come from the environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


class AnthropicClient:
    """`LLMClient` backed by Claude. Constructed lazily so importing this
    module never requires a key — only calling it does."""

    def __init__(
        self,
        *,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        effort: str = EFFORT,
        api_key: str | None = None,
    ) -> None:
        import anthropic

        _load_env()
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Run the scripted client instead "
                "(uv run collector-text) if you don't have one."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
        self._tools: list[Any] = [
            {
                "name": schema.name,
                "description": schema.description,
                "input_schema": schema.input_schema,
            }
            for schema in TOOL_SCHEMAS
        ]

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        system, conversation = _to_anthropic(messages)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            # Cast, not coercion: the turn shapes are built above and the SDK's
            # TypedDicts cannot be satisfied by a dict assembled at runtime.
            messages=cast(Any, conversation),
            tools=self._tools,
            # Stated rather than left to the default, so the intent survives a
            # future model swap: thinking on, held shallow for call latency.
            thinking={"type": "adaptive"},
            output_config=cast(Any, {"effort": self._effort}),
        )

        if response.stop_reason == "refusal":
            # The model declined. Say nothing rather than something unvetted;
            # the guardrails would hold anyway, but a blank turn is honest and
            # the call is recoverable from the consumer's next utterance.
            return LLMResponse(text="")

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=dict(block.input) if isinstance(block.input, dict) else {},
                        call_id=block.id,
                    )
                )
        return LLMResponse(text=" ".join(p.strip() for p in text_parts).strip(),
                           tool_calls=tuple(calls))


def _to_anthropic(messages: tuple[Message, ...]) -> tuple[str, list[JsonDict]]:
    """Map the loop's transcript onto the Messages API.

    Kept a free function so it can be tested without a key or a network.
    """
    system = ""
    conversation: list[JsonDict] = []

    for index, message in enumerate(messages):
        if message.role == "system":
            if index == 0:
                system = message.content
                continue
            # Mid-conversation system messages are unsupported on this model,
            # so a guardrail note rides in on a user turn. Tagged, because the
            # model must not mistake it for something the consumer said.
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<compliance_note>{message.content}</compliance_note>",
                        }
                    ],
                }
            )
        elif message.role == "consumer":
            conversation.append({"role": "user", "content": message.content})
        elif message.role == "agent":
            conversation.append({"role": "assistant", "content": message.content})
        elif message.role == "tool":
            conversation.extend(_tool_exchange(message))

    return system, conversation


def _tool_exchange(message: Message) -> list[JsonDict]:
    """Rebuild the assistant turn that asked for this result, then the result.

    The loop never records the request — only what reached the consumer's ear —
    but the API requires the pair, and a `tool_result` whose `tool_use_id` has
    no matching `tool_use` is rejected outright.
    """
    call = message.tool_call
    if call is None:
        return [{"role": "user", "content": message.content}]

    # A real response always carries the id. The fallback is for a transcript
    # replayed from a client that had none, and is content-derived rather than
    # hash()-derived so the same transcript maps the same way on every run.
    digest = sha256(f"{call.name}\x00{message.content}".encode()).hexdigest()[:16]
    call_id = call.call_id or f"toolu_{digest}"
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": call.name,
                    "input": call.arguments,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": message.content,
                }
            ],
        },
    ]


def tool_schemas_json() -> str:
    """The tool surface as the model sees it — handy for eyeballing a diff."""
    return json.dumps(
        [
            {"name": s.name, "description": s.description, "input_schema": s.input_schema}
            for s in TOOL_SCHEMAS
        ],
        indent=2,
    )
