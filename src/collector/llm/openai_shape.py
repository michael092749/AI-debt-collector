"""The OpenAI chat-completions wire shape, shared by every client that speaks it.

Two backends reach the same model surface through this format: OpenRouter
(`openrouter_client.py`) and LiveKit Inference (`livekit_client.py`). The
mapping lives here rather than in either of them because a second copy of it
is a compliance hazard, not just duplication — if the transcript a model sees
drifts between routes, the guardrail behaviour certified on one route is not
the behaviour running on the other.

What differs between the two clients is authentication, the model id, and the
provider-specific request extras. Everything below is identical for both.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, cast

from collector.llm.base import LLMResponse, Message, ToolCall
from collector.tools import TOOL_SCHEMAS

JsonDict = dict[str, Any]


def load_env() -> None:
    """Best-effort .env load. Absent library or file is not an error — the key
    may just as well come from the environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def tool_definitions() -> list[JsonDict]:
    """The tool whitelist in OpenAI function-calling shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.input_schema,
            },
        }
        for schema in TOOL_SCHEMAS
    ]


def to_openai_messages(messages: tuple[Message, ...]) -> list[JsonDict]:
    """Map the loop's transcript onto OpenAI-style chat messages.

    Kept a free function so it can be tested without a key or a network.
    Unlike the Anthropic mapping, a mid-conversation `system` role needs no
    workaround here — the chat completions format accepts it anywhere — but
    it is still tagged the same way the Anthropic client tags it, so the
    model reads it as an operator note rather than something the consumer
    said.
    """
    conversation: list[JsonDict] = []

    for index, message in enumerate(messages):
        if message.role == "system":
            if index == 0:
                conversation.append({"role": "system", "content": message.content})
                continue
            note = f"<compliance_note>{message.content}</compliance_note>"
            conversation.append({"role": "system", "content": note})
        elif message.role == "consumer":
            conversation.append({"role": "user", "content": message.content})
        elif message.role == "agent":
            conversation.append({"role": "assistant", "content": message.content})
        elif message.role == "tool":
            conversation.extend(_tool_exchange(message))

    if all(m.role == "system" for m in messages):
        # open_call's first respond() (ring 1), and every regeneration
        # retry off of it, has nothing but system-role turns — a note is
        # appended on each retry (above), never a real consumer/agent/tool
        # turn. The chat completions format tolerates that shape (unlike
        # Anthropic's Messages API, which rejects an empty `messages` list
        # outright), but an all-system context is a degenerate one for a
        # model trained to produce dialogue; mirrors `_to_anthropic`'s
        # `<call_started>` nudge so both clients kick off the same way.
        conversation.append(
            {
                "role": "user",
                "content": (
                    "<call_started>The call has just connected. Greet the "
                    "consumer and open the conversation.</call_started>"
                ),
            }
        )

    return conversation


def to_llm_response(message: Any) -> LLMResponse:
    """Read one assistant message back into the loop's own response type."""
    if message.refusal:
        # The model declined. Say nothing rather than something unvetted;
        # the guardrails would hold anyway, but a blank turn is honest and
        # the call is recoverable from the consumer's next utterance.
        return LLMResponse(text="")

    calls = []
    for tc in message.tool_calls or []:
        # `.function` isn't on the custom-tool half of the SDK's tool-call
        # union, but every tool call here originates from `tool_definitions`,
        # which only ever declares `type: "function"` tools.
        func = cast(Any, tc).function
        calls.append(
            ToolCall(
                name=func.name,
                arguments=json.loads(func.arguments) if func.arguments else {},
                call_id=tc.id,
            )
        )
    return LLMResponse(text=(message.content or "").strip(), tool_calls=tuple(calls))


def _tool_exchange(message: Message) -> list[JsonDict]:
    """Rebuild the assistant turn that asked for this result, then the result.

    The loop never records the request — only what reached the consumer's ear —
    but the API requires the pair: a `tool` message's `tool_call_id` must match
    an `id` on a preceding assistant `tool_calls` entry.
    """
    call = message.tool_call
    if call is None:
        return [{"role": "user", "content": message.content}]

    # A real response always carries the id. The fallback is for a transcript
    # replayed from a client that had none, and is content-derived rather than
    # hash()-derived so the same transcript maps the same way on every run.
    digest = sha256(f"{call.name}\x00{message.content}".encode()).hexdigest()[:16]
    call_id = call.call_id or f"call_{digest}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": message.content},
    ]
