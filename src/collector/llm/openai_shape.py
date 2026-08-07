"""The OpenAI chat-completions wire shape, shared by every client that speaks it.

Two backends reach the same model surface through this format: OpenRouter
(`openrouter_client.py`) and LiveKit Inference (`livekit_client.py`). The
mapping lives here rather than in either of them because a second copy of it
is a compliance hazard, not just duplication — if the transcript a model sees
drifts between routes, the guardrail behaviour certified on one route is not
the behaviour running on the other.

What differs between the two clients is authentication, the model id, and the
provider-specific request extras. Everything below is identical for both —
including the timeout, the retry budget and the transient-failure handling,
which are shared for the same reason the transcript mapping is: a route that
absorbs a 529 and a route that drops the call on one are not the same system,
however similar their request bodies look.

Pricing is *not* shared. Each gateway bills its own rates, so the price table
lives with the client that knows which gateway it is talking to; usage arrives
from here with ``cost_usd`` unset.
"""

from __future__ import annotations

import json
import logging
import time
from hashlib import sha256
from typing import Any, cast

from collector.llm.base import LLMResponse, LLMUsage, Message, ToolCall
from collector.tools import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

# The OpenAI SDK default is ten minutes — sane for batch work, fatal on a phone
# call where silence past about a second and a half reads as a dropped line.
# One retry, because the wall clock is timeout x (retries + 1) and two attempts
# is all the budget affords.
#
# Inherited from ``anthropic_client`` deliberately, values and all: one route's
# tail behaviour should not differ from another's by accident. Stated plainly
# because the arithmetic is worse than it first reads — 6.0 x 2 is a **12
# second** worst case, far past the tolerance that motivates the number. It
# bounds a stall; it does not deliver 1.5s. Retuning it is a decision for all
# three clients at once, not a thing to change on one route in passing.
TIMEOUT_SECONDS = 6.0
MAX_RETRIES = 1

#: 4xx that mean "the request is wrong and will stay wrong". Everything else
#: carrying a status — 408, 409, 429 and every 5xx — is worth the SDK's retry
#: and then, if it still fails, a scripted line.
#:
#: Named by *status*, not by exception class, for the reason spelled out in
#: ``anthropic_client``: there, ``OverloadedError`` (529) turned out to be a
#: sibling of ``InternalServerError`` rather than a subclass, so a class-based
#: list let the single most likely transient failure fall straight through.
#: Statuses are stable; the class hierarchy is not, and it is not even the
#: same hierarchy on this SDK.
FATAL_STATUSES = frozenset({400, 401, 403, 404, 405, 413, 422})


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


def elapsed_ms(started: float) -> int:
    """Monotonic, so a clock adjustment mid-call cannot produce a negative latency."""
    return int((time.monotonic() - started) * 1000)


def to_llm_usage(
    response: Any, *, latency_ms: int, model: str, stop_reason: str | None
) -> LLMUsage:
    """Read the chat-completions ``usage`` block rather than discarding it.

    Every field is defaulted, the way ``AnthropicClient._usage`` defaults its
    own: the shape has grown before and a missing counter should cost a number
    in a report, not the call it came from. That matters more here than there —
    ``cache_write_tokens`` is a LiveKit gateway extension to the stock OpenAI
    shape and is simply absent on other providers, so it is read by name and
    default rather than by attribute access.

    ``cost_usd`` is left unset. Rates are per-gateway; the client that knows
    which gateway this was fills it in.
    """
    raw = getattr(response, "usage", None)
    details = getattr(raw, "prompt_tokens_details", None)
    return LLMUsage(
        model=getattr(response, "model", None) or model,
        latency_ms=latency_ms,
        input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        cache_read_tokens=int(getattr(details, "cached_tokens", 0) or 0),
        cache_write_tokens=int(getattr(details, "cache_write_tokens", 0) or 0),
        stop_reason=stop_reason,
    )


def to_llm_response(response: Any, *, latency_ms: int, model: str) -> LLMResponse:
    """Read one chat completion back into the loop's own response type.

    Takes the whole response, not just the message: the token counters hang off
    ``response.usage`` and ``finish_reason`` off the *choice*, so a message-only
    signature could not report either.
    """
    choice = response.choices[0]
    message = choice.message
    # Usage first, so it survives every early return below. A refused turn
    # still spent tokens and still spent latency, and a turn that reports
    # neither is invisible to the per-call logging this route exists to feed.
    usage = to_llm_usage(
        response,
        latency_ms=latency_ms,
        model=model,
        stop_reason=getattr(choice, "finish_reason", None),
    )

    if message.refusal:
        # The model declined. Say nothing rather than something unvetted;
        # the guardrails would hold anyway, but a blank turn is honest and
        # the call is recoverable from the consumer's next utterance.
        return LLMResponse(text="", usage=usage)

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
    return LLMResponse(text=(message.content or "").strip(), tool_calls=tuple(calls), usage=usage)


def chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[JsonDict],
    tools: list[Any],
    max_tokens: int,
    **extra: Any,
) -> LLMResponse:
    """One chat-completions round trip, timed, with transient failures absorbed.

    Shared by both OpenAI-shaped routes. A blip — timeout, 429, 5xx, connection
    drop — must not kill the turn: this is a live phone call, and the loop can
    speak a scripted line from an ``LLMResponse`` carrying ``error``. Genuine
    misconfiguration is re-raised instead, because a 400/401/404 is a bad key or
    a model name that does not exist, and swallowing one would leave the agent
    silently mute on every turn while the call reported itself compliant.
    """
    import openai

    started = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            # Cast, not coercion: the turn shapes are built above and the SDK's
            # TypedDicts cannot be satisfied by a dict assembled at runtime.
            messages=cast(Any, messages),
            tools=tools,
            **extra,
        )
    except (openai.APIConnectionError, openai.APIStatusError) as exc:
        # APITimeoutError is a subclass of APIConnectionError, so it is covered.
        if (
            isinstance(exc, openai.APIStatusError)
            and getattr(exc, "status_code", None) in FATAL_STATUSES
        ):
            raise
        # Already retried by the SDK and still failing. Killing the turn over
        # one would drop the call; instead the loop gets a response it can
        # speak a scripted line from, and the reason reaches the log via
        # ``LLMResponse.error``.
        detail = f"{type(exc).__name__}: {exc}"
        logger.warning("model call failed after retries: %s", detail)
        return LLMResponse(
            error=detail,
            usage=LLMUsage(model=model, latency_ms=elapsed_ms(started)),
        )

    return to_llm_response(response, latency_ms=elapsed_ms(started), model=model)


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
