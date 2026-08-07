"""Claude client.

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
import logging
import os
import time
from collections.abc import Iterator
from decimal import Decimal
from hashlib import sha256
from typing import Any, cast

from collector.llm.base import (
    LLMResponse,
    LLMUsage,
    Message,
    StreamCompleted,
    StreamEvent,
    TextDelta,
    ToolCall,
)
from collector.tools import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"

#: Staged rollback without a redeploy: pin an older model here when a new one
#: regresses, rather than editing ``MODEL`` and shipping.
MODEL_ENV_VAR = "COLLECTOR_MODEL"

# A spoken turn is one or two sentences. The cap is a backstop against a
# runaway generation, not a style control.
MAX_TOKENS = 1024

# Adaptive thinking at low effort. Thinking stays *on* deliberately: with it
# disabled this model reaches for tools less readily, and an agent that stops
# calling the engine is the one failure this architecture cannot tolerate.
EFFORT = "low"

# The SDK default is ten minutes — sane for batch work, fatal on a phone call
# where silence past about a second and a half reads as a dropped line. One
# retry, because the wall clock is timeout x (retries + 1) and two attempts is
# all the budget affords.
TIMEOUT_SECONDS = 6.0
MAX_RETRIES = 1

# Published rates, USD per million tokens, as of 2026-08. A model absent from
# this table is reported with no cost rather than a guessed one.
PRICES: dict[str, tuple[Decimal, Decimal]] = {
    "claude-opus-5": (Decimal("5.00"), Decimal("25.00")),
    "claude-sonnet-5": (Decimal("3.00"), Decimal("15.00")),
    "claude-haiku-4-5": (Decimal("1.00"), Decimal("5.00")),
}

# Cache reads bill at a tenth of the input rate; writes at 1.25x.
_CACHE_READ_RATE = Decimal("0.1")
_CACHE_WRITE_RATE = Decimal("1.25")
_PER_MILLION = Decimal(1_000_000)

JsonDict = dict[str, Any]

# Prompt caching. Two breakpoints, and the ceiling is four.
#
# The reason it is worth the write premium here is the shape of a turn, not
# volume: ``stream_turn`` can make up to five sequential calls for one spoken
# reply, and each one re-sends the whole growing transcript plus the system
# prompt plus six tool schemas. Uncached, round three pays full input latency
# for text rounds one and two already sent. A 5-minute entry breaks even at two
# requests (1.25x to write plus 0.1x to read, against 2x); the shortest useful
# turn makes two.
#
# 1. **The last system block**, which by the API's render order (tools, then
#    system, then messages) covers the tool schemas too. Those are built once
#    in ``__init__`` from ``TOOL_SCHEMAS`` in a fixed order, so the prefix is
#    byte-stable for the life of the process — a re-serialization with a
#    different key order would silently cost every hit.
# 2. **The end of the transcript**, so each round reads the prefix the round
#    before it wrote. Within a turn that is where the compounding is.
#
# Scoped to one call, deliberately and unavoidably: ``system_prompt()``
# interpolates the consumer's name and account reference, so the prefix differs
# per consumer and there is no cross-call sharing to claim. The greeting's
# request is what writes the entry the first turn then reads.
#
# The minimum cacheable prefix on this model is 1024 tokens and the system
# prompt plus tool schemas clear it with room to spare. A prefix under the
# minimum does not error — it silently never caches — so the signal to watch is
# ``cache_read_tokens`` on the ``ModelCalled`` row, which is already recorded
# and already priced by ``estimate_cost``. Zero across a whole call means this
# is buying nothing.
_CACHE_CONTROL: JsonDict = {"type": "ephemeral"}


def resolve_model(explicit: str | None = None) -> str:
    """Explicit argument, then ``$COLLECTOR_MODEL``, then the default."""
    return explicit or os.environ.get(MODEL_ENV_VAR) or MODEL


def estimate_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal | None:
    """USD for one call, or ``None`` when the model is not in ``PRICES``.

    Decimal throughout, like every other figure in this system: a float here
    would be a rounding error in a cost report rather than in a payment
    schedule, but the rule is the rule.
    """
    rates = PRICES.get(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    billed = (
        input_rate * input_tokens
        + output_rate * output_tokens
        + input_rate * _CACHE_READ_RATE * cache_read_tokens
        + input_rate * _CACHE_WRITE_RATE * cache_write_tokens
    )
    return billed / _PER_MILLION


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

    #: 4xx that mean "the request is wrong and will stay wrong". Everything
    #: else carrying a status — 408, 409, 429 and every 5xx including 529 — is
    #: worth the SDK's retry and then, if it still fails, a scripted line.
    FATAL_STATUSES = frozenset({400, 401, 403, 404, 405, 413, 422})

    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int = MAX_TOKENS,
        effort: str = EFFORT,
        api_key: str | None = None,
        timeout: float = TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        import anthropic

        _load_env()
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Run the scripted client instead "
                "(uv run collector-text) if you don't have one."
            )
        # Timeout and retries on the client, not hand-rolled around the call:
        # the SDK already backs off on the retryable statuses (408/409/429/5xx)
        # and reads ``retry-after``, and duplicating that is how the two drift.
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=max_retries)
        # Absorb the transient failures; let misconfiguration through. A 401,
        # 404 or 400 is a bad key or a model name that does not exist, and
        # swallowing one would leave the agent silently mute on every turn
        # while the call reported itself compliant.
        #
        # Named by *status*, not by exception class. Listing the classes was
        # the first attempt and it was wrong: ``OverloadedError`` (529) is a
        # sibling of ``InternalServerError`` rather than a subclass, so the
        # single most likely transient failure in production fell straight
        # through — reintroducing the dropped call this exists to prevent.
        # Statuses are stable; the class hierarchy is not.
        self._connection_error = anthropic.APIConnectionError  # includes APITimeoutError
        self._status_error = anthropic.APIStatusError
        self._absorbed = (self._connection_error, self._status_error)
        self._model = resolve_model(model)
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

    def _is_fatal(self, exc: BaseException) -> bool:
        """Misconfiguration, not a blip — re-raise rather than speak through it."""
        return (
            isinstance(exc, self._status_error)
            and getattr(exc, "status_code", None) in self.FATAL_STATUSES
        )

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        system, conversation = _to_anthropic(messages)
        started = time.monotonic()
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=cast(Any, _cached_system(system)),
                # Cast, not coercion: the turn shapes are built above and the SDK's
                # TypedDicts cannot be satisfied by a dict assembled at runtime.
                messages=cast(Any, _cached_transcript(conversation)),
                tools=self._tools,
                # Stated rather than left to the default, so the intent survives a
                # future model swap: thinking on, held shallow for call latency.
                thinking={"type": "adaptive"},
                output_config=cast(Any, {"effort": self._effort}),
            )
        except self._absorbed as exc:
            if self._is_fatal(exc):
                raise
            # Timeout, overload, rate limit, connection drop — already retried
            # by the SDK and still failing. Killing the turn over one would drop
            # the call; instead the loop gets a response it can speak a scripted
            # line from, and the reason reaches the log via ``LLMResponse.error``.
            detail = f"{type(exc).__name__}: {exc}"
            logger.warning("model call failed after retries: %s", detail)
            return LLMResponse(
                error=detail,
                usage=LLMUsage(model=self._model, latency_ms=_elapsed_ms(started)),
            )

        # A ``refusal`` stop reason yields an empty turn rather than something
        # unvetted: the guardrails would hold anyway, but a blank turn is honest
        # and the call is recoverable from the consumer's next utterance.
        return self._from_message(response, _elapsed_ms(started))

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        """The same turn, emitted as it is written.

        The consumer hears the first sentence while the rest is still being
        generated, which is the only way a turn that makes several engine round
        trips fits inside a phone call's tolerance for silence. The caller
        guards each completed sentence before it reaches TTS — see
        ``NegotiationAgent.stream_turn``.
        """
        system, conversation = _to_anthropic(messages)
        started = time.monotonic()
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=cast(Any, _cached_system(system)),
                messages=cast(Any, _cached_transcript(conversation)),
                tools=self._tools,
                thinking={"type": "adaptive"},
                output_config=cast(Any, {"effort": self._effort}),
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield TextDelta(text)
                final = stream.get_final_message()
        except self._absorbed as exc:
            if self._is_fatal(exc):
                raise
            detail = f"{type(exc).__name__}: {exc}"
            logger.warning("streamed model call failed after retries: %s", detail)
            yield StreamCompleted(
                LLMResponse(
                    error=detail,
                    usage=LLMUsage(model=self._model, latency_ms=_elapsed_ms(started)),
                )
            )
            return

        yield StreamCompleted(self._from_message(final, _elapsed_ms(started)))

    def _from_message(self, response: Any, latency_ms: int) -> LLMResponse:
        """Assemble the non-streaming response shape from a finished message."""
        usage = self._usage(response, latency_ms)
        if response.stop_reason == "refusal":
            return LLMResponse(usage=usage)
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
        return LLMResponse(
            text=" ".join(p.strip() for p in text_parts).strip(),
            tool_calls=tuple(calls),
            usage=usage,
        )

    def _usage(self, response: Any, latency_ms: int) -> LLMUsage:
        """Read the SDK's ``usage`` block rather than discarding it.

        Every field is defaulted: the shape has grown before and a missing
        counter should cost a number in a report, not the call it came from.
        """
        raw = getattr(response, "usage", None)
        input_tokens = int(getattr(raw, "input_tokens", 0) or 0)
        output_tokens = int(getattr(raw, "output_tokens", 0) or 0)
        cache_read = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
        model = getattr(response, "model", None) or self._model
        return LLMUsage(
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=estimate_cost(
                model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
            stop_reason=getattr(response, "stop_reason", None),
        )


def _elapsed_ms(started: float) -> int:
    """Monotonic, so a clock adjustment mid-call cannot produce a negative latency."""
    return int((time.monotonic() - started) * 1000)


def _cached_system(system: str) -> list[JsonDict]:
    """The system prompt as a single cache-marked block.

    A list rather than the bare string ``_to_anthropic`` returns, because
    ``cache_control`` lives on a content block and there is nowhere to put it on
    a string. One block, so the breakpoint is unambiguously the last one and the
    tool schemas rendered ahead of it come along.
    """
    return [{"type": "text", "text": system, "cache_control": dict(_CACHE_CONTROL)}]


def _cached_transcript(conversation: list[JsonDict]) -> list[JsonDict]:
    """Mark the end of the transcript, so the next round in this turn reads the
    prefix this one wrote.

    Copies the entry it touches rather than mutating in place: ``_to_anthropic``
    is a pure mapping the tests read directly, and a caching concern that
    reached back into its output would make it lie about what it returns.
    A string ``content`` is promoted to a one-block list first, since that is
    the only shape ``cache_control`` attaches to.
    """
    if not conversation:
        return conversation
    last = dict(conversation[-1])
    content = last["content"]
    blocks: list[JsonDict] = (
        [{"type": "text", "text": content}]
        if isinstance(content, str)
        else [dict(block) for block in content]
    )
    blocks[-1] = {**blocks[-1], "cache_control": dict(_CACHE_CONTROL)}
    last["content"] = blocks
    return [*conversation[:-1], last]


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

    if not conversation:
        # open_call's first respond() (ring 1) has nothing but the system
        # prompt in `messages`, which is stripped above into `system`. The
        # Messages API rejects an empty `messages` list outright, so a
        # synthetic turn kicks the model into producing the opening line.
        conversation.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<call_started>The call has just connected. Greet the "
                            "consumer and open the conversation.</call_started>"
                        ),
                    }
                ],
            }
        )

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
