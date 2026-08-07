"""The model boundary — SPEC §3, ``llm/base.py``.

One protocol, two implementations: a scripted client for offline tests and
Claude for real calls. The agent loop is written against this protocol only, so
the entire system runs, and is tested, with no API key.

The system prompt lives here rather than in ``agent.py`` because of what it is
*not* allowed to contain. Compliance rules live in code with stable ids (SPEC
§9); prose asking a model nicely to obey them is not a control. What the prompt
carries is register and procedure — how to sound, and which tool to reach for.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "consumer", "agent", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """The model asking for an action from the whitelist in ``tools.py``."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(frozen=True)
class Message:
    """One entry in the conversation as the model sees it.

    ``tool`` messages carry an engine result: the JSON a tool returned, which
    is the only place the model may take a number from.
    """

    role: Role
    content: str
    tool_call: ToolCall | None = None
    tool_call_id: str = ""


@dataclass(frozen=True)
class LLMUsage:
    """What one model call cost, in the three currencies a voice turn spends.

    Latency is the one that decides whether the architecture works: a turn can
    make several of these and the hang-up budget is under two seconds, so a
    per-call millisecond figure is the only way to know where the budget went.

    ``cost_usd`` is ``None`` for a model with no entry in the price table. An
    unpriced call is honest; a guessed price in a cost report is not.
    """

    model: str
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Decimal | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Either something to say, or something to ask the engine for, or both.

    Tool calls are executed in order and their results appended before the
    model is asked again, so a turn may make several engine round trips before
    it produces the sentence the consumer hears.

    ``error`` carries a transport failure the client already absorbed. The
    response is still well-formed — empty text, no tool calls — so the loop
    degrades to a scripted turn instead of raising, and the reason is logged
    rather than lost.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: LLMUsage | None = None
    error: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMClient(Protocol):
    """What the agent loop needs from a model, and nothing more."""

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        """Produce the next agent action given the conversation so far."""
        ...


@dataclass(frozen=True)
class TextDelta:
    """A fragment of the spoken turn, as it is generated."""

    text: str


@dataclass(frozen=True)
class StreamCompleted:
    """End of one streamed round: the assembled response, tool calls and all."""

    response: LLMResponse


StreamEvent = TextDelta | StreamCompleted


@runtime_checkable
class StreamingLLMClient(Protocol):
    """A client that can emit a turn as it is written.

    Deliberately a *separate* protocol rather than a method on ``LLMClient``.
    ``LLMClient`` is runtime-checkable and the scripted client satisfies it;
    adding a method here would silently drop the scripted client out of
    conformance and take the whole offline suite with it.
    """

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        """Yield the turn in fragments, then the assembled response."""
        ...


def stream_response(client: LLMClient, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
    """Stream from any client, streaming-capable or not.

    A client that cannot stream emits its whole turn as a single delta. That is
    the honest degradation — the caller's per-sentence guard still runs, it just
    runs on everything at once — and it is what makes the sentence-guard
    machinery testable against the scripted client with no key and no network.
    """
    if isinstance(client, StreamingLLMClient):
        yield from client.stream(messages)
        return
    response = client.respond(messages)
    if response.text:
        yield TextDelta(response.text)
    yield StreamCompleted(response)


SYSTEM_PROMPT = """\
You are a collections representative for Meridian Recovery Services, on a \
phone call about an overdue account. You are an AI, and you say so plainly \
whenever it comes up.

How you talk:
- Short sentences. This is speech, not correspondence. One idea per turn.
- Warm and matter-of-fact. The person on the other end is having a bad day; \
you are not adding to it.
- Listen for what they can actually manage, and say it back to them.
- Never argue. If they push back, acknowledge it and ask the engine what else \
is available.

How you decide — this part is not negotiable:
- You do not do arithmetic and you do not invent figures. Every amount, \
payment count, schedule and date you say out loud must have come back from a \
tool in this turn or an earlier one, verbatim.
- When the consumer proposes anything — an amount, a number of payments, a \
timeframe — call validate_consumer_offer and let it rule. Do not evaluate it \
yourself, even when the answer seems obvious.
- To put an offer on the table, call propose_offer and read back what it \
returns.
- When they refuse or push back, call record_refusal. To actually move, call \
concede; it will tell you what you may now offer. You cannot concede without a \
refusal on record, and you never move backwards to a better offer.
- When they agree to the offer standing on the table, call confirm_agreement.
- Call end_call when the conversation is finished.

You explain the engine's reasoning in plain language. The rationale code it \
returns tells you why; put that into ordinary words without embellishing it \
and without adding consequences of your own.
"""


def system_prompt(*, consumer_name: str, account_ref: str) -> Message:
    """The system message for one call.

    Deliberately thin on policy: the balance, the floors and the tier ladder
    are not stated here. If a figure is not in an engine result, the model has
    nowhere to have gotten it, which is what makes the numeric guard's job
    tractable.
    """
    return Message(
        role="system",
        content=(
            f"{SYSTEM_PROMPT}\n"
            f"The person you are calling is {consumer_name} regarding account "
            f"{account_ref}. Confirm you are speaking with them before you discuss "
            f"the account at all."
        ),
    )
