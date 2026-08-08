"""The voice transport — `voice_app.py`, offline.

No room and no LiveKit connection: the transport is driven through LiveKit's
own test framework, which runs a session in text mode. `AgentSession.run()`
feeds a consumer utterance in and returns a `RunResult` carrying the events
the turn produced, so these tests exercise the real session machinery —
turn ordering, `conversation_item_added`, history accumulating across turns —
rather than calling `llm_node` by hand.

`CollectorAgent(..., audio=False)` keeps STT and TTS out of it: text mode has
no use for them, and attaching them would have the session dial Deepgram.
The two tests at the bottom of this file are the exception: they build an
`audio=True` agent to pin the speech wiring, without starting a session.

Everything these tests assert is about the audit record — what the consumer
heard has to be what the log says was said, on the failure paths as much as
the happy one.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from livekit.agents import (
    AgentFalseInterruptionEvent,
    AgentSession,
    ConversationItemAddedEvent,
    UserInputTranscribedEvent,
    UserTranscriptionTimeoutEvent,
    inference,
    llm,
)
from livekit.agents.llm import ChatMessage

from collector.agent import NegotiationAgent
from collector.audit.events import Speaker
from collector.audit.store import AuditStore
from collector.guardrails import detect_escalation
from collector.guardrails.rings import PreCallContext
from collector.llm.base import (
    LLMResponse,
    Message,
    StreamCompleted,
    StreamEvent,
    TextDelta,
    ToolCall,
)
from collector.llm.mock_client import MockLLMClient
from collector.offers import Cadence
from collector.policy import PolicyConfig
from collector.voice_app import (
    _TRANSIENT_ERROR_APOLOGY,
    AGENT_NAME,
    GREETING_PAUSE_SECONDS,
    STT_KEYTERMS,
    STT_LANGUAGE,
    STT_MODEL,
    TTS_LANGUAGE,
    TTS_MODEL,
    TTS_VOICE,
    TURN_HANDLING,
    CollectorAgent,
    _consumer_context,
    _ConsumerContextError,
    _llm_route,
    _log_session_events,
    _log_turn_latency,
    _recording_options,
    _say_opening,
    _split_salutation,
    entrypoint,
)

# These tests park a pump thread on a gate and then release it, so every wait
# below is a *hang detector*, not a performance assertion. They have to be
# generous: `asyncio.to_thread` draws on the default executor, which the rest of
# this suite is also using for `AuditStore` workers and other pumps, so how long
# a pump waits for a thread — and how long the loop takes to reach `release.set()`
# — depends on what else is running. A budget tight enough to double as a speed
# check fails under load and says nothing useful when it does. This one still
# catches a genuine hang, just later.
_HANG = 30


POLICY = PolicyConfig.default()


class _RaisingLLMClient:
    """Stands in for a rate limit or a network blip: the opening turn works,
    every turn after it raises. Faithful to the real failure — the exception
    surfaces from inside ``NegotiationAgent.turn()``, after it has already
    recorded the consumer's utterance.

    The opening delegates to ``MockLLMClient`` rather than returning a
    hand-written line, so the greeting clears the outbound guard's disclosure
    requirements and the first raise lands on the turn under test.
    """

    def __init__(self) -> None:
        self._opening = MockLLMClient()

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        # A consumer line in the history means ``turn()`` is underway; until
        # then this is the opening, regenerations included.
        if any(m.role == "consumer" for m in messages):
            raise RuntimeError("upstream rate limit")
        return self._opening.respond(messages)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[AuditStore]:
    """Closed even when a test fails partway — the store owns a worker
    thread, and a leaked one outlives the test that opened it."""
    store = AuditStore(tmp_path / "collector.db", json_dir=tmp_path / "agreements")
    yield store
    store.close()


def _voice_agent(store: AuditStore, llm_client: object) -> tuple[NegotiationAgent, CollectorAgent]:
    negotiation_agent = NegotiationAgent(
        llm=llm_client,  # type: ignore[arg-type]
        policy=POLICY,
        store=store,
        channel="voice",
    )
    negotiation_agent.open_call(PreCallContext(account_loaded=True, within_calling_window=True))
    return negotiation_agent, CollectorAgent(negotiation_agent, audio=False)


def _agent_lines(store: AuditStore, call_id: str) -> list[str]:
    return [t.text for t in store.turns(call_id) if t.speaker is Speaker.AGENT]


# --------------------------------------------------------------------------
# A spoken line the audit log never saw
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transient_failure_apology_is_recorded_as_spoken(store: AuditStore) -> None:
    """The consumer hears the apology, so the log has to say it was said.

    R5 established that a raising ``turn()`` leaves nothing inconsistent for
    the *next* turn. C4 is about the record of *this* one: without this, the
    timeline shows the consumer speaking twice in a row with agent silence
    between, which is precisely the thing the store exists to make impossible.
    """
    negotiation_agent, agent = _voice_agent(store, _RaisingLLMClient())

    async with AgentSession() as session:
        await session.start(agent)
        result = await session.run(user_input="Who is this?")

    spoken = result.expect.next_event().is_message(role="assistant")
    assert spoken.event().item.text_content == _TRANSIENT_ERROR_APOLOGY
    result.expect.no_more_events()

    assert _TRANSIENT_ERROR_APOLOGY in _agent_lines(store, negotiation_agent.call_id)


@pytest.mark.asyncio
async def test_transient_failure_turn_is_counted(store: AuditStore) -> None:
    """The errored turn still counts. ``turn()`` never reaches its own
    ``self.turns.append`` when ``_act()`` raises — but
    ``record_fallback_speech`` appends an ``AgentTurn`` of its
    own on the way out, so the call report and the ``CallEnded`` count agree
    with what the consumer actually experienced: one turn, one reply."""
    negotiation_agent, agent = _voice_agent(store, _RaisingLLMClient())

    async with AgentSession() as session:
        await session.start(agent)
        await session.run(user_input="Who is this?")

    assert negotiation_agent.close().turns == 1


@pytest.mark.asyncio
async def test_transient_failure_apology_reaches_the_model_context(store: AuditStore) -> None:
    """The model's transcript has to match the call's. If the apology is
    spoken but never appended, the next turn is generated against a history
    that skips a sentence the consumer actually heard."""
    negotiation_agent, agent = _voice_agent(store, _RaisingLLMClient())

    async with AgentSession() as session:
        await session.start(agent)
        await session.run(user_input="Who is this?")

    assert any(
        m.role == "agent" and m.content == _TRANSIENT_ERROR_APOLOGY
        for m in negotiation_agent.messages
    )


@pytest.mark.asyncio
async def test_successful_turn_is_recorded_once(store: AuditStore) -> None:
    """The guard against fixing C4 by double-recording, restated for the
    streaming path.

    ``stream_turn`` guards and records one *sentence* at a time — that is what
    lets TTS start before the turn finishes — so a turn no longer maps to a
    single audit row. The invariant it has to preserve is the same one: what
    the log says was said is exactly what the consumer heard, once. So the rows
    this turn added, joined the way ``llm_node`` joins its chunks, are the
    assistant message character for character — no line dropped, none doubled,
    and no separator invented on either side.
    """
    negotiation_agent, agent = _voice_agent(store, MockLLMClient())
    opening = len(_agent_lines(store, negotiation_agent.call_id))

    async with AgentSession() as session:
        await session.start(agent)
        result = await session.run(user_input="Yes, this is Dana.")

    spoken = result.expect.next_event().is_message(role="assistant").event().item.text_content
    recorded = _agent_lines(store, negotiation_agent.call_id)[opening:]
    assert recorded, "the consumer heard a reply the audit log says was never spoken"
    assert " ".join(recorded) == spoken


@pytest.mark.asyncio
async def test_consecutive_turns_share_one_session_history(store: AuditStore) -> None:
    """What driving ``llm_node`` directly could never show: the session
    threads its own history across turns. ``llm_node`` reads only the last
    item, so a second ``run()`` on the same session has to arrive with the
    second consumer line on top — not the first — or every turn after the
    opening would answer the wrong question."""
    negotiation_agent, agent = _voice_agent(store, MockLLMClient())

    async with AgentSession() as session:
        await session.start(agent)
        await session.run(user_input="Yes, this is Dana.")
        second = await session.run(user_input="How much do I owe?")

        heard = [
            item.text_content
            for item in session.history.items
            if getattr(item, "role", None) == "user"
        ]
        said = [
            item.text_content
            for item in session.history.items
            if getattr(item, "role", None) == "assistant"
        ]

    second.expect.next_event().is_message(role="assistant")
    second.expect.no_more_events()
    # ``heard`` is what proves no replay: the second ``run()`` saw its own
    # utterance on top, not the first one again.
    assert heard == ["Yes, this is Dana.", "How much do I owe?"]
    # The separate claim that each turn spoke once. One assistant message per
    # turn, and the audit rows behind them — many per turn on the streaming
    # path, since each guarded sentence is recorded as it is released — account
    # for those messages and nothing else. The opening line, recorded by
    # ``open_call()`` before the session existed, is the row skipped here.
    assert len(said) == 2
    assert " ".join(_agent_lines(store, negotiation_agent.call_id)[1:]) == " ".join(said)


# --------------------------------------------------------------------------
# The sentence reaches TTS before the turn finishes
# --------------------------------------------------------------------------
#
# The whole point of routing ``llm_node`` through ``stream_turn`` rather than
# ``turn()``. A test that only collects every chunk and compares the text
# passes identically on both paths and proves nothing, so the clients below
# *block* mid-round: the first sentence has to arrive while the model call that
# produced it is still open. On the ``turn()`` path that is a deadlock, which
# is exactly what makes these tests discriminating rather than decorative.


class _GatedStreamClient:
    """One turn, two sentences, with a gate between them.

    ``stream()`` hands over the first sentence and then blocks until the test
    releases it, so a chunk observed before ``release`` is set is proof the
    generation had not finished. The opening delegates to ``MockLLMClient`` so
    the greeting clears the disclosure rules and the turn under test starts
    from a clean guard state.
    """

    FIRST = "That works."
    SECOND = "Let me read the arrangement back to you."

    def __init__(self) -> None:
        self._opening = MockLLMClient()
        self.release = threading.Event()
        self.first_delta_sent = threading.Event()

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        return self._opening.respond(messages)

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        yield TextDelta(f"{self.FIRST} ")
        self.first_delta_sent.set()
        assert self.release.wait(timeout=_HANG), "the test never released the gate"
        yield TextDelta(f"{self.SECOND} ")
        yield StreamCompleted(LLMResponse(text=f"{self.FIRST} {self.SECOND}"))


@pytest.mark.asyncio
async def test_the_first_sentence_arrives_before_the_turn_completes(store: AuditStore) -> None:
    """The change this file exists to pin: TTS gets sentence one while the model
    is still writing sentence two.

    ``turn()`` could not satisfy this at any latency — it returns one string
    after the last round trip — so the assertion is not "the text is right" but
    "the chunk arrived while the round was open". The gate makes that
    observable instead of a timing guess.
    """
    _, agent = _voice_agent(store, client := _GatedStreamClient())
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="Yes, that's fine.")

    stream = agent.llm_node(ctx, [], {})  # type: ignore[arg-type]
    try:
        first = await asyncio.wait_for(anext(stream), timeout=_HANG)

        assert first == _GatedStreamClient.FIRST
        # The round is provably mid-flight: the client is parked on the gate,
        # so no `StreamCompleted` has been emitted and no round trip has
        # returned. This is the assertion `turn()` cannot pass.
        assert client.first_delta_sent.is_set()
        assert not client.release.is_set()

        client.release.set()
        rest = [chunk async for chunk in stream]
    finally:
        client.release.set()
        await stream.aclose()

    # And every chunk after the first carries the separator the framework does
    # not add, so the assistant message reads as prose rather than "works.Let".
    # Asserted on shape rather than on the exact second sentence: the outbound
    # guard decides what the rest of a turn is allowed to be, and this test is
    # about when chunks arrive, not about what the guard permits.
    assert rest, "the turn stopped after one sentence"
    assert all(chunk.startswith(" ") for chunk in rest)


class _ClosingStreamClient:
    """Ends the call in round one, then speaks the goodbye in round two — with
    a gate part-way through it, because the goodbye is the line a consumer is
    most likely to talk over."""

    GOODBYE = "You'll get an email confirming everything."
    TAIL = "Take care now."

    def __init__(self) -> None:
        self._opening = MockLLMClient()
        self.release = threading.Event()
        self.rounds = 0

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        return self._opening.respond(messages)

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        self.rounds += 1
        if self.rounds == 1:
            yield StreamCompleted(
                LLMResponse(
                    tool_calls=(
                        ToolCall(name="end_call", arguments={"reason": "done"}, call_id="toolu_1"),
                    )
                )
            )
            return
        yield TextDelta(f"{self.GOODBYE} ")
        assert self.release.wait(timeout=_HANG), "the test never released the gate"
        yield TextDelta(f"{self.TAIL} ")
        yield StreamCompleted(LLMResponse(text=f"{self.GOODBYE} {self.TAIL}"))


@pytest.mark.asyncio
async def test_a_barge_in_on_the_closing_turn_still_hangs_up(
    store: AuditStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call-killing bug the streaming path introduces if the shutdown check
    sits after the loop instead of in a ``finally``.

    A tool ends the call in round one; ``stream_turn`` owes one more round so
    the consumer hears the arrangement read back. If they talk over *that*, the
    framework tears this generator down at the ``yield`` — and a check placed
    after the loop never runs. The room then stays open with an agent whose
    every later turn returns early on ``ended``: silent for the rest of a call
    that never hangs up.
    """
    negotiation_agent, agent = _voice_agent(store, client := _ClosingStreamClient())
    hung_up: list[bool] = []
    monkeypatch.setattr(agent, "_end_the_call", lambda: hung_up.append(True))
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="Yes, let's do that.")

    stream = agent.llm_node(ctx, [], {})  # type: ignore[arg-type]
    try:
        assert await asyncio.wait_for(anext(stream), timeout=_HANG) == _ClosingStreamClient.GOODBYE
        # The precondition that makes the barge-in dangerous rather than
        # ordinary: the call is already closed out, and only this generator
        # knows to shut the room.
        assert negotiation_agent.ended
        assert hung_up == []
    finally:
        client.release.set()
        # The barge-in itself. `aclose()` is what the framework calls on the
        # node when a SpeechHandle is cancelled.
        await stream.aclose()

    assert hung_up == [True], "a barge-in on the closing turn left the room open"
    # And the turn is still on the record — those sentences were audio, so
    # abandoning the stream must not lose the exchange. It lands once the pump
    # has closed the generator, which is what ``drain()`` waits for.
    await asyncio.wait_for(agent.drain(timeout=_HANG), timeout=_HANG * 2)
    assert negotiation_agent.turns
    assert negotiation_agent.turns[-1].ended


@pytest.mark.asyncio
async def test_an_abandoned_turn_finishes_writing_before_the_store_closes(
    store: AuditStore,
) -> None:
    """``drain()`` is what keeps the audit chain single-writer across a
    barge-in.

    The gate is *not* released before ``aclose()`` here, deliberately: that is
    what leaves the pump genuinely parked inside ``next()`` when the generator is
    closed, so the abandonment is cooperative — ``abandoned`` is what stops it —
    rather than the generator simply having run to completion first. Releasing
    the gate before closing (as the tests above do, since they are about
    something else) makes this test pass with ``drain()`` deleted.

    What has to land before ``store.close()`` is the ``AgentTurn`` that
    ``CallEnded.turn_count`` is taken from. Asserted on the *count*: the
    ``TurnRecorded`` rows for sentences already yielded were written by
    ``_guard_sentence`` before ``anext()`` ever returned them, so looking for
    one of those in the store proves nothing about the drain.
    """
    negotiation_agent, agent = _voice_agent(store, client := _GatedStreamClient())
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="Yes, that's fine.")

    stream = agent.llm_node(ctx, [], {})  # type: ignore[arg-type]
    await asyncio.wait_for(anext(stream), timeout=_HANG)
    turns_before = len(negotiation_agent.turns)

    await stream.aclose()
    # Still parked on the gate: the turn cannot have been recorded yet, which is
    # precisely the window in which `store.close()` would lose it.
    assert len(negotiation_agent.turns) == turns_before
    client.release.set()

    await asyncio.wait_for(agent.drain(timeout=_HANG), timeout=_HANG * 2)

    assert len(negotiation_agent.turns) == turns_before + 1


@pytest.mark.asyncio
async def test_drain_waits_for_every_pump_not_just_the_last(store: AuditStore) -> None:
    """Two generations can be in flight at once, and a cancelled ``to_thread``
    does not stop the thread under it — so the abandoned pump keeps generating,
    keeps guarding sentences, and keeps writing to the store.

    Tracked in a single slot, the older pump went untracked the moment the newer
    one replaced it. ``drain()`` returned as soon as the newer one finished,
    ``store.close()`` ran, and the older pump's next write hit a closed store:
    a lost row on the single-writer compliance store, an ``AgentTurn`` appended
    after ``CallEnded.turn_count`` was taken, and no diagnostic anywhere,
    because the failure is relayed to a queue nobody reads.
    """
    negotiation_agent, agent = _voice_agent(store, first := _GatedStreamClient())
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="Yes, that's fine.")

    try:
        # Generation one: started, parked on its gate, then abandoned.
        stale = agent.llm_node(ctx, [], {})  # type: ignore[arg-type]
        await asyncio.wait_for(anext(stale), timeout=_HANG)
        (stale_pump,) = agent._pumps
        await stale.aclose()

        # Generation two, on a negotiation agent of its own, runs to completion.
        agent._negotiation_agent = _voice_agent(store, second := _GatedStreamClient())[0]
        second.release.set()
        fresh = agent.llm_node(ctx, [], {})  # type: ignore[arg-type]
        assert [chunk async for chunk in fresh]

        # The load-bearing assertion, and the one a single slot fails: the
        # abandoned pump is *still tracked* after a second generation replaced
        # it, and still live — parked on its gate, so this cannot race. Under
        # one slot it was not tracked at all and `drain()` had nothing to wait
        # on. Asserted about this pump rather than by counting, because
        # generation two's task settles a tick after its generator drains.
        assert stale_pump in agent._pumps
        assert not stale_pump.done()

        turns_before = len(negotiation_agent.turns)
    finally:
        first.release.set()

    await asyncio.wait_for(agent.drain(timeout=_HANG), timeout=_HANG * 2)

    # Waited, not merely returned — so the store is safe to close.
    assert stale_pump.done()
    assert len(negotiation_agent.turns) == turns_before + 1


# --------------------------------------------------------------------------
# The ``llm_node`` guards a session cannot reach
# --------------------------------------------------------------------------
#
# ``AgentSession.run(user_input=...)`` always puts a well-formed user message
# on top of the context, so the defensive branches at the head of ``llm_node``
# are unreachable through the harness above — the session is the very thing
# that makes them impossible. They still have to hold: in a real call the
# context is assembled by the framework from STT output, not by this test.
# Driving ``llm_node`` directly is the only way to present the inputs it
# guards against, so these three keep doing it deliberately.


async def _drive(agent: CollectorAgent, ctx: llm.ChatContext) -> list[str]:
    return [chunk async for chunk in agent.llm_node(ctx, [], {})]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_context_with_no_user_turn_on_top_says_nothing(
    store: AuditStore, caplog: pytest.LogCaptureFixture
) -> None:
    """``llm_node`` reads only the last item. Handed a context whose last item
    is the agent's own line, it must not feed that back through ``turn()`` as
    if the consumer had said it."""
    _, agent = _voice_agent(store, MockLLMClient())
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="assistant", content="Is that something you can take care of today?")

    with caplog.at_level(logging.WARNING, logger="collector-voice"):
        assert await _drive(agent, ctx) == []

    assert "no user turn on top of the context" in caplog.text


@pytest.mark.asyncio
async def test_an_empty_utterance_says_nothing(store: AuditStore) -> None:
    """STT can emit an empty final transcript. Passing it to ``turn()`` would
    spend a model round trip — and a recorded consumer turn — on silence."""
    _, agent = _voice_agent(store, MockLLMClient())
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="")

    assert await _drive(agent, ctx) == []


@pytest.mark.asyncio
async def test_an_ended_call_says_nothing_more(store: AuditStore) -> None:
    """``turn()`` raises on an ended call. A late utterance — the consumer
    talking over the closing line — must be dropped here rather than taking
    the call down after it has already been closed out."""
    negotiation_agent, agent = _voice_agent(store, MockLLMClient())
    negotiation_agent.close()
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="Wait, one more thing.")

    assert await _drive(agent, ctx) == []


# --------------------------------------------------------------------------
# The multi-round-trip turn has to be measurable
# --------------------------------------------------------------------------


def _assistant(metrics: dict[str, float]) -> ChatMessage:
    message = ChatMessage(role="assistant", content=["Here's what I can do."])
    message.metrics = metrics  # type: ignore[typeddict-item]
    return message


def _only_record(caplog: pytest.LogCaptureFixture, message: str) -> logging.LogRecord:
    """The single record with this message, so a test can read the structured
    fields off it. Asserts there is exactly one — a duplicated log line is a
    duplicated measurement."""
    matching = [r for r in caplog.records if r.getMessage() == message]
    assert len(matching) == 1, f"expected one {message!r} record, got {len(matching)}"
    return matching[0]


async def _emit_item(message: ChatMessage) -> None:
    """Register the latency hook on a session and hand it one history item.
    ``AgentSession`` grabs the running loop in its constructor, so this needs
    to happen inside one."""
    session: AgentSession[None] = AgentSession()
    _log_turn_latency(session)
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=message))


@pytest.mark.asyncio
async def test_assistant_turn_latency_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """``llm_node_ttft`` is the whole ``turn()`` here, not a first token — the
    override yields only once the blocking turn is done."""
    with caplog.at_level(logging.INFO, logger="collector-voice"):
        await _emit_item(
            _assistant({"e2e_latency": 2.5, "llm_node_ttft": 2.1, "tts_node_ttfb": 0.18})
        )

    record = _only_record(caplog, "turn_latency")
    # Asserted as record attributes, not as a substring of the message: they
    # are emitted through `extra=`, and the framework's JSON formatter turns
    # each one into a real top-level field. Reading them back the same way is
    # what proves a log drain could filter on `e2e_ms`.
    assert (record.e2e_ms, record.turn_ms, record.tts_ttfb_ms) == (2500, 2100, 180)


@pytest.mark.asyncio
async def test_missing_metrics_do_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    """A turn can end without playback ever starting (interrupted, failed
    TTS); the metrics are simply absent and must not take the call down."""
    with caplog.at_level(logging.INFO, logger="collector-voice"):
        await _emit_item(_assistant({}))

    record = _only_record(caplog, "turn_latency")
    # `None`, not the string "n/a" — a missing measurement stays missing
    # rather than becoming a value that sorts and filters like one.
    assert (record.e2e_ms, record.turn_ms, record.tts_ttfb_ms) == (None, None, None)


@pytest.mark.asyncio
async def test_consumer_turns_are_not_logged_as_agent_latency(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """User items carry their own, different metric set — logging them under
    the same line would make the agent look faster than it is."""
    user = ChatMessage(role="user", content=["I can do fifty a month."])
    user.metrics = {"transcription_delay": 0.3}  # type: ignore[typeddict-item]
    with caplog.at_level(logging.INFO, logger="collector-voice"):
        await _emit_item(user)

    assert "turn latency" not in caplog.text


# --------------------------------------------------------------------------
# Dispatch metadata must not fail open
# --------------------------------------------------------------------------


class _FakeJob:
    def __init__(self, metadata: str) -> None:
        self.metadata = metadata


class _FakeCtx:
    def __init__(self, metadata: str) -> None:
        self.job = _FakeJob(metadata)


def test_absent_metadata_falls_back_to_the_fixture_consumer() -> None:
    """Local and manual testing — legitimate."""
    name, account = _consumer_context(_FakeCtx(""))  # type: ignore[arg-type]
    assert (name, account) == ("Dana Whitfield", "ACCT-4471")


def test_well_formed_metadata_is_used() -> None:
    ctx = _FakeCtx(json.dumps({"consumer_name": "Rae Okonkwo", "account_ref": "ACCT-9001"}))
    assert _consumer_context(ctx) == ("Rae Okonkwo", "ACCT-9001")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "metadata",
    [
        "{not json",
        '["a", "list"]',
        '{"account_ref": "ACCT-9001"}',
        '{"consumer_name": "", "account_ref": "ACCT-9001"}',
        '{"consumer_name": "Rae Okonkwo", "account_ref": null}',
    ],
)
def test_malformed_metadata_refuses_the_call(metadata: str) -> None:
    """Never silently substitute a different real consumer's identity."""
    with pytest.raises(_ConsumerContextError):
        _consumer_context(_FakeCtx(metadata))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Speech is served by LiveKit Inference, and which voice it is matters
# --------------------------------------------------------------------------


def test_audio_mode_wires_speech_through_livekit_inference(store: AuditStore) -> None:
    """The rest of this file runs `audio=False`, so nothing else exercises the
    speech pipeline at all. This pins the two models and — the part that would
    otherwise change silently — the voice the consumer hears, which the Cartesia
    plugin used to pick from a library default."""
    negotiation_agent = NegotiationAgent(
        llm=MockLLMClient(), policy=POLICY, store=store, channel="voice"
    )
    agent = CollectorAgent(negotiation_agent, audio=True)

    assert isinstance(agent.stt, inference.STT)
    assert isinstance(agent.tts, inference.TTS)
    assert (STT_MODEL, STT_LANGUAGE) == ("deepgram/nova-3", "en-US")
    assert (TTS_MODEL, TTS_VOICE, TTS_LANGUAGE) == (
        "cartesia/sonic-3",
        "f786b574-daa5-4673-aa0c-cbe3e8534c02",
        "en",
    )


def test_text_mode_attaches_no_speech_pipeline(store: AuditStore) -> None:
    """`audio=False` must stay inert: the STT pump opens a Deepgram stream the
    moment a session starts, which in a test is a live network call."""
    negotiation_agent = NegotiationAgent(
        llm=MockLLMClient(), policy=POLICY, store=store, channel="voice"
    )
    agent = CollectorAgent(negotiation_agent, audio=False)
    assert agent.stt is None
    assert agent.tts is None


# --------------------------------------------------------------------------
# Recording mode — the default must not move
# --------------------------------------------------------------------------


def test_recording_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The load-bearing assertion of this section. An unset environment must
    upload nothing: the SDK's own default is "record everything", so a
    regression here is a silent third-party copy of consumer audio."""
    monkeypatch.delenv("COLLECTOR_VOICE_RECORDING", raising=False)
    assert _recording_options() is False


@pytest.mark.parametrize("mode", ["off", "", "typo", "TRUE", "1", "yes"])
def test_only_the_named_modes_record(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Anything that is not exactly `diagnostics` or `full` records nothing.
    A misspelled opt-in must fail closed — the direction a mistake should
    resolve when the alternative is uploading a consumer's speech."""
    monkeypatch.setenv("COLLECTOR_VOICE_RECORDING", mode)
    assert _recording_options() is False


def test_diagnostics_mode_keeps_speech_out_of_the_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The granular middle ground: traces and logs for latency and failures,
    with the consumer's words — audio *and* transcript — still excluded."""
    monkeypatch.setenv("COLLECTOR_VOICE_RECORDING", "diagnostics")
    assert _recording_options() == {
        "audio": False,
        "transcript": False,
        "traces": True,
        "logs": True,
    }


def test_full_mode_is_the_sdk_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLLECTOR_VOICE_RECORDING", "FULL")
    assert _recording_options() is True


def test_llm_route_names_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLLECTOR_LLM", raising=False)
    assert _llm_route() == "anthropic"
    monkeypatch.setenv("COLLECTOR_LLM", "openrouter")
    assert _llm_route() == "openrouter"


def test_llm_client_dispatches_the_livekit_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eval simulator certifies whatever this function returns
    (``tests/evals/simulator.py::_agent_llm``), so a route the env selects but
    this function ignores lets the suite certify one model while production
    answers on another — the exact drift that let Gemini carry calls
    uncertified the first time around."""
    from collector.llm.livekit_client import LiveKitInferenceClient
    from collector.voice_app import _llm_client

    monkeypatch.setenv("COLLECTOR_LLM", "livekit")
    monkeypatch.setenv("LIVEKIT_API_KEY", "lk_test_key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "lk_test_secret")
    assert _llm_route() == "livekit"
    assert isinstance(_llm_client(), LiveKitInferenceClient)


# --------------------------------------------------------------------------
# Session-event logging
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_transcripts_log_length_not_words(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The PII boundary, asserted rather than merely commented. The consumer's
    words go to the audit store, which is the record built to hold them; the
    log — which leaves the process by stdout and any drain — gets a count."""
    session: AgentSession[None] = AgentSession()
    _log_session_events(session)
    utterance = "My account number is 4471 and I can pay fifty a month."

    with caplog.at_level(logging.INFO, logger="collector-voice"):
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript=utterance, is_final=True),
        )

    record = _only_record(caplog, "user_transcript")
    assert record.chars == len(utterance)
    # Every field of every record, not `caplog.text` — the formatted text
    # never contains the `extra` dict, so checking it would pass even if the
    # transcript had been put in `extra={"transcript": ...}`, which is exactly
    # the mistake this test exists to prevent.
    leaked = [
        (r.getMessage(), key)
        for r in caplog.records
        for key, value in r.__dict__.items()
        if any(needle in str(value) for needle in ("4471", "fifty", "pay"))
    ]
    assert not leaked, f"the consumer's words reached the log: {leaked}"


@pytest.mark.asyncio
async def test_interim_transcripts_are_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """STT emits interim results continuously; logging them would bury the
    call in noise and repeat the same speech a dozen times over."""
    session: AgentSession[None] = AgentSession()
    _log_session_events(session)

    with caplog.at_level(logging.INFO, logger="collector-voice"):
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="my account", is_final=False),
        )

    assert "user_transcript" not in caplog.text


@pytest.mark.asyncio
async def test_transcription_timeout_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The consumer spoke and STT returned nothing — otherwise the call just
    appears to go quiet with no recorded cause."""
    session: AgentSession[None] = AgentSession()
    _log_session_events(session)

    with caplog.at_level(logging.INFO, logger="collector-voice"):
        session.emit(
            "user_transcription_timeout",
            UserTranscriptionTimeoutEvent(speech_duration=3.2, vad_speech_started_at=0.0),
        )

    record = _only_record(caplog, "transcription_timeout")
    assert record.levelno == logging.WARNING
    assert record.speech_duration_ms == 3200


@pytest.mark.asyncio
async def test_false_interruption_records_whether_speech_resumed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fires with the stock session: `false_interruption_timeout` defaults to
    2.0s and `resume_false_interruption` to True, so this handler is live
    without any turn-handling configuration."""
    session: AgentSession[None] = AgentSession()
    _log_session_events(session)

    with caplog.at_level(logging.INFO, logger="collector-voice"):
        session.emit("agent_false_interruption", AgentFalseInterruptionEvent(resumed=True))

    assert _only_record(caplog, "false_interruption").resumed is True


def test_the_worker_registers_under_a_dispatch_name() -> None:
    """An empty `agent_name` puts the worker on automatic dispatch, where it
    joins every room in the project and receives no job metadata — so
    `_consumer_context()` would fall through to the fixture consumer on every
    real call, and the worker could be pulled into a room nobody authorized.
    The name is what keeps dispatch explicit."""
    assert AGENT_NAME
    assert AGENT_NAME == "collections-negotiator"


# --------------------------------------------------------------------------
# The opening line's delivery — a beat after the salutation
# --------------------------------------------------------------------------


class _SayRecorder:
    """Stands in for `AgentSession`, capturing what would be spoken. Only
    `say()` is reached — `_say_opening` touches nothing else on the session."""

    def __init__(self) -> None:
        self.said: list[tuple[str, bool]] = []

    async def say(self, text: str, *, allow_interruptions: bool = True) -> None:
        self.said.append((text, allow_interruptions))


@pytest.fixture
def no_pause(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Records what `_say_opening` would have slept for, without spending it."""
    slept: list[float] = []

    async def _record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("collector.voice_app.asyncio.sleep", _record)
    return slept


def _words(text: str) -> list[str]:
    """Words, with the boundary punctuation the split is allowed to change
    stripped off. A digit or a `$` is never boundary punctuation, so a figure
    like `$1,000.00` survives this intact and a dropped one would show up."""
    return [w for w in (t.strip(".,;:!?—-") for t in text.lower().split()) if w]


OPENINGS_WITH_A_SALUTATION = [
    "Hi, this is Avery calling from Meridian Recovery Services.",
    "Hello. This is Avery from Meridian Recovery Services.",
    "Hey there — this is Avery calling about your account.",
    "Hi there, this is Avery. This call is handled by an AI assistant.",
    "Good morning, this is Avery from Meridian Recovery Services.",
    "Good afternoon! This is Avery.",
]


@pytest.mark.parametrize("opening", OPENINGS_WITH_A_SALUTATION)
def test_splitting_the_salutation_never_changes_the_words(opening: str) -> None:
    """The load-bearing assertion of this section. The pause is a delivery
    choice: it may change *when* words are said and never *which*, or in what
    order. What the outbound guard cleared and `open_call()` logged was this
    line's content — the figures, the claims, the disclosure — so nothing the
    split does may add, drop, or reorder a word."""
    split = _split_salutation(opening)
    assert split is not None
    salutation, remainder = split
    assert _words(salutation) + _words(remainder) == _words(opening)


@pytest.mark.parametrize("opening", OPENINGS_WITH_A_SALUTATION)
def test_each_half_is_speakable_on_its_own(opening: str) -> None:
    """Each half is synthesized as a separate utterance, so each has to stand
    as one. A salutation left as "Hi," gets the rising intonation of a clause
    that continues — into a pause that says it doesn't — and a remainder left
    lowercase opens mid-clause. Both read as a machine reciting a fragment."""
    split = _split_salutation(opening)
    assert split is not None
    salutation, remainder = split
    assert salutation.endswith((".", "!", "?")), salutation
    assert remainder[0].isupper(), remainder


def test_the_salutation_is_all_that_is_split_off() -> None:
    assert _split_salutation("Hi, this is Avery calling from Meridian Recovery Services.") == (
        "Hi.",
        "This is Avery calling from Meridian Recovery Services.",
    )
    assert _split_salutation("Hey there — this is Avery calling about your account.") == (
        "Hey there.",
        "This is Avery calling about your account.",
    )


def test_the_disclosure_survives_the_split() -> None:
    """The second half is the half the AI disclosure lives in. It has to arrive
    there whole — a split that clipped it would strip the one sentence the
    opening exists to deliver."""
    split = _split_salutation(
        "Hi, this call is handled by an AI assistant. Am I speaking with Dana?"
    )
    assert split is not None
    _, remainder = split
    assert "handled by an AI assistant" in remainder


@pytest.mark.parametrize(
    "opening",
    [
        "This is Avery calling from Meridian Recovery Services.",
        "Am I speaking with the account holder?",
        # A salutation with nothing after it: no second half to pause before,
        # so it is spoken as one piece rather than left hanging on the line.
        "Hello.",
        "Hi",
        "",
    ],
)
def test_a_line_without_a_leading_salutation_is_not_split(opening: str) -> None:
    assert _split_salutation(opening) is None


def test_a_salutation_that_is_not_leading_is_not_split() -> None:
    """Anchored at the start, and on a word boundary. A greeting word inside a
    later clause is not a greeting, and "Hi" inside "Highway" is not a word —
    splitting at either would cut the line somewhere meaningless."""
    assert _split_salutation("I'd say hello to a payment plan.") is None
    assert _split_salutation("Highway Recovery Services calling.") is None


@pytest.mark.asyncio
async def test_the_opening_is_spoken_in_two_parts_with_a_pause(no_pause: list[float]) -> None:
    """The change itself: salutation, a beat, then the substance."""
    session = _SayRecorder()

    await _say_opening(  # type: ignore[arg-type]
        session, "Hi, this is Avery calling from Meridian Recovery Services."
    )

    assert [text for text, _ in session.said] == [
        "Hi.",
        "This is Avery calling from Meridian Recovery Services.",
    ]
    assert no_pause == [GREETING_PAUSE_SECONDS]


@pytest.mark.asyncio
async def test_neither_half_of_the_opening_can_be_interrupted(no_pause: list[float]) -> None:
    """The opening carries the AI disclosure. Splitting it into two `say()`
    calls must not hand a talkative consumer a way to talk over the second
    half — which is the half the disclosure is in."""
    session = _SayRecorder()

    await _say_opening(  # type: ignore[arg-type]
        session, "Hi, this is Avery. This call is handled by an AI assistant."
    )

    assert len(session.said) == 2
    assert all(allow_interruptions is False for _, allow_interruptions in session.said)


@pytest.mark.asyncio
async def test_an_opening_without_a_salutation_is_spoken_whole(no_pause: list[float]) -> None:
    """No salutation, no pause, one `say()`, and the line untouched — the
    behaviour before this change, unchanged."""
    session = _SayRecorder()
    opening = "This is Avery calling from Meridian Recovery Services."

    await _say_opening(session, opening)  # type: ignore[arg-type]

    assert session.said == [(opening, False)]
    assert no_pause == []


# --------------------------------------------------------------------------
# Preemptive generation — `turn()` must never run on an uncommitted transcript
# --------------------------------------------------------------------------


def test_preemptive_generation_is_off() -> None:
    """`turn()` is not a pure text->text node. It advances `_turn_index`,
    records a ConsumerUtterance, and can move the concession ladder — and
    `llm_node` runs it under `asyncio.to_thread`, which cannot be cancelled.
    Left on, the SDK calls it speculatively on transcripts that are not yet a
    committed turn and then throws the result away, having already written it
    to the audit log. Two rows and two concessions for one thing the consumer
    said once.

    Asserted against the constant rather than a live call because no test in
    this file can reach the real path: preemptive generation fires from
    `audio_recognition.py`, which needs an audio stream, and this suite runs
    the session in text mode. The config *is* the guarantee here."""
    assert TURN_HANDLING["preemptive_generation"] == {"enabled": False}


def test_preemptive_tts_is_never_enabled() -> None:
    """One step worse than a speculative `turn()`: audio synthesized for a
    reply derived from a sentence the consumer had not finished saying. Off by
    SDK default — this pins it so it cannot be turned on alongside anything
    else added to `TURN_HANDLING` later."""
    assert TURN_HANDLING["preemptive_generation"].get("preemptive_tts") is not True


def test_the_session_is_built_with_the_turn_handling_options() -> None:
    """The constant is only worth anything if `entrypoint()` passes it. Guards
    against the options being defined and then quietly not wired in."""
    source = inspect.getsource(entrypoint)
    assert "AgentSession(turn_handling=TURN_HANDLING)" in source


def test_endpointing_commits_unsure_turns_faster_than_the_sdk_default() -> None:
    """Measured calls sat the streaming default's full 2.5s ceiling on three
    of five turns — the ceiling only binds when the turn detector is unsure,
    and 2.5s of dead air on a phone reads as a hang. The floor stays at the
    streaming default: the log has warned that transcripts can arrive after
    the turn commits, and a lower floor widens exactly that race."""
    endpointing = TURN_HANDLING["endpointing"]
    assert endpointing["max_delay"] <= 1.5
    assert endpointing["min_delay"] >= 0.3


# --------------------------------------------------------------------------
# STT keyterm biasing — help the recognizer, never steer it
# --------------------------------------------------------------------------


def test_no_keyterm_carries_a_figure() -> None:
    """The load-bearing assertion of this section, and the one rule that is not
    a preference. Keyterms bias what the recognizer is willing to hear. A
    dollar amount in this list would bias transcription of what the *consumer*
    said toward the amounts this agent is authorized to offer — manufacturing
    agreement to a number they never spoke, upstream of every guardrail, and
    invisible in the audit log because the false figure would be recorded as
    what they said. No digits, no currency, ever."""
    for term in STT_KEYTERMS:
        assert not any(c.isdigit() for c in term), term
        assert "$" not in term, term


def test_keyterms_stay_within_the_provider_limits() -> None:
    """Deepgram accepts up to 100 terms totaling 1200 characters
    (`inference/stt.py:79`). Over either, the request is the provider's problem
    at call time — on a live call."""
    assert len(STT_KEYTERMS) <= 100
    assert sum(len(t) for t in STT_KEYTERMS) <= 1200


def test_keyterms_are_unique_and_lowercase() -> None:
    """`_keyterms_extra_for_model` dedupes with `dict.fromkeys`, so a duplicate
    is silently dropped rather than flagged — worth catching here instead."""
    assert len(set(STT_KEYTERMS)) == len(STT_KEYTERMS)
    assert all(t == t.lower().strip() for t in STT_KEYTERMS)


@pytest.mark.parametrize(
    ("term", "utterance"),
    [
        ("cease and desist", "I want you to cease and desist."),
        ("identity theft", "This is identity theft, not my account."),
        ("validation", "I want validation of this debt."),
        ("retained", "I have retained an attorney."),
        ("bankruptcy", "I filed for bankruptcy."),
        ("chemo", "I'm going through chemo right now."),
        ("terminally", "I'm terminally ill."),
        ("suicidal", "I'm suicidal."),
    ],
)
def test_escalation_keyterms_key_on_a_real_trigger(term: str, utterance: str) -> None:
    """Every escalation term in the list has to actually drive an escalation,
    or it is dead weight diluting the ones that do. A missed trigger here is
    not a degraded call — it is the consumer asserting a right and the agent
    carrying on negotiating over it."""
    assert term in STT_KEYTERMS
    assert detect_escalation(utterance), utterance


def test_cadence_keyterms_name_a_real_cadence() -> None:
    """ "biweekly" is the one the recognizer actually gets wrong ("by weekly"),
    and a misheard cadence is a misheard schedule."""
    assert "biweekly" in STT_KEYTERMS
    assert "biweekly" in {c.value for c in Cadence}


def test_common_words_are_not_boosted() -> None:
    """Keyterm prompting is for terms the recognizer struggles with. Padding
    the list with words Nova-3 already gets right spends the budget and dilutes
    the terms that need the help."""
    assert not ({"weekly", "monthly", "payment", "dollars", "account"} & set(STT_KEYTERMS))


def test_the_stt_is_built_with_the_keyterms(store: AuditStore) -> None:
    """The list is only worth anything if it reaches the recognizer. Guards
    against it being defined and then quietly not wired in."""
    source = inspect.getsource(CollectorAgent.__init__)
    assert '"keyterm": STT_KEYTERMS' in source
