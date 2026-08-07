"""LiveKit Agents worker.

    lk agent dev src/collector/voice_app.py    # connect to LiveKit, wait for a room
    uv run collector-voice start               # production mode

Same core as ``text_app.py``: the same ``NegotiationAgent``, the same tools,
the same guardrails, the same audit store. Only the transport differs — audio
in, audio out, instead of a terminal.

STT (Deepgram) and TTS (Cartesia) are the framework's, served through LiveKit
Inference. The turn itself is not:
``llm_node`` is overridden below to bypass the framework's own LLM and
function-calling machinery entirely and hand the transcribed consumer
utterance straight to ``NegotiationAgent.stream_turn()``, which resolves tool
calls, guardrails, and disclosures itself and emits the reply a guarded
sentence at a time. What this file adds is only the plumbing to get a sentence
in and sentences out — and, because that generator is synchronous and this
node is not, the pump in ``_stream_sentences`` that bridges the two without
losing the audit write when a consumer talks over the agent.

The bypass is not an optimization, it is the architecture: every dollar figure
is computed by the decision engine and cleared by the outbound guard *before*
it becomes a text chunk here. Handing the tools to the framework's own
function-calling loop instead would put the numeric guard downstream of
nothing, which is the one thing this design does not permit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import threading
from collections.abc import AsyncIterable, AsyncIterator
from typing import Never, NoReturn

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentFalseInterruptionEvent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    ConversationItemAddedEvent,
    ErrorEvent,
    JobContext,
    TurnHandlingOptions,
    UserInputTranscribedEvent,
    UserTranscriptionTimeoutEvent,
    cli,
    inference,
    llm,
)
from livekit.agents.llm import ChatMessage
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.agent_session import RecordingOptions

from collector.agent import NegotiationAgent
from collector.audit.store import DEFAULT_DB_PATH, AuditStore
from collector.guardrails.rings import PreCallContext
from collector.llm.anthropic_client import AnthropicClient
from collector.llm.base import LLMClient
from collector.policy import PolicyConfig
from collector.tracing import configure_tracing, flush_traces

load_dotenv()

logger = logging.getLogger("collector-voice")

_DEFAULT_CONSUMER_NAME = "Dana Whitfield"
_DEFAULT_ACCOUNT_REF = "ACCT-4471"

_TRANSIENT_ERROR_APOLOGY = (
    "Sorry, I had trouble hearing that for a second — could you say that again?"
)

# Speech runs through LiveKit Inference rather than the Deepgram and Cartesia
# plugins: same two models, reached with the LiveKit credentials the worker
# already holds instead of two more third-party keys, and zero data retention
# by default — the same reason `_recording_options()` leaves Cloud recording off.
#
# Every value below is the plugin default this replaced, pinned explicitly.
# `cartesia.TTS()` in particular chose the voice implicitly; the agent's
# audible identity is not something to leave to a library default that can
# change under it between releases.
STT_MODEL = "deepgram/nova-3"
STT_LANGUAGE = "en-US"

# Keyterm prompting, Nova-3 only (`inference/stt.py:_keyterms_extra_for_model`
# maps any `deepgram/` model onto the `keyterm` extra; the ceiling is 100 terms
# / 1200 characters). Every term here earns its place by being (a) a word a
# code path actually keys on and (b) uncommon enough that telephony audio gets
# it wrong. Ordinary words the model already handles — "weekly", "monthly",
# "payment" — are deliberately absent: boosting the common case buys nothing
# and dilutes the terms that need it.
#
# Deliberately NOT here: any dollar amount. Biasing the recognizer toward the
# figures this agent is authorized to offer would bias transcription of what
# the *consumer* said toward those same figures — manufacturing agreement to a
# number they never spoke. The one place a thumb on the scale would be
# indefensible.
STT_KEYTERMS = [
    # Cadence. The engine keys on these; "biweekly" is routinely heard as
    # "by weekly" or "bi-weekly", and a wrong cadence is a wrong schedule.
    "biweekly",
    "semimonthly",
    # Escalation triggers. A missed one of these does not degrade the call, it
    # ends the wrong way: the consumer asserted a right and the agent kept
    # negotiating. Each is verified to fire `detect_escalation`.
    "cease and desist",
    "identity theft",
    "validation",
    "retained",
    "attorney",
    "bankruptcy",
    "garnishment",
    "chemo",
    "terminally",
    "suicidal",
    "disability",
    # Domain vocabulary the negotiation runs on.
    "settlement",
    "delinquent",
    "installment",
    "downpayment",
]

TTS_MODEL = "cartesia/sonic-3"
TTS_VOICE = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
TTS_LANGUAGE = "en"

# A salutation that runs straight into the introduction reads as a recording.
# A person says "Hi," waits for the other end to register that someone is
# there, and only then says who they are. The beat also buys the consumer's
# audio path a moment to settle before the first substantive words — the ones
# carrying the AI disclosure, which they have to actually hear.
#
# Real dead air between two `say()` calls, not markup: Cartesia's sonic-3
# exposes speed/volume/emotion and IPA pronunciation overrides, and does not
# document SSML `<break>` support (the SSML table in LiveKit's audio-
# customization page is a generic provider-varies list, not a Cartesia one).
# An unsupported tag would be read aloud or silently dropped, and either way
# it would put characters into a line the guardrails already cleared verbatim.
GREETING_PAUSE_SECONDS = 0.7

# Preemptive generation is on by the SDK's own default, and it is wrong for
# this agent specifically. It calls `Agent.llm_node` — and so
# `NegotiationAgent.stream_turn()` — on transcripts that are not yet a
# committed turn, up to `max_retries` times per consumer utterance, discarding
# all but at most one. Verified in the pinned livekit-agents 1.6.8:
# `_pipeline_reply_task_impl` runs `perform_llm_inference(node=llm_node, ...)`
# roughly a hundred lines before it awaits `speech_handle._wait_for_scheduled()`,
# and the preflight path reaches it whenever `_vad_base_turn_detection` holds
# — which it does for the default `inference.TurnDetector()`.
#
# For an agent whose node is a pure text->text call, a discarded generation
# costs only tokens. `stream_turn()` is not that: it advances `_turn_index`,
# records a ConsumerUtterance, and can move the concession ladder through
# `concede` or `record_refusal`. Streaming narrows the window but does not
# close it — a cancelled generation stops the pump within one sentence, and
# the guardrail state, the ladder, and the audit rows for the sentences it did
# release all stand. "I can do two fifty" finalizes, the ladder moves,
# "...a month" arrives, the generation is invalidated, and the ladder moves
# again: one utterance the consumer spoke once, two rows in the audit log and
# two concessions off a ladder that is only ever supposed to move on a refusal
# on record.
#
# `preemptive_tts` is off by default and must stay off for the same reason
# one step further along — it would synthesize audio for a reply derived from
# a sentence the consumer had not finished saying.
TURN_HANDLING: TurnHandlingOptions = {"preemptive_generation": {"enabled": False}}

# Only a leading salutation, and only when something follows it. Anything else
# is spoken whole — this splits delivery, it never rewords the line.
_SALUTATION_RE = re.compile(
    r"^\s*(?:(?:hi|hello|hey)(?:\s+there)?|good\s+(?:morning|afternoon|evening))\b[\s,.!—-]*",
    re.IGNORECASE,
)


def _split_salutation(opening: str) -> tuple[str, str] | None:
    """Split a leading salutation off the opening line so the two halves can be
    spoken with a beat between them. ``None`` when the line does not open with
    one, or opens with nothing else — the caller then speaks it unchanged.

    The halves carry the same words, in the same order, and only those words.
    That is the invariant that matters: the audit record for this line was
    written by ``open_call()`` against the full string, and what the outbound
    guardrails cleared was its *content* — the figures, the claims, the
    disclosure. Neither adjustment below can touch any of those. The salutation
    is a closed set of greeting words, so it holds no digits to alter, and
    upper-casing one leading letter cannot change a number either.

    What does get adjusted is punctuation and case, because each half is
    synthesized as its own utterance and has to stand as one. Left raw, "Hi,"
    gets the rising, unfinished intonation of a clause that continues —
    straight into a pause that says it doesn't — and the remainder opens
    lowercase, mid-clause. Sonic-3 reads punctuation and casing for prosody,
    so the trailing comma becomes a full stop and the remainder gets its
    sentence capital back.
    """
    match = _SALUTATION_RE.match(opening)
    if match is None:
        return None
    remainder = opening[match.end() :].strip()
    if not remainder:
        return None
    salutation = opening[: match.end()].strip().rstrip(" ,;:-—")
    if not salutation.endswith((".", "!", "?")):
        salutation += "."
    return salutation, remainder[0].upper() + remainder[1:]


async def _say_opening(session: AgentSession[None], opening: str) -> None:
    """Speak the opening line, with a beat after the salutation if it has one.

    ``allow_interruptions=False`` on both halves for the same reason it was on
    the single call this replaced: the opening carries the AI disclosure, and a
    consumer talking over it must not cost them the disclosure.
    """
    split = _split_salutation(opening)
    if split is None:
        await session.say(opening, allow_interruptions=False)
        return
    salutation, remainder = split
    await session.say(salutation, allow_interruptions=False)
    await asyncio.sleep(GREETING_PAUSE_SECONDS)
    await session.say(remainder, allow_interruptions=False)


def _retrieve_pump_exception(pump: asyncio.Task[None]) -> None:
    """Keep an unexpected pump failure from surfacing as an asyncio warning
    about an exception nobody read.

    ``_stream_sentences``' worker relays everything to its caller, so this
    should never find anything. The ``cancelled()`` guard is not cosmetic:
    ``Task.exception()`` re-raises ``CancelledError`` on a cancelled task, and
    the pump is routinely still pending when a worker's loop shuts down under
    it — the thread cannot be cancelled, only the task waiting on it.
    """
    if pump.cancelled():
        return
    exc = pump.exception()
    if exc is not None:
        logger.error("stream_turn pump failed", exc_info=exc)


class _ConsumerContextError(ValueError):
    """Dispatch metadata was present but could not be parsed into a consumer
    context. Raised rather than silently substituting the fixture consumer —
    a live call must never proceed under the wrong name or
    account just because its dispatch metadata was malformed."""


def _recording_options() -> bool | RecordingOptions:
    """What, if anything, this call uploads to LiveKit Cloud's observability
    store. ``COLLECTOR_VOICE_RECORDING``:

    * unset / ``off`` — **the default.** Nothing is uploaded.
    * ``diagnostics`` — pipeline traces and agent-server logs, no audio and no
      transcript. The granular middle ground: it
      buys the Agent-insights timeline for latency and failures without a
      third-party copy of the consumer's words.
    * ``full`` — everything, i.e. the SDK's own default. Local debugging in
      the Agent Console only.

    The default does not move: this project's ``AuditStore`` is the
    compliance record, and nothing here collects consent for a second copy of
    a consumer's speech held by anyone else. ``diagnostics`` is narrower than
    a full upload, but it is still a
    product decision to make deliberately, per call deployment — not a
    default to drift into.
    """
    mode = os.environ.get("COLLECTOR_VOICE_RECORDING", "off").lower()
    if mode == "full":
        return True
    if mode == "diagnostics":
        return {"audio": False, "transcript": False, "traces": True, "logs": True}
    if mode not in ("off", ""):
        logger.warning("unknown_recording_mode", extra={"mode": mode})
    return False


def _llm_route() -> str:
    """The name ``_llm_client()`` will dispatch on, for the log context — so a
    line in a drain says which model actually answered the call."""
    return os.environ.get("COLLECTOR_LLM") or "anthropic"


def _llm_client() -> LLMClient:
    """Anthropic by default; ``COLLECTOR_LLM`` routes the same calls elsewhere —
    ``openrouter`` for the backend ``text_app.py`` reaches with ``--openrouter``,
    ``livekit`` for Gemini 3 Flash on LiveKit Inference (``--livekit``). An env
    var rather than a CLI flag because the worker's own argv is consumed
    entirely by ``cli.run_app``'s dev/start subcommands (and, when launched via
    ``lk agent dev``, by the CLI wrapper itself).

    Neither alternate route is certified: ``MAX_TOOL_ROUNDS`` and the
    regeneration-strike budget were tuned against Claude, so a route change
    needs ``tests/evals/`` and the ADVERSARIAL_TESTING pass re-run before it
    carries a real call."""
    route = os.environ.get("COLLECTOR_LLM")
    if route == "openrouter":
        from collector.llm.openrouter_client import OpenRouterClient

        return OpenRouterClient()
    if route == "livekit":
        from collector.llm.livekit_client import LiveKitInferenceClient

        return LiveKitInferenceClient()
    return AnthropicClient()


class _NoOpLLM(llm.LLM[Never]):
    """Satisfies the framework's requirement that an ``Agent`` carry an LLM.
    Never actually invoked: ``llm_node`` below replaces the entire reasoning
    step, so this exists only to fill the constructor slot."""

    def chat(self, *args: object, **kwargs: object) -> NoReturn:
        raise NotImplementedError("reasoning happens in NegotiationAgent, not here")


class CollectorAgent(Agent):
    """The phone-call transport. All negotiation, guardrail, and disclosure
    logic lives in ``NegotiationAgent`` and runs exactly as it does in
    ``text_app.py``; this class only moves text in and out of it."""

    def __init__(self, negotiation_agent: NegotiationAgent, *, audio: bool = True) -> None:
        """``audio=False`` attaches no STT or TTS. LiveKit's test framework
        drives a session in text mode — ``AgentSession.run()`` feeds a
        transcript straight in and reads the reply back out — so the audio
        pipeline is not merely unused there, it is harmful: the STT pump opens
        a recognition stream the moment the session starts, which in a test
        means either a crash (no job context) or a live network call. Real
        calls take the default and are unaffected."""
        super().__init__(
            instructions=(
                "Reasoning is delegated entirely to a deterministic engine outside "
                "this class; this agent only carries audio."
            ),
            stt=inference.STT(
                model=STT_MODEL,
                language=STT_LANGUAGE,
                extra_kwargs={"keyterm": STT_KEYTERMS},
            )
            if audio
            else None,
            llm=_NoOpLLM(),
            tts=inference.TTS(model=TTS_MODEL, voice=TTS_VOICE, language=TTS_LANGUAGE)
            if audio
            else None,
        )
        self._negotiation_agent = negotiation_agent
        # The in-flight ``_stream_sentences`` pump, so ``drain()`` can wait for
        # an abandoned turn's audit write before the store closes.
        self._pump: asyncio.Task[None] | None = None

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        if self._negotiation_agent.ended:
            # Normally unreachable — the turn that ends the call shuts the
            # session down itself. Not always: a barge-in can tear that turn
            # down after a tool closed the call but before this node returned,
            # and then nothing has asked the room to close. Without the
            # shutdown here the agent is mute for the rest of a call that
            # never hangs up.
            self._end_the_call()
            return

        user_msg = chat_ctx.items[-1] if chat_ctx.items else None
        if not isinstance(user_msg, llm.ChatMessage) or user_msg.role != "user":
            logger.warning("llm_node invoked with no user turn on top of the context")
            return
        utterance = user_msg.text_content
        if not utterance:
            return

        spoke = False
        try:
            async for sentence in self._stream_sentences(utterance):
                if not sentence:
                    # ``stream_turn`` yields its escalation line through
                    # ``spoken or ""``; an empty chunk is not speech, and the
                    # framework drops it downstream anyway.
                    continue
                # The framework concatenates chunks *raw* into the assistant
                # message and into what TTS receives, so the separator is this
                # node's to supply: without it the history reads
                # "That works.Let me confirm" and the synthesizer runs two
                # sentences together. A single space is the same join
                # ``stream_turn`` uses for ``AgentTurn.spoken`` and
                # ``_transcribe`` uses for the model's own transcript, so all
                # three agree on the text character for character.
                chunk = sentence if not spoke else f" {sentence}"
                spoke = True
                yield chunk
        except Exception:
            # A transient failure here (rate limit, network blip) must not
            # silently drop the call — stream_turn already recorded the
            # consumer's utterance and updated guardrail state before any
            # point it could raise, so the call is safe to continue on the
            # next turn.
            if spoke:
                # Mid-stream the apology's own premise is already false:
                # earlier sentences are audio, and a scripted "could you say
                # that again?" after real speech reads as a non-sequitur
                # rather than a recovery. Same rule ``_Round.needs_fallback``
                # applies one level down. The turn keeps what it did say.
                logger.exception("stream_turn raised mid-turn; leaving the line where it is")
            else:
                logger.exception("stream_turn raised; apologizing and continuing the call")
                # The apology is about to be spoken, so it has to be in the
                # record before it is. Off the event loop for the same reason
                # the turn itself is: AuditStore.record() blocks on the
                # store's own worker thread.
                await asyncio.to_thread(
                    self._negotiation_agent.record_fallback_speech, _TRANSIENT_ERROR_APOLOGY
                )
                yield _TRANSIENT_ERROR_APOLOGY
        finally:
            # In a ``finally`` rather than after the loop, because the turn
            # that ends the call is exactly the turn a consumer is most likely
            # to talk over — "we're all set, you'll get an email" — and a
            # barge-in tears this generator down at the ``yield``. The check
            # has to run on that path too, or the room stays open with an
            # agent that will not speak again.
            if self._negotiation_agent.ended:
                self._end_the_call()

    def _end_the_call(self) -> None:
        """``drain=True`` (the default) waits for the lines already yielded to
        finish playing before the room actually closes — confirmed against
        ``AgentActivity.drain()``, which processes the entire queued-speech
        backlog, not just what's already in flight. ``NegotiationAgent.close()``
        itself runs once, from the job's shutdown callback below — the only
        path guaranteed to fire whether the call ends here or the room just
        disconnects.

        ``Agent.session`` raises when the agent is not attached to a running
        activity — a closing turn driven directly, with no room to close. That
        is a no-op here, not a failure: this method's whole job is to hang up a
        call, and there is no call."""
        try:
            session = self.session
        except RuntimeError:
            logger.debug("no running session to shut down; the call is already over")
            return
        session.shutdown()

    async def _stream_sentences(self, utterance: str) -> AsyncIterator[str]:
        """``NegotiationAgent.stream_turn`` — a *synchronous* generator making
        blocking model calls — driven as an async one, so each guarded sentence
        reaches TTS the moment it clears instead of after the whole turn
        completes. That is the entire point of the streaming path, and the
        reason this replaced a ``to_thread(turn, ...)``: TTS starts on sentence
        one while the model is still writing sentence two, and the framework's
        ``llm_node_ttft`` becomes a real time-to-first-token again.

        A single pump thread drives the generator and hands sentences over
        through the loop's own queue. Two properties are why it is a pump
        rather than one ``to_thread(next, ...)`` per sentence:

        * **The generator is closed by the thread that owns it.** A cancelled
          ``to_thread`` leaves its thread running — the interpreter cannot
          interrupt a blocking read — so a per-``next()`` design abandons the
          generator suspended at a ``yield``, and ``stream_turn``'s ``finally``,
          which writes the turn to the audit store, then runs whenever the
          garbage collector reaches it, on whatever thread that happens to be.
          Here the pump owns the generator start to finish and ``closing()``
          runs that ``finally`` deterministically, on one thread.
        * **Abandonment is cooperative and bounded.** A barge-in sets
          ``abandoned``; the pump, mid-``next()``, finishes that one sentence,
          sees the flag, and closes. So exactly one sentence past the
          interruption is guarded and written as spoken without reaching TTS.
          That over-record cannot be eliminated — the guard fires when a
          sentence is *released* and the generator is by design a step ahead of
          the ear — but it is bounded at one, where the ``turn()`` path this
          replaces recorded a whole turn the consumer heard only part of.
        """
        loop = asyncio.get_running_loop()
        # Unbounded: a spoken turn is a handful of sentences, and a bounded
        # queue would park the pump inside a ``put`` where the abandonment
        # check below cannot reach it.
        handoff: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
        abandoned = threading.Event()
        stream = self._negotiation_agent.stream_turn(utterance)

        def hand_over(item: str | BaseException | None) -> None:
            try:
                loop.call_soon_threadsafe(handoff.put_nowait, item)
            except RuntimeError:
                # The loop is gone: the worker is shutting down mid-turn.
                # Nobody is left to deliver to, so stop generating — the
                # ``closing()`` below still writes the turn.
                abandoned.set()

        def drive() -> None:
            try:
                with contextlib.closing(stream):
                    for sentence in stream:
                        hand_over(sentence)
                        if abandoned.is_set():
                            break
            except BaseException as exc:  # noqa: BLE001 — relayed, not swallowed
                hand_over(exc)
            else:
                hand_over(None)

        pump = asyncio.create_task(asyncio.to_thread(drive))
        self._pump = pump
        pump.add_done_callback(_retrieve_pump_exception)
        try:
            while True:
                item = await handoff.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            abandoned.set()

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for an abandoned ``stream_turn`` to finish unwinding.

        A barge-in leaves the pump thread mid-``next()``. It finishes that
        sentence, then closes the generator — and closing it is not free. Two
        things the shutdown callback depends on happen during that unwind, on
        the pump's thread:

        * ``_stream_round``'s own ``finally`` writes the aborted ``ModelCalled``
          row for the round it cut short. That is a store write, and
          ``store.close()`` is a few lines away in ``_finalize_call``.
        * ``stream_turn``'s ``finally`` appends the ``AgentTurn``, which is what
          ``NegotiationAgent.close()`` counts into ``CallEnded.turn_count``.
          Draining after that count is taken would leave the record claiming
          one turn fewer than the consumer had.

        So this is ordered before both, and the audit chain keeps the single
        writer it is built on.

        Bounded rather than unbounded: a wedged model call must not hold the
        room open, and one lost turn row is a smaller loss than a call that
        never hangs up.
        """
        pump = self._pump
        if pump is None or pump.done():
            return
        # ``wait``, not ``wait_for``: a timeout here must not cancel the task,
        # because cancelling it would not stop the thread underneath and would
        # only hide the pump's own exception.
        _, pending = await asyncio.wait({pump}, timeout=timeout)
        if pending:
            logger.warning("stream_turn did not finish writing before shutdown")


def _consumer_context(ctx: JobContext) -> tuple[str, str]:
    """Consumer name and account ref for this call, from job metadata if the
    dispatcher supplied it, the same defaults ``text_app.py`` uses otherwise.

    No metadata at all (manual or local testing) falls
    back to the fixture consumer — that case is legitimate and expected.
    Metadata that IS present but malformed raises instead: silently
    substituting a different real consumer's identity on a live call would
    undermine the identity-confirmation guardrail rather than merely fail to
    help it."""
    import json

    metadata = ctx.job.metadata
    if not metadata:
        return _DEFAULT_CONSUMER_NAME, _DEFAULT_ACCOUNT_REF
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise _ConsumerContextError(f"dispatch metadata is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise _ConsumerContextError("dispatch metadata JSON must be an object")
    name = parsed.get("consumer_name")
    account = parsed.get("account_ref")
    if not isinstance(name, str) or not name:
        raise _ConsumerContextError("dispatch metadata missing a non-empty 'consumer_name'")
    if not isinstance(account, str) or not account:
        raise _ConsumerContextError("dispatch metadata missing a non-empty 'account_ref'")
    return name, account


def _log_turn_latency(session: AgentSession[None]) -> None:
    """Log what a turn actually cost the consumer in silence.

    ``NegotiationAgent.stream_turn()`` can stack five sequential model round
    trips into one spoken reply — ``MAX_TOOL_ROUNDS`` (4) plus the final ask
    for words. ``agent.py``'s per-call ``elapsed_ms`` line shows each leg;
    nothing showed what the consumer experienced, which is the number a phone
    call is judged on.

    All three figures come from the framework and none needs the LiveKit Cloud
    recording this project leaves off:

    * ``e2e_latency`` — consumer stopped speaking to agent audio playing.
    * ``llm_node_ttft`` — time to the *first guarded sentence*, which since
      ``llm_node`` streams is a genuine time-to-first-token and not the whole
      turn. It is the number the streaming path exists to move; the rest of the
      turn generates while that first sentence is already playing, so
      ``turn_ms`` is now well under ``e2e_ms`` rather than nearly all of it.
    * ``tts_node_ttfb`` — the synthesis leg, which now starts on sentence one
      rather than waiting for the last.
    """

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        if not isinstance(ev.item, ChatMessage) or ev.item.role != "assistant":
            return
        metrics = ev.item.metrics
        logger.info(
            "turn_latency",
            extra={
                "e2e_ms": _ms(metrics.get("e2e_latency")),
                "turn_ms": _ms(metrics.get("llm_node_ttft")),
                "tts_ttfb_ms": _ms(metrics.get("tts_node_ttfb")),
            },
        )


def _ms(seconds: float | None) -> int | None:
    """The SDK reports seconds; the agent loop's own latency line is in
    milliseconds, so match it rather than making a reader convert.

    A number, not a formatted string: these go out as `extra=` fields, which
    the framework's JSON formatter emits as real JSON values, so a drain can
    filter on `e2e_ms > 2000` instead of parsing it back out of a sentence.
    `None` stays `None` (`null`) rather than becoming `"n/a"`, for the same
    reason — a missing measurement should not read as a value.
    """
    return None if seconds is None else round(seconds * 1000)


def _log_session_events(session: AgentSession[None]) -> None:
    """The parts of a call that ``turn()`` cannot see.

    ``llm_node``'s own ``try``/``except`` covers the reasoning step, and
    nothing else. Everything around it — STT never returning a transcript,
    TTS failing mid-sentence, the consumer's audio dropping — was invisible
    to this codebase: the framework logs its own view of those, but nothing
    tied them to the call. These handlers are the missing half.

    Deliberately metadata-only. ``user_input_transcribed`` carries the
    consumer's words; the character count and finality are what diagnose a
    stuck STT stream, and the words themselves are already in the audit store
    under a consent posture the log does not share.
    """

    @session.on("error")
    def _on_error(ev: ErrorEvent) -> None:
        # `error` fires for STT/TTS/LLM failures the session catches itself.
        # `recoverable` is the framework's own judgement of whether the call
        # can continue, so it decides the level: an unrecoverable TTS error
        # means the consumer is listening to silence.
        recoverable = getattr(ev.error, "recoverable", True)
        logger.log(
            logging.WARNING if recoverable else logging.ERROR,
            "session_error",
            extra={
                "source": type(ev.source).__name__,
                "error": type(ev.error).__name__,
                "recoverable": recoverable,
            },
        )

    @session.on("user_transcription_timeout")
    def _on_transcription_timeout(ev: UserTranscriptionTimeoutEvent) -> None:
        # The consumer spoke and STT returned nothing. Without this, the call
        # simply appears to go quiet from the agent's side with no cause.
        logger.warning(
            "transcription_timeout",
            extra={"speech_duration_ms": round(ev.speech_duration * 1000)},
        )

    @session.on("agent_false_interruption")
    def _on_false_interruption(ev: AgentFalseInterruptionEvent) -> None:
        # Background noise cut the agent off mid-sentence. Fires by default:
        # `false_interruption_timeout` is 2.0s in `_INTERRUPTION_DEFAULTS`,
        # and `resume_false_interruption` is True, so `resumed` records
        # whether the consumer actually heard the rest of the line.
        logger.info("false_interruption", extra={"resumed": ev.resumed})

    @session.on("user_input_transcribed")
    def _on_transcribed(ev: UserInputTranscribedEvent) -> None:
        if not ev.is_final:
            return
        logger.info("user_transcript", extra={"chars": len(ev.transcript)})


def _log_session_usage(session: AgentSession[None]) -> None:
    """One line per model at the end of the call: how much speech each one
    processed. ``session.usage`` is a local rollup the SDK keeps regardless of
    the recording mode, so this works with the default ``record=False``.

    It covers STT and TTS through LiveKit Inference. It does **not** cover the
    negotiation model: ``llm_node`` bypasses the framework's LLM entirely, so
    Anthropic's tokens never pass through the session's collector. This is a
    speech-pipeline usage line, not the call's cost.
    """
    # `model_dump()` rather than a hand-picked field list: the five usage
    # variants carry different counters (STT has `audio_duration`, the
    # interruption detector only `total_requests`), and each dump already
    # names its own shape through a `type` discriminator.
    #
    # Wrapped because the dict is the SDK's shape, not ours: `logging` raises
    # KeyError if an `extra` key shadows a LogRecord attribute (`name`,
    # `module`, `msg`, ...). None of the five variants collide today —
    # checked against `LogRecord.__dict__` on 1.6.8 — but this runs on the
    # shutdown path, and a usage line is not worth risking the callback that
    # writes the agreement.
    try:
        for usage in session.usage.model_usage:
            logger.info("model_usage", extra=usage.model_dump())
    except Exception:
        logger.exception("could not log session usage")


server = AgentServer()

# The dispatch name, not a label. Setting it moves this worker off automatic
# dispatch — it no longer joins every room in the project — onto explicit
# dispatch, where a job names it. That is the mode this file was already
# written for: `_consumer_context()` reads whose account is being called from
# `ctx.job.metadata`, and automatic dispatch cannot carry metadata at all, so
# under it every call would have fallen through to the fixture consumer. It is
# also what a collections agent has to be: one that auto-joins any room that
# happens to open can place a call nobody authorized.
AGENT_NAME = "collections-negotiator"


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    # Before anything connects, and before a consumer is on the line: a
    # malformed COLLECTOR_TRACING raises here and the job never dials. Off
    # unless the operator asked for it, idempotent across the jobs one worker
    # handles, and it points the SDK's own session spans at the same backend
    # (docs.livekit.io, "Export traces"). See tracing.py for what does and does
    # not reach a span.
    configure_tracing()

    # These fields are stamped onto every log record emitted for this job,
    # including `collector.agent`'s: the SDK installs its context filter on
    # the *root* handlers (`job.py:_on_setup`), so it reaches this project's
    # own loggers, not just `livekit.*`. That is what makes `call_id` a
    # correlator — one call's lines can be pulled out of a worker handling
    # several at once, across both modules.
    #
    # `call_id` and `account_ref` identify the call; the consumer's *name* is
    # deliberately absent. It buys nothing a reader of the audit store cannot
    # get, and logs leave the process by a route (stdout, a log drain) that
    # carries none of the consent posture that governs their speech.
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "call_id": ctx.job.id,
        "channel": "voice",
        "llm_route": _llm_route(),
    }
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    try:
        consumer_name, account_ref = _consumer_context(ctx)
    except _ConsumerContextError as exc:
        logger.error("refusing to place call: %s", exc)
        ctx.shutdown(f"invalid dispatch metadata: {exc}")
        return
    ctx.log_context_fields["account_ref"] = account_ref

    store = AuditStore(DEFAULT_DB_PATH, json_dir=DEFAULT_DB_PATH.parent)
    negotiation_agent = NegotiationAgent(
        llm=_llm_client(),
        policy=PolicyConfig.default(),
        call_id=ctx.job.id,
        consumer_name=consumer_name,
        account_ref=account_ref,
        store=store,
        channel="voice",
    )

    check, opening = await asyncio.to_thread(
        negotiation_agent.open_call,
        PreCallContext(account_loaded=True, within_calling_window=True),
    )
    if not check.allowed or opening is None:
        # open_call() already recorded CallStarted, the violations, and a
        # CallEnded(ABANDONED) itself when it refused to place the call — the
        # same convention text_app.py follows. NegotiationAgent.close() must
        # not run here too, or the audit trail gets a duplicate CallEnded.
        logger.warning(
            "precall_blocked",
            extra={"rules": [v.rule_id for v in check.violations]},
        )
        store.close()
        ctx.shutdown("pre-call guardrail blocked")
        return

    # Both constructed here rather than just before `start()` so
    # `_finalize_call` can close over them; neither constructor opens anything.
    session: AgentSession[None] = AgentSession(turn_handling=TURN_HANDLING)
    collector = CollectorAgent(negotiation_agent)

    async def _finalize_call() -> None:
        # The one call site for NegotiationAgent.close(), reached whether the
        # call ends through our own end_call/escalation path (which calls
        # session.shutdown() and lets the job wind down into this callback)
        # or the room just disconnects out from under us. A phone call that
        # ends without this running writes no agreement — the brief's graded
        # deliverable — so it must not depend on how the call ended.
        #
        # The drain comes first: a turn the consumer talked over is still
        # writing its record on a pump thread, and that write has to land
        # before close() stamps the turn count and store.close() shuts the
        # audit writer down.
        await collector.drain()
        await asyncio.to_thread(negotiation_agent.close)
        store.close()
        # Last, deliberately: the agreement and the closed store are the
        # graded deliverable, and a usage line is the least important thing
        # this callback does. Nothing after this point can cost the call its
        # record.
        _log_session_usage(session)
        # After close(), so the CallEnded that ends the call's root span is in
        # the batch this drains. BatchSpanProcessor holds spans that process
        # exit would otherwise drop.
        await asyncio.to_thread(flush_traces)

    ctx.add_shutdown_callback(_finalize_call)

    _log_turn_latency(session)
    _log_session_events(session)
    # Explicit, not the SDK default: omitting `record` uploads audio,
    # transcripts, traces, and logs to LiveKit Cloud's observability store
    # for every call. See `_recording_options()` for the three modes and why
    # the default stays off.
    await session.start(agent=collector, room=ctx.room, record=_recording_options())
    await _say_opening(session, opening)


def run() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    run()
