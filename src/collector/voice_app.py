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
utterance straight to ``NegotiationAgent.turn()``, which already resolves
tool calls, guardrails, and disclosures before a word reaches here. What
this file adds is only the plumbing to get a sentence in and a sentence out.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import AsyncIterable
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

from collector.agent import AgentTurn, NegotiationAgent
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
# `NegotiationAgent.turn()` — on transcripts that are not yet a committed
# turn, up to `max_retries` times per consumer utterance, discarding all but
# at most one. Verified in the pinned livekit-agents 1.6.8:
# `_pipeline_reply_task_impl` runs `perform_llm_inference(node=llm_node, ...)`
# roughly a hundred lines before it awaits `speech_handle._wait_for_scheduled()`,
# and the preflight path reaches it whenever `_vad_base_turn_detection` holds
# — which it does for the default `inference.TurnDetector()`.
#
# For an agent whose node is a pure text->text call, a discarded generation
# costs only tokens. `turn()` is not that: it advances `_turn_index`, records
# a ConsumerUtterance, and can move the concession ladder through `concede`
# or `record_refusal`. Worse, `llm_node` runs it under `asyncio.to_thread`,
# and a thread cannot be cancelled — the SDK cancels the SpeechHandle while
# `turn()` runs to completion regardless. "I can do two fifty" finalizes,
# the ladder moves, "...a month" arrives, the generation is invalidated, and
# the ladder moves again: one utterance the consumer spoke once, two rows in
# the audit log and two concessions off a ladder that is only ever supposed
# to move on a refusal on record.
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


class _ConsumerContextError(ValueError):
    """Dispatch metadata was present but could not be parsed into a consumer
    context. Raised rather than silently substituting the fixture consumer —
    a live call must never proceed under the wrong name or
    account just because its dispatch metadata was malformed."""


def _recording_options() -> bool | RecordingOptions:
    """What, if anything, this call uploads to LiveKit Cloud's observability
    store. ``COLLECTOR_VOICE_RECORDING``:

    * unset / ``off`` — **the default.** Nothing is uploaded.
    * ``diagnostics`` — no audio track and no session-report transcript, but
      **it does upload the consumer's words.** See the warning below.
    * ``full`` — everything, i.e. the SDK's own default. Local debugging in
      the Agent Console only.

    ``diagnostics`` is not the privacy middle ground its name suggests, and
    this docstring claimed for a while that it was. ``transcript: False`` gates
    only the end-of-session chat-history report (``telemetry/traces.py``); it
    does not touch the live span pipeline. Under ``traces: True`` the SDK sets
    ``lk.user_transcript`` to the consumer's verbatim utterance on every
    user-turn span (``voice/audio_recognition.py``) and ``lk.chat_ctx`` to the
    entire serialized conversation on every LLM-node span
    (``voice/generation.py``) — neither consults the recording options. Our
    ``llm_node`` override is called *by* that wrapper, so it is not exempt.

    So ``diagnostics`` buys the Agent-insights timeline at the price of a
    third-party verbatim copy of the call: materially the same consent decision
    already declined for audio, differing only in medium. ``logs: True`` adds a
    second route the SDK logs transcripts on at DEBUG, which production's INFO
    floor happens to filter and ``lk agent dev`` does not.

    The default does not move: this project's ``AuditStore`` is the compliance
    record, and nothing here collects consent for a second copy of a consumer's
    speech held by anyone else. If the goal is only latency data, note that
    ``turn_latency`` (``e2e_ms``/``turn_ms``/``tts_ttfb_ms``) is already logged
    locally and works with recording off.
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
    """Anthropic by default; ``COLLECTOR_LLM=openrouter`` routes the same calls
    to the backend ``text_app.py`` reaches with ``--openrouter``. An env var
    rather than a CLI flag because the worker's own argv is consumed entirely by
    ``cli.run_app``'s dev/start subcommands (and, when launched via ``lk agent
    dev``, by the CLI wrapper itself).

    The alternate route is not certified: ``MAX_TOOL_ROUNDS`` and the
    regeneration-strike budget were tuned against Claude, so a route change
    needs ``tests/evals/`` and the ADVERSARIAL_TESTING pass re-run before it
    carries a real call."""
    if os.environ.get("COLLECTOR_LLM") == "openrouter":
        from collector.llm.openrouter_client import OpenRouterClient

        return OpenRouterClient()
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

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        if self._negotiation_agent.ended:
            return

        user_msg = chat_ctx.items[-1] if chat_ctx.items else None
        if not isinstance(user_msg, llm.ChatMessage) or user_msg.role != "user":
            logger.warning("llm_node invoked with no user turn on top of the context")
            return
        utterance = user_msg.text_content
        if not utterance:
            return

        # NegotiationAgent.turn() is synchronous and makes up to several
        # blocking Anthropic calls (MAX_TOOL_ROUNDS round trips plus
        # regeneration strikes) — run it off the event loop so it doesn't
        # stall audio I/O, VAD, and STT streaming for the whole turn.
        try:
            turn: AgentTurn = await asyncio.to_thread(self._negotiation_agent.turn, utterance)
        except Exception:
            # A transient failure here (rate limit, network blip) must not
            # silently drop the call — turn() already recorded the consumer's
            # utterance and updated guardrail state before any point it could
            # raise, so the call is safe to continue on the next turn; it
            # just never produced a response to this one.
            logger.exception("negotiation_agent.turn() raised; apologizing and continuing the call")
            # The apology is about to be spoken, so it has to be in the record
            # before it is. Off the event loop for the same
            # reason turn() is: AuditStore.record() blocks on the store's own
            # worker thread.
            await asyncio.to_thread(
                self._negotiation_agent.record_fallback_speech, _TRANSIENT_ERROR_APOLOGY
            )
            yield _TRANSIENT_ERROR_APOLOGY
            return
        if turn.spoken:
            yield turn.spoken

        if self._negotiation_agent.ended:
            # ``drain=True`` (the default) waits for the line just yielded to
            # finish playing before the room actually closes — confirmed
            # against AgentActivity.drain(), which processes the entire
            # queued-speech backlog, not just what's already in flight.
            # NegotiationAgent.close() itself runs once, from the job's
            # shutdown callback below — the only path guaranteed to fire
            # whether the call ends here or the room just disconnects.
            self.session.shutdown()


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

    ``NegotiationAgent.turn()`` can stack seven sequential model round trips
    into one spoken reply — ``MAX_TOOL_ROUNDS`` (4), the final ask for words,
    and up to ``MAX_REGENERATION_STRIKES`` (2) rewrites behind the outbound
    guard. ``agent.py``'s per-call ``elapsed_ms`` line shows each leg; nothing
    showed the sum, which is the number a phone call is judged on.

    Both figures come from the framework and neither needs the LiveKit Cloud
    recording this project leaves off:

    * ``e2e_latency`` — consumer stopped speaking to agent audio playing.
    * ``llm_node_ttft`` — normally time-to-first-token, but ``llm_node`` here
      yields only after the whole blocking turn completes, so it *is* the
      total ``turn()`` latency, isolated from STT and TTS.
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

    # Constructed here rather than just before `start()` so `_finalize_call`
    # can close over it; the constructor opens nothing.
    session: AgentSession[None] = AgentSession(turn_handling=TURN_HANDLING)

    async def _finalize_call() -> None:
        # The one call site for NegotiationAgent.close(), reached whether the
        # call ends through our own end_call/escalation path (which calls
        # session.shutdown() and lets the job wind down into this callback)
        # or the room just disconnects out from under us. A phone call that
        # ends without this running writes no agreement — the brief's graded
        # deliverable — so it must not depend on how the call ended.
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
    await session.start(
        agent=CollectorAgent(negotiation_agent), room=ctx.room, record=_recording_options()
    )
    await _say_opening(session, opening)


def run() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    run()
