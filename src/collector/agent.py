"""The turn loop — SPEC §3, §5.2.

    perceive -> guard -> think -> tool -> guard -> speak

Every arrow is a checkpoint, and the two guard steps are not the same guard:
inbound reads the consumer for escalation triggers and never blocks them,
outbound holds a generated sentence back from TTS until it clears prohibited
persuasion, the numeric authorization set, and the disclosure state machine.

This class is the one mutable object in the system. A call is a sequence, not a
value, and pretending otherwise would mean threading six pieces of state
through every method. What it mutates is only ever a reference to the next
frozen state the layers below returned, so the history is still intact and a
call still replays exactly.

Deliberately absent: any path by which the model can escalate, end the call
compliantly, confirm identity, or author a figure. Those are code decisions and
they stay code decisions.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from collector.audit.events import (
    CallEnded,
    CallStarted,
    ConsumerConfirmation,
    DecisionRecorded,
    Escalated,
    GuardrailAction,
    GuardrailTripped,
    ModelCalled,
    Speaker,
    ToolInvoked,
    TurnRecorded,
    dumps,
)
from collector.audit.store import AuditStore
from collector.decision_engine import Verdict
from collector.guardrails.disclosures import confirms_identity
from collector.guardrails.numeric import AuthorizedFigures, authorized_for
from collector.guardrails.rings import (
    MAX_REGENERATION_STRIKES,
    CallSummary,
    EscalationRecord,
    GuardrailRing,
    GuardrailState,
    InboundCheck,
    PreCallCheck,
    PreCallContext,
    check_inbound,
    check_outbound,
    check_pre_call,
    fallback_for,
    finalize_call,
)
from collector.llm.base import (
    LLMClient,
    LLMResponse,
    LLMUsage,
    Message,
    StreamCompleted,
    TextDelta,
    ToolCall,
    stream_response,
    system_prompt,
)
from collector.negotiation import CallOutcome
from collector.offers import Offer
from collector.policy import PolicyConfig
from collector.tools import ToolContext, ToolResult, execute

# A turn that has made this many engine round trips is looping, not thinking.
MAX_TOOL_ROUNDS = 4

# A sentence ends at terminal punctuation followed by whitespace. The trailing
# whitespace is what keeps "$250.00 a month" and "3 p.m." in one piece: a
# decimal point has a digit after it, not a space. Good enough for speech, which
# is what this splits — an abbreviation mid-sentence costs one early TTS flush,
# not a compliance failure, because every fragment is guarded either way.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s)")


def _loggable(arguments: dict[str, object]) -> dict[str, object]:
    """The model's raw arguments, made safe to write to the audit log.

    ``to_jsonable`` raises on a float, deliberately — a float in a payment
    schedule is a compliance defect (SPEC §9). But JSON has one number type and
    it decodes to float, so the model saying ``total: 500.5`` is *ordinary* and
    ``_parse_money`` exists to absorb it. Logging the raw arguments without
    this would mean the tool succeeded and then recording that success killed
    the call — instrumentation defeating the tolerance it was added to observe.

    Floats become their exact decimal string, the same route ``_parse_money``
    takes, so the log keeps the digits the model actually emitted.
    """
    return {key: _loggable_value(value) for key, value in arguments.items()}


def _loggable_value(value: object) -> object:
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _loggable_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_loggable_value(v) for v in value]
    return str(value)


def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """Complete sentences in ``buffer``, and whatever is still being written.

    Called on every delta, so the first sentence reaches the guard — and from
    there TTS — while the model is still generating the second.
    """
    sentences: list[str] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(buffer):
        if match.start() < cursor:
            continue
        sentence = buffer[cursor : match.end()].strip()
        if sentence:
            sentences.append(sentence)
        cursor = match.end()
    return sentences, buffer[cursor:].lstrip()


@dataclass
class _Round:
    """What one streamed round produced, beside the sentences it yielded.

    A generator cannot hand a return value to a caller that is iterating it,
    and stashing this on the agent would make two concurrent turns share it.
    So the caller passes one of these in and reads it once the round is done.
    """

    response: LLMResponse | None = None
    blocked: str | None = None

    @property
    def failed(self) -> bool:
        return self.response is not None and self.response.error is not None

    def needs_fallback(self, spoken: list[str]) -> bool:
        """A blocked sentence always falls back; a failed call only if the
        round said nothing, since a scripted line after real speech reads as a
        non-sequitur rather than a recovery."""
        return self.blocked is not None or (self.failed and not spoken)


@dataclass(frozen=True)
class AgentTurn:
    """One exchange, from the consumer's words to what was actually spoken."""

    consumer: str
    spoken: str | None
    tool_results: tuple[ToolResult, ...] = ()
    blocked: tuple[str, ...] = ()
    escalated: bool = False
    ended: bool = False

    @property
    def was_regenerated(self) -> bool:
        return bool(self.blocked)


@dataclass(frozen=True)
class CallReport:
    """What the call amounted to, for the caller and for the log."""

    call_id: str
    outcome: CallOutcome
    summary: CallSummary
    agreed_offer: Offer | None
    turns: int

    @property
    def compliant(self) -> bool:
        return self.summary.compliant


@dataclass
class NegotiationAgent:
    llm: LLMClient
    policy: PolicyConfig = field(default_factory=PolicyConfig.default)
    call_id: str = "call-1"
    consumer_name: str = "the account holder"
    account_ref: str = "ACCT-0001"
    store: AuditStore | None = None
    channel: str = "text"

    def __post_init__(self) -> None:
        self.tools = ToolContext.opening(self.policy)
        self.authorized: AuthorizedFigures = authorized_for(self.policy)
        self.guard = GuardrailState.opening(self.policy, authorized=self.authorized)
        self.messages: list[Message] = [
            system_prompt(consumer_name=self.consumer_name, account_ref=self.account_ref)
        ]
        self.turns: list[AgentTurn] = []
        self.verdicts: list[Verdict] = []
        self.agreed_offer: Offer | None = None
        self.last_consumer_utterance: str = ""
        self.ended = False
        self._turn_index = 0

    # -- ring 1 ------------------------------------------------------------

    def open_call(self, context: PreCallContext | None = None) -> tuple[PreCallCheck, str | None]:
        """Run the pre-call ring and, if it clears, produce the opening line.

        Returns the check alongside the greeting so a refusal to dial is a
        value the caller handles, not an exception it has to catch.
        """
        check = self._dial(context)
        return check, None if not check.allowed else self._generate_and_speak()

    def _dial(self, context: PreCallContext | None) -> PreCallCheck:
        """The pre-call ring plus the ``CallStarted`` record, shared by both paths."""
        check = check_pre_call(
            context
            if context is not None
            else PreCallContext(account_loaded=True, within_calling_window=True)
        )
        if not check.allowed:
            self.ended = True
            self._record(
                CallStarted(
                    call_id=self.call_id,
                    account_ref=self.account_ref,
                    consumer_ref=self.consumer_name,
                    original_balance=self.policy.original_balance,
                    channel=self.channel,
                )
            )
            for violation in check.violations:
                self._record(
                    GuardrailTripped(
                        call_id=self.call_id,
                        turn_index=0,
                        ring=GuardrailRing.PRE_CALL,
                        rule_id=violation.rule_id,
                        action=GuardrailAction.BLOCKED,
                        detail=violation.detail,
                    )
                )
            self._record(CallEnded(self.call_id, CallOutcome.ABANDONED, turn_count=0))
            return check

        self._record(
            CallStarted(
                call_id=self.call_id,
                account_ref=self.account_ref,
                consumer_ref=self.consumer_name,
                original_balance=self.policy.original_balance,
                channel=self.channel,
            )
        )
        return check

    # -- ring 2 ------------------------------------------------------------

    def turn(self, consumer_utterance: str) -> AgentTurn:
        """One full exchange. The only method a transport needs to call."""
        if self.ended:
            raise ValueError("the call has ended; start a new one")

        inbound = self._perceive(consumer_utterance)
        if inbound.escalated and inbound.escalation is not None:
            return self._escalate(consumer_utterance, inbound.escalation)

        spoken, results, blocked = self._act()
        turn = AgentTurn(
            consumer=consumer_utterance,
            spoken=spoken,
            tool_results=results,
            blocked=blocked,
            ended=self.ended,
        )
        self.turns.append(turn)
        return turn

    def stream_turn(self, consumer_utterance: str) -> Iterator[str]:
        """One full exchange, emitted a sentence at a time — the voice path.

        Same rings, same engine, same whitelist as ``turn()``. What moves is
        where the guard sits: ``turn()`` guards a finished paragraph, this
        guards each sentence the moment it completes, so the first one is in
        the consumer's ear while the rest is still being written. That is the
        entire sub-500ms first-audio budget.

        One rule differs, and it is not negotiable. ``turn()`` can regenerate a
        blocked turn because nothing was spoken yet and the contract holds —
        "blocked, so the consumer did not hear it". Mid-stream that contract is
        already false: earlier sentences are audio. So a block here **aborts the
        stream and speaks the scripted fallback**. Retrying would mean
        contradicting something the consumer just heard, which is worse than the
        sentence that was blocked.

        The completed turn is appended to ``self.turns`` when the iterator is
        exhausted, so a caller that abandons it early leaves no turn recorded —
        which is correct: an abandoned stream is a dropped call, not a turn.

        **The greeting is not streamed, and cannot be.** ``open_call`` stays
        synchronous because the AI-disclosure rule is scoped to the *turn*: the
        first agent turn must disclose that the caller is an AI, and a first
        sentence that is not the disclosure cannot be judged against that rule
        without knowing what the second sentence will say. Guarding it whole is
        the only way to hold it. The Mini-Miranda rule does compose sentence by
        sentence — it constrains order within a turn, and a per-sentence guard
        enforces order by construction — so mid-call turns stream fine.

        A deployment chasing first-audio latency on the greeting should not
        reach for streaming here; it should pre-render it. The opening is the
        one utterance whose required content is fixed, so it does not need the
        model at all.
        """
        if self.ended:
            raise ValueError("the call has ended; start a new one")

        inbound = self._perceive(consumer_utterance)
        if inbound.escalated and inbound.escalation is not None:
            yield self._escalate(consumer_utterance, inbound.escalation).spoken or ""
            return

        spoken: list[str] = []
        blocked: list[str] = []
        results: list[ToolResult] = []
        # Set when a tool closed the call. One more round is still owed: the
        # consumer has to hear the arrangement read back, and a turn that ends
        # the call in silence is a dead line, not a goodbye.
        closing = False

        for _ in range(MAX_TOOL_ROUNDS + 1):
            round_ = _Round()
            for sentence in self._stream_round(round_):
                spoken.append(sentence)
                yield sentence

            if round_.blocked is not None:
                blocked.append(round_.blocked)
            if round_.needs_fallback(spoken):
                fallback = fallback_for(self.guard)
                self._speak_verbatim(fallback)
                spoken.append(fallback)
                yield fallback
                break

            response = round_.response
            if closing or response is None or round_.failed or not response.wants_tools:
                break
            for call in response.tool_calls:
                results.append(self._run_tool(call))
            closing = self.ended

        if spoken:
            # One assistant turn in the transcript, not one per sentence: the
            # model wrote a paragraph and the next round trip should see it that
            # way, whatever granularity TTS consumed it at.
            self.messages.append(Message(role="agent", content=" ".join(spoken)))

        self.turns.append(
            AgentTurn(
                consumer=consumer_utterance,
                spoken=" ".join(spoken) or None,
                tool_results=tuple(results),
                blocked=tuple(blocked),
                ended=self.ended,
            )
        )

    def _stream_round(self, round_: _Round) -> Iterator[str]:
        """One streamed round, yielding each sentence that clears the guard.

        Stops at the first blocked sentence — the abort the streaming contract
        requires. What the round produced besides speech (the assembled
        response, the blocked sentence) lands on ``round_``.
        """
        buffer = ""
        for event in stream_response(self.llm, tuple(self.messages)):
            match event:
                case TextDelta():
                    buffer += event.text
                    sentences, buffer = _split_sentences(buffer)
                    for sentence in sentences:
                        allowed = self._guard_sentence(sentence)
                        if allowed is None:
                            round_.blocked = sentence
                            return
                        yield allowed
                case StreamCompleted():
                    round_.response = event.response
                    self._record_model_call(event.response)

        # Whatever the model left without terminal punctuation is still a
        # sentence; a turn ending "so let me know" must not be swallowed.
        if buffer.strip():
            allowed = self._guard_sentence(buffer)
            if allowed is None:
                round_.blocked = buffer.strip()
            else:
                yield allowed

    def _guard_sentence(self, sentence: str) -> str | None:
        """The pre-TTS gate for one sentence. ``None`` means do not speak it.

        Records the trip as ``BLOCKED`` rather than ``REGENERATED``: on the
        streaming path there is no retry, and an audit log that says
        "regenerated" about a turn that was abandoned is a false record.
        """
        candidate = sentence.strip()
        if not candidate:
            return None

        check = check_outbound(self.guard, candidate, authorized=self.authorized)
        self.guard = check.state
        if check.allowed:
            self._record(
                TurnRecorded(
                    call_id=self.call_id,
                    turn_index=self._turn_index,
                    speaker=Speaker.AGENT,
                    text=candidate,
                )
            )
            return candidate

        for violation in check.blocking_violations:
            self._record(
                GuardrailTripped(
                    call_id=self.call_id,
                    turn_index=self._turn_index,
                    ring=GuardrailRing.DURING_CALL,
                    rule_id=violation.rule_id,
                    action=GuardrailAction.BLOCKED,
                    detail=violation.detail,
                    blocked_text=candidate,
                )
            )
        return None

    def _perceive(self, consumer_utterance: str) -> InboundCheck:
        """The inbound half of a turn, shared by the text and voice paths."""
        self.last_consumer_utterance = consumer_utterance
        self._turn_index += 1

        inbound = check_inbound(self.guard, consumer_utterance)
        self.guard = inbound.state
        self._record(
            TurnRecorded(
                call_id=self.call_id,
                turn_index=self._turn_index,
                speaker=Speaker.CONSUMER,
                text=consumer_utterance,
            )
        )
        self.messages.append(Message(role="consumer", content=consumer_utterance))

        # Identity is settled in code, before anything substantive can be said.
        if not self.guard.identity_confirmed and confirms_identity(consumer_utterance):
            self.guard = self.guard.with_identity_confirmed()
        return inbound

    def _escalate(self, consumer_utterance: str, escalation: EscalationRecord) -> AgentTurn:
        """A6: negotiation stops here. The closing line is code-authored, so
        there is no generated turn to guard and nothing left to negotiate."""
        trigger = escalation.trigger
        closing = escalation.closing_line
        detail = f"{trigger}: {consumer_utterance}"

        self._record(
            Escalated(
                call_id=self.call_id,
                turn_index=self._turn_index,
                trigger=trigger,
                detail=detail,
            )
        )
        self._record(
            GuardrailTripped(
                call_id=self.call_id,
                turn_index=self._turn_index,
                ring=GuardrailRing.DURING_CALL,
                rule_id=str(trigger),
                action=GuardrailAction.ESCALATED,
                detail=detail,
            )
        )
        self.tools = ToolContext(policy=self.policy, state=self.tools.state.escalate(str(trigger)))
        self._speak_verbatim(closing)
        self.ended = True

        turn = AgentTurn(consumer=consumer_utterance, spoken=closing, escalated=True, ended=True)
        self.turns.append(turn)
        return turn

    # -- think -> tool -> guard -> speak ------------------------------------

    def _act(self) -> tuple[str | None, tuple[ToolResult, ...], tuple[str, ...]]:
        results: list[ToolResult] = []
        for _ in range(MAX_TOOL_ROUNDS):
            response = self._ask_model()
            if response.error is not None or not response.wants_tools:
                spoken, blocked = self._speak_or_fall_back(response)
                return spoken, tuple(results), blocked
            for call in response.tool_calls:
                results.append(self._run_tool(call))
            if self.ended:
                break

        # Out of round trips, or the call closed inside a tool. Ask once more
        # for words; a turn that produces nothing to say is a dead phone line.
        spoken, blocked = self._speak_or_fall_back(self._ask_model())
        return spoken, tuple(results), blocked

    def _ask_model(self) -> LLMResponse:
        """One model round trip, with its cost on the record.

        Every call goes through here so the log accounts for all of them — a
        turn can spend four tool rounds and two regeneration strikes, and a
        latency budget you cannot attribute is one you cannot defend.
        """
        response = self.llm.respond(tuple(self.messages))
        self._record_model_call(response)
        return response

    def _record_model_call(self, response: LLMResponse) -> None:
        # A failed call with no usage record still has to leave a reason on the
        # trail. Gating the whole event on ``usage`` meant "never fail into
        # silence" held for the consumer's ear and not for the audit log — the
        # half that matters afterwards.
        usage = response.usage or (
            LLMUsage(model="unknown") if response.error is not None else None
        )
        if usage is not None:
            self._record(
                ModelCalled(
                    call_id=self.call_id,
                    turn_index=self._turn_index,
                    model=usage.model,
                    latency_ms=usage.latency_ms,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    cost_usd=usage.cost_usd,
                    stop_reason=usage.stop_reason,
                    error=response.error,
                )
            )

    def _run_tool(self, call: ToolCall) -> ToolResult:
        started = time.monotonic()
        result = execute(call, self.tools)
        latency_ms = int((time.monotonic() - started) * 1000)
        self.tools = result.context

        # Every tool, not just the two that produce a verdict. ``propose_offer``,
        # ``record_refusal``, ``concede`` and ``end_call`` move the negotiation
        # and previously left no trace of having been called.
        self._record(
            ToolInvoked(
                call_id=self.call_id,
                turn_index=self._turn_index,
                tool=call.name,
                arguments=_loggable(call.arguments),
                ok=result.ok,
                latency_ms=latency_ms,
                error=None if result.ok else str(result.payload.get("error", "")),
            )
        )

        if result.verdict is not None:
            self.verdicts.append(result.verdict)
            if result.proposal is not None:
                self._record(
                    DecisionRecorded(
                        call_id=self.call_id,
                        turn_index=self._turn_index,
                        proposal=result.proposal,
                        verdict=result.verdict,
                    )
                )
        if result.agreed_offer is not None:
            self.agreed_offer = result.agreed_offer
        if result.ends_call:
            self.ended = True

        # Re-point the numeric guard at everything the engine has authorized so
        # far. Figures age out of nothing: an offer made three turns ago may
        # still be referred to, and re-deriving the whole set keeps that true
        # without tracking expiry.
        self.authorized = authorized_for(
            self.policy, offers=self.tools.state.offers_made, verdicts=self.verdicts
        )
        self.guard = self.guard.with_authorized(self.authorized)

        self.messages.append(
            Message(
                role="tool",
                content=dumps(result.payload, indent=None),
                tool_call=call,
                tool_call_id=call.call_id,
            )
        )
        return result

    def _generate_and_speak(self) -> str | None:
        spoken, _ = self._speak_or_fall_back(self._ask_model())
        return spoken

    def _speak_or_fall_back(self, response: LLMResponse) -> tuple[str | None, tuple[str, ...]]:
        """Guard what the model produced, or speak the scripted line if it
        produced nothing because the call to it failed.

        A transport failure must not become silence. On a phone line silence
        *is* the dropped call, which is the outcome the timeout and the retry
        exist to avoid — so a failed call spends the same fallback the guard
        reaches for after two strikes.
        """
        if response.error is not None:
            fallback = fallback_for(self.guard)
            self._speak_verbatim(fallback)
            return fallback, ()
        return self._guard_and_speak(response.text)

    def _guard_and_speak(self, candidate: str) -> tuple[str | None, tuple[str, ...]]:
        """The pre-TTS gate, with the regeneration loop behind it.

        A blocked turn is not a failure — it is the guard working — so the
        violation is named back to the model and the turn is retried. After
        ``MAX_REGENERATION_STRIKES`` the scripted fallback is spoken instead,
        because a third rewrite of the same idea is not going to be the one
        that clears.
        """
        blocked: list[str] = []
        for _ in range(MAX_REGENERATION_STRIKES + 1):
            if not candidate.strip():
                return None, tuple(blocked)

            check = check_outbound(self.guard, candidate, authorized=self.authorized)
            self.guard = check.state

            if check.allowed:
                self._record_spoken(candidate)
                return candidate, tuple(blocked)

            blocked.append(candidate)
            note = check.regeneration_note()
            for violation in check.blocking_violations:
                self._record(
                    GuardrailTripped(
                        call_id=self.call_id,
                        turn_index=self._turn_index,
                        ring=GuardrailRing.DURING_CALL,
                        rule_id=violation.rule_id,
                        action=(
                            GuardrailAction.SAFE_FALLBACK
                            if check.fallback_text is not None
                            else GuardrailAction.REGENERATED
                        ),
                        detail=violation.detail,
                        blocked_text=candidate,
                    )
                )

            if check.fallback_text is not None:
                self._speak_verbatim(check.fallback_text)
                return check.fallback_text, tuple(blocked)

            self.messages.append(
                Message(
                    role="system",
                    content=(
                        "That turn was blocked before it was spoken and the consumer "
                        f"did not hear it: {note}. Say it again without whatever caused "
                        "that, and do not restate any figure the engine has not returned."
                    ),
                )
            )
            candidate = self._ask_model().text

        return None, tuple(blocked)

    def _speak_verbatim(self, text: str) -> None:
        """Speak a code-authored line. It bypasses regeneration because there
        is no model turn to regenerate — but it is still logged as spoken."""
        self._record_spoken(text)

    def _record_spoken(self, text: str) -> None:
        self.messages.append(Message(role="agent", content=text))
        self._record(
            TurnRecorded(
                call_id=self.call_id,
                turn_index=self._turn_index,
                speaker=Speaker.AGENT,
                text=text,
            )
        )

    # -- ring 3 ------------------------------------------------------------

    def close(self, *, transcript_persisted: bool | None = None) -> CallReport:
        """Post-call ring: score the trace, write the agreement if there is one."""
        persisted = self.store is not None if transcript_persisted is None else transcript_persisted
        summary = finalize_call(self.guard, transcript_persisted=persisted)
        outcome = self.tools.state.outcome
        if outcome is CallOutcome.IN_PROGRESS:
            outcome = CallOutcome.ABANDONED

        # The score reaches the log rather than only the caller: "was this call
        # compliant?" is the question the log exists to answer, and computing
        # it and then dropping it left that answer nowhere (SPEC §5.3).
        self._record(
            CallEnded(
                call_id=self.call_id,
                outcome=outcome,
                turn_count=len(self.turns),
                compliant=summary.compliant,
                blocked_turns=summary.blocked_turns,
                violation_count=len(summary.violations),
            )
        )

        if self.store is not None and self.agreed_offer is not None:
            authorizing = self._authorizing_verdict()
            if authorizing is not None:
                self.store.finalize_agreement(
                    call_id=self.call_id,
                    final_offer=self.agreed_offer,
                    authorizing_verdict=authorizing,
                    confirmation=ConsumerConfirmation(
                        confirmed=True,
                        utterance=self.last_consumer_utterance,
                        turn_index=self._turn_index,
                    ),
                )

        self.ended = True
        return CallReport(
            call_id=self.call_id,
            outcome=outcome,
            summary=summary,
            agreed_offer=self.agreed_offer,
            turns=len(self.turns),
        )

    def _authorizing_verdict(self) -> Verdict | None:
        """The accepting verdict over the final terms — written by
        ``confirm_agreement``, which is the last thing to rule on them."""
        for verdict in reversed(self.verdicts):
            if verdict.outcome == "accept":
                return verdict
        return None

    # -- audit -------------------------------------------------------------

    def _record(self, event: object) -> None:
        if self.store is not None:
            self.store.record(event)  # type: ignore[arg-type]
