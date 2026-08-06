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

from dataclasses import dataclass, field

from collector.audit.events import (
    CallEnded,
    CallStarted,
    ConsumerConfirmation,
    DecisionRecorded,
    Escalated,
    GuardrailAction,
    GuardrailTripped,
    Speaker,
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
    PreCallCheck,
    PreCallContext,
    check_inbound,
    check_outbound,
    check_pre_call,
    finalize_call,
)
from collector.llm.base import LLMClient, Message, ToolCall, system_prompt
from collector.negotiation import CallOutcome
from collector.offers import Offer
from collector.policy import PolicyConfig
from collector.tools import ToolContext, ToolResult, execute

# A turn that has made this many engine round trips is looping, not thinking.
MAX_TOOL_ROUNDS = 4


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
            return check, None

        self._record(
            CallStarted(
                call_id=self.call_id,
                account_ref=self.account_ref,
                consumer_ref=self.consumer_name,
                original_balance=self.policy.original_balance,
                channel=self.channel,
            )
        )
        return check, self._generate_and_speak()

    # -- ring 2 ------------------------------------------------------------

    def turn(self, consumer_utterance: str) -> AgentTurn:
        """One full exchange. The only method a transport needs to call."""
        if self.ended:
            raise ValueError("the call has ended; start a new one")

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
        self.tools = ToolContext(
            policy=self.policy, state=self.tools.state.escalate(str(trigger))
        )
        self._speak_verbatim(closing)
        self.ended = True

        turn = AgentTurn(
            consumer=consumer_utterance, spoken=closing, escalated=True, ended=True
        )
        self.turns.append(turn)
        return turn

    # -- think -> tool -> guard -> speak ------------------------------------

    def _act(self) -> tuple[str | None, tuple[ToolResult, ...], tuple[str, ...]]:
        results: list[ToolResult] = []
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.llm.respond(tuple(self.messages))
            if not response.wants_tools:
                spoken, blocked = self._guard_and_speak(response.text)
                return spoken, tuple(results), blocked
            for call in response.tool_calls:
                results.append(self._run_tool(call))
            if self.ended:
                break

        # Out of round trips, or the call closed inside a tool. Ask once more
        # for words; a turn that produces nothing to say is a dead phone line.
        response = self.llm.respond(tuple(self.messages))
        spoken, blocked = self._guard_and_speak(response.text)
        return spoken, tuple(results), blocked

    def _run_tool(self, call: ToolCall) -> ToolResult:
        result = execute(call, self.tools)
        self.tools = result.context

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
        response = self.llm.respond(tuple(self.messages))
        spoken, _ = self._guard_and_speak(response.text)
        return spoken

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
            candidate = self.llm.respond(tuple(self.messages)).text

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

        self._record(
            CallEnded(call_id=self.call_id, outcome=outcome, turn_count=len(self.turns))
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
