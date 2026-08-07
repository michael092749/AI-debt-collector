"""Structured trace events.

Everything the call did, in a form a compliance reviewer can read without a
decoder ring: who said what, which rules the engine evaluated, which guardrails
tripped, and the agreement that came out the other end.

Two rules govern this module:

1. **Money serializes as an exact decimal string, never a float.** A float in a
   payment schedule is a compliance defect, not a rounding nit, so
   ``to_jsonable`` raises on one rather than quietly emitting it.
2. **The condition trail is carried verbatim.** The research report's vendor
   test is "show me the decision record; if they show you a transcript, the
   model decided; if they show you evaluated conditions and a policy path, the
   engine did." Every ``Verdict`` reaches the log with all of its conditions,
   passing ones included.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import IntEnum, StrEnum
from typing import Any, ClassVar, cast

from collector.decision_engine import Condition, Outcome, RationaleCode, RuleId, Verdict
from collector.guardrails.rings import EscalationTrigger, GuardrailRing
from collector.money import Money
from collector.negotiation import CallOutcome
from collector.offers import Cadence, ConsumerProposal, Installment, Offer, Tier

JsonDict = dict[str, Any]


def utc_now() -> str:
    """ISO-8601 UTC. The only clock in the audit layer, and it is injectable:
    every event takes ``at`` as an argument so traces can be made reproducible."""
    return datetime.now(UTC).isoformat()


# -- stable identifiers ----------------------------------------------------


class EventType(StrEnum):
    """Stable ids. These land in the database and must not churn."""

    CALL_STARTED = "call_started"
    TURN = "turn"
    DECISION = "decision"
    TOOL_CALL = "tool_call"
    MODEL_CALL = "model_call"
    GUARDRAIL_TRIP = "guardrail_trip"
    ESCALATION = "escalation"
    CALL_ENDED = "call_ended"
    AGREEMENT = "agreement"


class Speaker(StrEnum):
    AGENT = "agent"
    CONSUMER = "consumer"
    SYSTEM = "system"


class GuardrailAction(StrEnum):
    """What the guardrail did about it. ``BLOCKED`` means nothing was spoken."""

    BLOCKED = "blocked"
    REGENERATED = "regenerated"
    SAFE_FALLBACK = "safe_fallback"
    ESCALATED = "escalated"


# -- JSON encoding ---------------------------------------------------------


def to_jsonable(value: object) -> Any:
    """Convert a value object graph to JSON-safe primitives.

    ``Money`` becomes its exact decimal string ("800.00"), not a float and not
    the display form: the string round-trips back through ``Money`` bit for bit.
    ``Tier`` is an ``IntEnum``, so it is emitted by name — a log that says
    ``3`` where it means ``SETTLEMENT`` is not a record anyone can audit.
    """
    if isinstance(value, Money):
        return str(value.amount)
    if isinstance(value, float):
        raise TypeError(f"float {value!r} in an audit record; money and rates must be Decimal")
    if isinstance(value, IntEnum):
        return value.name
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, Offer):
        # Totals are derived properties rather than fields, and a reviewer
        # reading a counter-offer should not have to add the installments up.
        return {
            **{f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)},
            "total": str(value.total.amount),
            "payment_count": value.payment_count,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple | list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    raise TypeError(f"cannot serialize {type(value).__name__} into an audit record")


def dumps(value: object, *, indent: int | None = 2) -> str:
    # allow_nan=False: NaN/Infinity are not JSON and would signal float money.
    return json.dumps(to_jsonable(value), indent=indent, allow_nan=False)


# -- events ----------------------------------------------------------------


@dataclass(frozen=True)
class CallStarted:
    EVENT_TYPE: ClassVar[EventType] = EventType.CALL_STARTED

    call_id: str
    account_ref: str
    consumer_ref: str
    original_balance: Money
    channel: str = "text"
    at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class TurnRecorded:
    EVENT_TYPE: ClassVar[EventType] = EventType.TURN

    call_id: str
    turn_index: int
    speaker: Speaker
    text: str
    at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class DecisionRecorded:
    """One trip through ``validate_offer``: what the consumer proposed and what
    the engine ruled, conditions and counter included. The sequence of these is
    the negotiation's counter-offer exchange."""

    EVENT_TYPE: ClassVar[EventType] = EventType.DECISION

    call_id: str
    turn_index: int
    proposal: ConsumerProposal
    verdict: Verdict
    at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class ToolInvoked:
    """Every trip through the whitelist, whatever the tool.

    ``DecisionRecorded`` is the *compliance* record and only the engine-ruling
    tools produce one. This is the *operational* record and every tool produces
    one: what the model asked for, what came back, and how long it took. Without
    it the four tools that never touch ``validate_offer`` — the ones that end
    calls and move the ladder — leave no trace of having been called at all.
    """

    EVENT_TYPE: ClassVar[EventType] = EventType.TOOL_CALL

    call_id: str
    turn_index: int
    tool: str
    arguments: JsonDict
    ok: bool
    latency_ms: int
    # The tool's own ``error`` payload, not an exception: tools.py returns
    # failures as values, and this is where that value is kept.
    error: str | None = None
    at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class ModelCalled:
    """One round trip to the model: tokens, money, milliseconds.

    A voice turn has a hang-up budget measured in hundreds of milliseconds and
    a turn can make several of these, so latency here is the number that
    decides whether the architecture works. ``cost_usd`` is ``None`` when the
    model has no entry in the price table rather than guessed at.
    """

    EVENT_TYPE: ClassVar[EventType] = EventType.MODEL_CALL

    call_id: str
    turn_index: int
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Decimal | None = None
    stop_reason: str | None = None
    # Set when the call failed outright. The turn still gets a logged reason.
    error: str | None = None
    at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class GuardrailTripped:
    EVENT_TYPE: ClassVar[EventType] = EventType.GUARDRAIL_TRIP

    call_id: str
    turn_index: int
    ring: GuardrailRing
    rule_id: str  # guardrails/ owns its own rule ids; the log records them verbatim
    action: GuardrailAction
    detail: str
    blocked_text: str | None = None  # what would have been said, and never was
    at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class Escalated:
    """An escalation, and the callback obligation it leaves behind.

    Deliberately self-contained, for the same reason ``AgreementRecord`` is:
    this row is what a human picks the consumer back up from, and an
    obligation that can only be understood by going and reading the
    transcript alongside it is one that gets dropped. So the account and the
    words that triggered it are carried here rather than referenced.

    ``callback_owed`` is false for the two triggers where calling back is
    itself the wrong move — see ``rings.owes_callback``. The escalation still
    stands and a human still owns it; the phone call is what does not happen.
    """

    EVENT_TYPE: ClassVar[EventType] = EventType.ESCALATION

    call_id: str
    turn_index: int
    trigger: EscalationTrigger
    detail: str
    # The consumer's own words, as a field rather than only inside ``detail``:
    # the human acting on this reads them, and picking them back out of a
    # formatted string is not something a reader should have to do.
    consumer_utterance: str = ""
    account_ref: str = ""
    callback_owed: bool = False
    at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class CallEnded:
    EVENT_TYPE: ClassVar[EventType] = EventType.CALL_ENDED

    call_id: str
    outcome: CallOutcome
    turn_count: int
    # The post-call compliance score. ``None`` means the call never
    # reached the post-call ring — a dropped process, not a clean sheet.
    compliant: bool | None = None
    blocked_turns: int = 0
    violation_count: int = 0
    at: str = field(default_factory=utc_now)


TraceEvent = (
    CallStarted
    | TurnRecorded
    | DecisionRecorded
    | ToolInvoked
    | ModelCalled
    | GuardrailTripped
    | Escalated
    | CallEnded
)


def event_json(event: TraceEvent) -> JsonDict:
    """Lossless JSON form of an event, tagged with its stable event type."""
    payload: JsonDict = {f.name: to_jsonable(getattr(event, f.name)) for f in fields(event)}
    return {"event_type": event.EVENT_TYPE.value, **payload}


# -- the agreement record — the brief's deliverable ------------------------


@dataclass(frozen=True)
class ConsumerConfirmation:
    """The consumer saying yes, in their own words, at a locatable turn."""

    confirmed: bool
    utterance: str
    turn_index: int
    at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class AgreementRecord:
    """The final agreement plus everything that authorized it.

    Deliberately self-contained: tier, total, the full schedule with day
    offsets, the evaluated-condition trail from the authorizing verdict, every
    proposal/verdict exchanged during the call, every guardrail trip, and the
    consumer's confirmation. A reviewer holding only this record can answer
    "why was this arrangement allowed?" without a transcript and without the
    rest of the database.
    """

    EVENT_TYPE: ClassVar[EventType] = EventType.AGREEMENT

    agreement_id: str
    call_id: str
    tier: Tier
    total: Money
    cadence: Cadence
    schedule: tuple[Installment, ...]
    conditions: tuple[Condition, ...]
    rationale_code: RationaleCode
    exchanges: tuple[DecisionRecorded, ...]
    guardrail_events: tuple[GuardrailTripped, ...]
    escalations: tuple[Escalated, ...]
    confirmation: ConsumerConfirmation
    at: str = field(default_factory=utc_now)

    @classmethod
    def build(
        cls,
        *,
        call_id: str,
        final_offer: Offer,
        authorizing_verdict: Verdict,
        confirmation: ConsumerConfirmation,
        exchanges: tuple[DecisionRecorded, ...] = (),
        guardrail_events: tuple[GuardrailTripped, ...] = (),
        escalations: tuple[Escalated, ...] = (),
        agreement_id: str | None = None,
        at: str | None = None,
    ) -> AgreementRecord:
        """Assemble a record from the offer and the verdict that authorized it.

        The two refusals below are the record's integrity guarantees: an
        agreement no consumer confirmed, or one whose authorizing verdict was a
        rejection, is not an agreement and must never reach the log looking
        like one.
        """
        if not confirmation.confirmed:
            raise ValueError("an unconfirmed arrangement is not an agreement")
        if authorizing_verdict.outcome == "reject":
            raise ValueError("a rejected verdict cannot authorize an agreement")

        return cls(
            agreement_id=agreement_id or f"AGR-{call_id}",
            call_id=call_id,
            tier=final_offer.tier,
            total=final_offer.total,
            cadence=final_offer.cadence,
            schedule=final_offer.installments,
            conditions=authorizing_verdict.conditions,
            rationale_code=authorizing_verdict.rationale_code,
            exchanges=exchanges,
            guardrail_events=guardrail_events,
            escalations=escalations,
            confirmation=confirmation,
            at=at or utc_now(),
        )

    def to_json_dict(self) -> JsonDict:
        return {
            "event_type": self.EVENT_TYPE.value,
            "agreement_id": self.agreement_id,
            "call_id": self.call_id,
            "tier": self.tier.name,
            "total": str(self.total.amount),
            "cadence": self.cadence.value,
            "schedule": [to_jsonable(i) for i in self.schedule],
            "conditions": [to_jsonable(c) for c in self.conditions],
            "rationale_code": self.rationale_code.value,
            "exchanges": [event_json(e) for e in self.exchanges],
            "guardrail_events": [event_json(g) for g in self.guardrail_events],
            "escalations": [event_json(e) for e in self.escalations],
            "confirmation": to_jsonable(self.confirmation),
            "at": self.at,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_json_dict(cls, data: JsonDict) -> AgreementRecord:
        return cls(
            agreement_id=data["agreement_id"],
            call_id=data["call_id"],
            tier=Tier[data["tier"]],
            total=Money(data["total"]),
            cadence=Cadence(data["cadence"]),
            schedule=tuple(installment_from_json(i) for i in data["schedule"]),
            conditions=tuple(condition_from_json(c) for c in data["conditions"]),
            rationale_code=RationaleCode(data["rationale_code"]),
            exchanges=tuple(cast(DecisionRecorded, event_from_json(e)) for e in data["exchanges"]),
            guardrail_events=tuple(
                cast(GuardrailTripped, event_from_json(g)) for g in data["guardrail_events"]
            ),
            escalations=tuple(cast(Escalated, event_from_json(e)) for e in data["escalations"]),
            confirmation=ConsumerConfirmation(
                confirmed=bool(data["confirmation"]["confirmed"]),
                utterance=data["confirmation"]["utterance"],
                turn_index=int(data["confirmation"]["turn_index"]),
                at=data["confirmation"]["at"],
            ),
            at=data["at"],
        )

    @classmethod
    def from_json(cls, raw: str) -> AgreementRecord:
        return cls.from_json_dict(json.loads(raw))


# -- decoding --------------------------------------------------------------


def _optional_bool(value: object) -> bool | None:
    """``None`` and ``False`` are different answers here — a call with no
    compliance score is not a call that failed one."""
    return None if value is None else bool(value)


def money_from_json(value: str | None) -> Money | None:
    return None if value is None else Money(value)


def installment_from_json(data: JsonDict) -> Installment:
    return Installment(amount=Money(data["amount"]), due_day_offset=int(data["due_day_offset"]))


def offer_from_json(data: JsonDict | None) -> Offer | None:
    if data is None:
        return None
    return Offer(
        tier=Tier[data["tier"]],
        installments=tuple(installment_from_json(i) for i in data["installments"]),
        cadence=Cadence(data["cadence"]),
    )


def condition_from_json(data: JsonDict) -> Condition:
    return Condition(
        rule_id=RuleId(data["rule_id"]),
        passed=bool(data["passed"]),
        actual=data["actual"],
        limit=data["limit"],
    )


def verdict_from_json(data: JsonDict) -> Verdict:
    return Verdict(
        outcome=cast(Outcome, data["outcome"]),
        tier=Tier[data["tier"]] if data["tier"] is not None else None,
        conditions=tuple(condition_from_json(c) for c in data["conditions"]),
        counter=offer_from_json(data["counter"]),
        rationale_code=RationaleCode(data["rationale_code"]),
    )


def proposal_from_json(data: JsonDict) -> ConsumerProposal:
    return ConsumerProposal(
        total=Money(data["total"]),
        payment_count=int(data["payment_count"]),
        cadence=Cadence(data["cadence"]),
        signaled_capacity=money_from_json(data["signaled_capacity"]),
    )


def event_from_json(data: JsonDict) -> TraceEvent:
    """Rebuild an event from its stored JSON payload."""
    event_type = EventType(data["event_type"])
    match event_type:
        case EventType.CALL_STARTED:
            return CallStarted(
                call_id=data["call_id"],
                account_ref=data["account_ref"],
                consumer_ref=data["consumer_ref"],
                original_balance=Money(data["original_balance"]),
                channel=data["channel"],
                at=data["at"],
            )
        case EventType.TURN:
            return TurnRecorded(
                call_id=data["call_id"],
                turn_index=int(data["turn_index"]),
                speaker=Speaker(data["speaker"]),
                text=data["text"],
                at=data["at"],
            )
        case EventType.DECISION:
            return DecisionRecorded(
                call_id=data["call_id"],
                turn_index=int(data["turn_index"]),
                proposal=proposal_from_json(data["proposal"]),
                verdict=verdict_from_json(data["verdict"]),
                at=data["at"],
            )
        case EventType.TOOL_CALL:
            return ToolInvoked(
                call_id=data["call_id"],
                turn_index=int(data["turn_index"]),
                tool=data["tool"],
                arguments=dict(data["arguments"]),
                ok=bool(data["ok"]),
                latency_ms=int(data["latency_ms"]),
                error=data["error"],
                at=data["at"],
            )
        case EventType.MODEL_CALL:
            cost = data["cost_usd"]
            return ModelCalled(
                call_id=data["call_id"],
                turn_index=int(data["turn_index"]),
                model=data["model"],
                latency_ms=int(data["latency_ms"]),
                input_tokens=int(data["input_tokens"]),
                output_tokens=int(data["output_tokens"]),
                cache_read_tokens=int(data["cache_read_tokens"]),
                cache_write_tokens=int(data["cache_write_tokens"]),
                cost_usd=None if cost is None else Decimal(cost),
                stop_reason=data["stop_reason"],
                error=data["error"],
                at=data["at"],
            )
        case EventType.GUARDRAIL_TRIP:
            return GuardrailTripped(
                call_id=data["call_id"],
                turn_index=int(data["turn_index"]),
                ring=GuardrailRing(data["ring"]),
                rule_id=data["rule_id"],
                action=GuardrailAction(data["action"]),
                detail=data["detail"],
                blocked_text=data["blocked_text"],
                at=data["at"],
            )
        case EventType.ESCALATION:
            return Escalated(
                call_id=data["call_id"],
                turn_index=int(data["turn_index"]),
                trigger=EscalationTrigger(data["trigger"]),
                detail=data["detail"],
                # ``.get``: rows written before the callback obligation existed
                # carry none of these keys, and an evidence store that cannot
                # read its own older rows back has lost the evidence.
                consumer_utterance=data.get("consumer_utterance", ""),
                account_ref=data.get("account_ref", ""),
                callback_owed=bool(data.get("callback_owed", False)),
                at=data["at"],
            )
        case EventType.CALL_ENDED:
            return CallEnded(
                call_id=data["call_id"],
                outcome=CallOutcome(data["outcome"]),
                turn_count=int(data["turn_count"]),
                compliant=_optional_bool(data.get("compliant")),
                blocked_turns=int(data.get("blocked_turns") or 0),
                violation_count=int(data.get("violation_count") or 0),
                at=data["at"],
            )
        case EventType.AGREEMENT:
            raise ValueError("use AgreementRecord.from_json_dict for agreement records")
