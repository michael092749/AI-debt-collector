"""The action whitelist.

Six tools. The model may do these things and nothing else: no arithmetic, no
free-form state changes, no way to reach the audit log or the guardrails. Every
figure it is allowed to say aloud entered the conversation through a return
value from this module.

Two properties hold by construction:

1. **Failures come back as payloads, not exceptions.** A bad cadence, a
   concession with nothing to concede, a call already closed — each returns an
   ``ok: false`` result the model can read and recover from. An exception here
   would end a call over a typo.
2. **Nothing mutates.** Every tool returns its successor ``ToolContext``. The
   caller threads it forward, so a dropped call replays exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from collector.audit.events import to_jsonable
from collector.decision_engine import (
    Verdict,
    build_counter,
    effective_capacity,
    validate_offer,
)
from collector.llm.base import ToolCall
from collector.money import Money
from collector.negotiation import CallOutcome, NegotiationState
from collector.offers import Cadence, ConsumerProposal, Offer, Tier
from collector.policy import PolicyConfig

JsonDict = dict[str, Any]


# -- schemas the model sees ------------------------------------------------


_CADENCE_ENUM = [c.value for c in Cadence]

# A sanity ceiling, not a policy one. The engine must be allowed to *rule* on
# an absurd proposal — countering an impossible schedule is the job, and a
# refusal at this layer produces no verdict, no counter and no condition trail,
# so the model gets "bad argument" where it needed "here is why, and here is
# what I can do instead".
#
# 1000 was too low and contradicted that: "five dollars a week for twenty
# years" is 1042 payments and "$20 a week over 25 years" is 1303 — exactly the
# deliberately-absurd offers an adversarial consumer makes, and precisely the
# ones the engine should be countering. This is set far above any utterable
# schedule so that what it rejects is only a malformed generation.
MAX_PROPOSED_PAYMENTS = 100_000


class ArgumentError(ValueError):
    """A tool argument that could not be read as its declared type."""


class ParamKind(StrEnum):
    MONEY = "money"
    COUNT = "count"
    CADENCE = "cadence"
    TEXT = "text"


@dataclass(frozen=True)
class Param:
    """One tool argument, declared once.

    The JSON schema the model sees, the coercion out of JSON's loose types, and
    the bounds check are all generated from this declaration. They used to be
    three separate statements of the same fact — a ``"minimum": 1`` in the
    schema, an ``if count < 1`` in a coercion helper, and an assumption in the
    engine — and three statements of one fact drift.
    """

    name: str
    kind: ParamKind
    description: str
    required: bool = False
    minimum: int | None = None
    maximum: int | None = None

    @property
    def json_schema(self) -> JsonDict:
        """The property as the model sees it."""
        schema: JsonDict = {"type": _JSON_TYPES[self.kind], "description": self.description}
        if self.kind is ParamKind.CADENCE:
            schema["enum"] = _CADENCE_ENUM
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        return schema

    def parse(self, value: object) -> object:
        """Coerce and bounds-check one supplied value.

        Raises ``ArgumentError`` naming the parameter, because the message is
        handed back to the model as a tool result and it has to be actionable.
        """
        try:
            parsed = _PARSERS[self.kind](value)
        except (ValueError, TypeError, ArithmeticError, InvalidOperation) as exc:
            raise ArgumentError(f"{self.name}: {exc}") from exc
        if isinstance(parsed, int) and not isinstance(parsed, bool):
            if self.minimum is not None and parsed < self.minimum:
                raise ArgumentError(f"{self.name} must be at least {self.minimum}; got {parsed}")
            if self.maximum is not None and parsed > self.maximum:
                raise ArgumentError(f"{self.name} must be at most {self.maximum}; got {parsed}")
        return parsed


def _parse_money(value: object) -> Money:
    """Parse an amount from tool arguments.

    JSON has one number type and it decodes to ``float``, which ``Money``
    refuses on purpose. Routing through ``str`` at this boundary
    keeps the exact digits the model actually emitted — ``str(50.5)`` is
    "50.5" — without letting a float any further into the system.
    """
    if isinstance(value, Money):
        return value
    if not isinstance(value, float | int | str | Decimal):
        raise ValueError(f"expected an amount, got {type(value).__name__}")
    amount = Decimal(str(value))
    # ``Decimal("NaN")`` is a valid Decimal and quantizes to NaN without
    # complaint, so it clears Money and every check here — and then raises
    # ``InvalidOperation`` on the first comparison inside ``validate_offer``,
    # out of a function documented as pure and total. ``Infinity`` is caught by
    # Money's quantize; NaN is the one that has to be named. (Rejected here
    # rather than in Money so the model gets a readable tool error it can
    # correct, instead of an exception that ends the call.)
    if not amount.is_finite():
        raise ValueError(f"expected a finite amount, got {value!r}")
    return Money(amount)


def _parse_count(value: object) -> int:
    if not isinstance(value, int | float | str | Decimal):
        raise ValueError(f"expected a whole number, got {type(value).__name__}")
    count = Decimal(str(value))
    if not count.is_finite():
        raise ValueError(f"expected a finite whole number, got {value!r}")
    # Reject rather than truncate. The schema tells the model this field is an
    # integer; silently turning 3.9 into 3 accepts a type the declaration says
    # is not accepted, and the model never learns it sent the wrong thing.
    if count != count.to_integral_value():
        raise ValueError(f"expected a whole number, got {value!r}")
    return int(count)


def _parse_cadence(value: object) -> Cadence:
    try:
        return Cadence(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"must be one of {', '.join(_CADENCE_ENUM)}; got {value!r}") from exc


def _parse_text(value: object) -> str:
    return str(value)


_JSON_TYPES: dict[ParamKind, str] = {
    ParamKind.MONEY: "string",
    ParamKind.COUNT: "integer",
    ParamKind.CADENCE: "string",
    ParamKind.TEXT: "string",
}

_PARSERS: dict[ParamKind, Callable[[object], object]] = {
    ParamKind.MONEY: _parse_money,
    ParamKind.COUNT: _parse_count,
    ParamKind.CADENCE: _parse_cadence,
    ParamKind.TEXT: _parse_text,
}


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    params: tuple[Param, ...] = ()

    @property
    def input_schema(self) -> JsonDict:
        """JSON Schema for the model, generated from ``params``."""
        return {
            "type": "object",
            "properties": {p.name: p.json_schema for p in self.params},
            "required": [p.name for p in self.params if p.required],
        }

    def parse(self, arguments: JsonDict) -> JsonDict:
        """Validate and coerce the model's arguments in one pass.

        Absent optional arguments stay absent rather than becoming ``None``:
        the tools distinguish "they named no sum" from "they named nothing",
        and a default injected here would erase that distinction.

        An *unknown* argument is an error, not something to ignore. Dropping it
        silently is how a key confusion becomes a falsified decision record:
        ``amount="500"`` instead of ``total="500"`` leaves ``total`` absent, so
        the full balance is assumed, the engine accepts $1,000 in one payment,
        and the agreement record describes a proposal the consumer never made.
        Naming the bad key gives the model something it can correct.
        """
        unknown = sorted(set(arguments) - {p.name for p in self.params})
        if unknown:
            known = ", ".join(p.name for p in self.params) or "none"
            raise ArgumentError(
                f"{self.name} has no argument {', '.join(repr(u) for u in unknown)}; "
                f"accepted: {known}"
            )
        parsed: JsonDict = {}
        for param in self.params:
            if param.name not in arguments or arguments[param.name] is None:
                if param.required:
                    raise ArgumentError(f"{param.name} is required")
                continue
            parsed[param.name] = param.parse(arguments[param.name])
        return parsed


_PREFERRED_CADENCE = Param(
    name="preferred_cadence",
    kind=ParamKind.CADENCE,
    description="Cadence the consumer asked for, honoured where possible.",
)

TOOL_SCHEMAS: tuple[ToolSchema, ...] = (
    ToolSchema(
        name="validate_consumer_offer",
        description=(
            "Rule on what the consumer just proposed. Call this for any amount, "
            "payment count or timeframe they name, however unreasonable it sounds — "
            "you do not judge proposals yourself. Returns the verdict, every policy "
            "condition that was evaluated, and a counter-offer when the proposal "
            "cannot be accepted."
        ),
        params=(
            Param(
                name="total",
                kind=ParamKind.MONEY,
                description=(
                    "Total they offered to pay, as a decimal string, e.g. '500.00'. "
                    "Omit it when they proposed only a structure and no sum — "
                    "'weekly for a year' — and the full balance is assumed."
                ),
            ),
            Param(
                name="payment_count",
                kind=ParamKind.COUNT,
                description="How many payments to split it into. 1 for a lump sum.",
                required=True,
                minimum=1,
                maximum=MAX_PROPOSED_PAYMENTS,
            ),
            Param(
                name="cadence",
                kind=ParamKind.CADENCE,
                description="How often they proposed to pay.",
                required=True,
            ),
            Param(
                name="signaled_capacity",
                kind=ParamKind.MONEY,
                description=(
                    "What they said they can afford at one time, if they said it "
                    "at all, as a decimal string. Omit rather than guessing."
                ),
            ),
        ),
    ),
    ToolSchema(
        name="propose_offer",
        description=(
            "Ask for the offer to put on the table now. Use it to open the "
            "negotiation and any time you need the current terms restated."
        ),
        params=(_PREFERRED_CADENCE,),
    ),
    ToolSchema(
        name="record_refusal",
        description=(
            "Record that the consumer turned down the offer on the table or pushed "
            "back on it. This is what earns the right to concede later; without it "
            "concede will refuse."
        ),
    ),
    ToolSchema(
        name="concede",
        description=(
            "Move to the next arrangement down. Returns the new offer, which is the "
            "only one you may now describe. Fails if nothing has been refused."
        ),
        params=(_PREFERRED_CADENCE,),
    ),
    ToolSchema(
        name="confirm_agreement",
        description=(
            "The consumer has accepted the offer currently on the table. Closes the "
            "negotiation and returns the agreed schedule for you to read back."
        ),
    ),
    ToolSchema(
        name="end_call",
        description="End the call without an agreement.",
        params=(
            Param(
                name="reason",
                kind=ParamKind.TEXT,
                description="Short note on why the call is ending, for the log.",
            ),
        ),
    ),
)

TOOL_NAMES: frozenset[str] = frozenset(s.name for s in TOOL_SCHEMAS)
_SCHEMAS: dict[str, ToolSchema] = {s.name: s for s in TOOL_SCHEMAS}


# -- context and results ---------------------------------------------------


@dataclass(frozen=True)
class ToolContext:
    policy: PolicyConfig
    state: NegotiationState

    @classmethod
    def opening(cls, policy: PolicyConfig) -> ToolContext:
        return cls(policy=policy, state=NegotiationState.opening(policy))

    def _with(self, state: NegotiationState) -> ToolContext:
        return replace(self, state=state)

    @property
    def standing_offer(self) -> Offer | None:
        """The offer currently on the table — the last one the engine authored."""
        return self.state.offers_made[-1] if self.state.offers_made else None


@dataclass(frozen=True)
class ToolResult:
    """What the model gets back, plus what the rest of the system needs.

    ``payload`` is the model's whole view. The typed fields beside it exist so
    the agent loop can extend the numeric authorization and write the audit
    trail without re-parsing JSON.
    """

    name: str
    payload: JsonDict
    context: ToolContext
    proposal: ConsumerProposal | None = None
    verdict: Verdict | None = None
    offer: Offer | None = None
    agreed_offer: Offer | None = None
    ends_call: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.payload.get("ok"))


def _error(name: str, context: ToolContext, detail: str, **extra: Any) -> ToolResult:
    return ToolResult(name=name, payload={"ok": False, "error": detail, **extra}, context=context)


def _cadence_arg(args: JsonDict, default: Cadence = Cadence.MONTHLY) -> Cadence:
    """``preferred_cadence`` is a hint, so its absence is not an error.

    Already validated by the schema layer if present — this only supplies the
    fallback for a model that named a structure without naming a rhythm.
    """
    cadence = args.get("preferred_cadence")
    return cadence if isinstance(cadence, Cadence) else default


# -- the tools -------------------------------------------------------------


def _validate_consumer_offer(args: JsonDict, context: ToolContext) -> ToolResult:
    name = "validate_consumer_offer"
    if context.state.is_exhausted:
        # The round cap (anti-badgering) is enforced on propose_offer and
        # concede; validate_consumer_offer must refuse the same way once
        # exhausted, or the consumer can keep the negotiation open forever
        # simply by continuing to name terms (ADVERSARIAL_TESTING.md M3).
        return _error(
            name,
            context,
            "this call has run its course; close it out with end_call rather than "
            "evaluating another proposal",
        )
    proposal = ConsumerProposal(
        # Membership, not truthiness. "They proposed $0" and "they named no sum
        # at all" are different negotiation states — the first is a lowball to
        # rule on, the second means the balance stands and only the structure
        # is in question — and a falsy-zero would silently merge them.
        total=args.get("total", context.policy.original_balance),
        payment_count=args["payment_count"],
        cadence=args["cadence"],
        signaled_capacity=args.get("signaled_capacity"),
    )

    verdict = validate_offer(proposal, context.state, context.policy)

    # An accepted proposal becomes the standing offer: it is what the consumer
    # will be asked to confirm, so it has to be a real schedule and not a
    # remembered sentence.
    accepted = (
        Offer.from_proposal(proposal, verdict.tier)
        if verdict.outcome == "accept" and verdict.tier is not None
        else None
    )
    # We never take more than we *asked* for. Same tier, same money, harsher
    # schedule than the one on the table means they have agreed to our
    # arrangement and mis-stated it, so it closes on ours: "two payments" read
    # against the balance is an even $500/$500, and taking that over a standing
    # $400/$600 raises the ask $100 on a turn the consumer spent agreeing.
    #
    # Only when the totals match. More than we asked for is not a mistake to
    # correct — $900 against a standing $800 settlement is $100 they offered,
    # and a $400/$600 offer answered with the whole $1,000 today is the best
    # outcome on the list, not something to talk back down into instalments.
    standing = context.standing_offer
    if (
        accepted is not None
        and standing is not None
        and standing.tier is accepted.tier
        and standing.total == accepted.total
        and _is_concession(standing, accepted)
    ):
        accepted = standing
    on_table = accepted or verdict.counter
    # Whatever they just proposed tells us what they think they can manage, and
    # every later counter and concession is built against that read.
    state = context.state.with_capacity(effective_capacity(proposal, context.state))
    state = state.record_round(proposal, on_table, verdict.rationale_code)
    if verdict.outcome != "accept":
        # They named terms we cannot meet; that is a refusal of ours in all but
        # wording, and it is what makes a later concession legitimate.
        state = state.record_refusal()

    payload: JsonDict = {
        "ok": True,
        "outcome": verdict.outcome,
        "rationale_code": verdict.rationale_code.value,
        "tier": verdict.tier.label if verdict.tier else None,
        "conditions": [to_jsonable(c) for c in verdict.conditions],
        "offer_on_the_table": _offer_payload(on_table),
        "you_may_say": _sayable(on_table),
    }
    return ToolResult(
        name=name,
        payload=payload,
        context=context._with(state),
        proposal=proposal,
        verdict=verdict,
        offer=on_table,
    )


def _propose_offer(args: JsonDict, context: ToolContext) -> ToolResult:
    name = "propose_offer"
    if context.state.is_exhausted:
        return _error(
            name,
            context,
            "this call has run its course; close it out with end_call rather than "
            "putting another offer up",
        )
    offer = build_counter(context.state, context.policy, preferred_cadence=_cadence_arg(args))
    state = context.state.record_round(None, offer, None)
    return ToolResult(
        name=name,
        payload={
            "ok": True,
            "offer_on_the_table": _offer_payload(offer),
            "you_may_say": _sayable(offer),
        },
        context=context._with(state),
        offer=offer,
    )


def _record_refusal(args: JsonDict, context: ToolContext) -> ToolResult:
    if context.state.is_exhausted:
        return _error(
            "record_refusal",
            context,
            "this call has run its course; close it out with end_call rather than "
            "recording another refusal",
        )
    state = context.state.record_refusal()
    return ToolResult(
        name="record_refusal",
        payload={
            "ok": True,
            "refusals_on_record": state.pending_refusals,
            "may_concede": state.can_concede,
        },
        context=context._with(state),
    )


def _concede(args: JsonDict, context: ToolContext) -> ToolResult:
    name = "concede"
    if not context.state.can_concede:
        return _error(
            name,
            context,
            "nothing to concede against: record_refusal first. Concessions are "
            "earned, not offered up front",
            may_concede=False,
        )
    if context.state.is_exhausted:
        return _error(name, context, "round limit reached; close the call out with end_call")
    # A concession is the first place the engine has to guess at what the
    # consumer can put down: stepping to DOWNPAYMENT_PLUS_ONE with no capacity
    # on record takes the ceiling by design (decision_engine.py:401-407), so a
    # consumer who said "two hundred" and was never relayed gets asked for
    # $750 today. Nothing above the engine can catch that — the engine authored
    # the figures, so it authorized them (decision_engine.py:127-131). The
    # signal is only missing because the model skipped the tool that carries
    # it: ``signaled_capacity`` is written nowhere but validate_consumer_offer,
    # and once written it is never cleared, so this refuses at most once, before
    # the first ruling of the call.
    #
    # ``propose_offer`` reads the same field and needs no equivalent guard: it
    # only reaches that branch below PAY_IN_FULL, and this is the sole caller of
    # advance_ladder(), so a moved ladder now implies a capacity on record. That
    # is an argument, so it is also swept: test_engine_invariants.py walks every
    # short tool sequence and asserts it of whatever reaches the table.
    #
    # The round is recorded even though the call is refused, for the same reason
    # a non-concession step records one further down: a consumer who keeps
    # refusing without ever naming a figure would otherwise never advance the
    # round count, leaving the cap unreachable and the call unable to close.
    if context.state.signaled_capacity is None:
        return _error(
            name,
            context._with(context.state.record_round(None, None, None)),
            "nothing on record about what they can manage, and a concession has to "
            "be built around that. If they named an amount, a payment count or a "
            "timeframe, put it through validate_consumer_offer first; if they have "
            "not, ask them what they can manage before conceding",
            may_concede=True,
        )
    cadence = _cadence_arg(args)

    # Step until the terms actually improve for the consumer. Two ways a step
    # can fail to be a concession: the offer comes back identical (capacity had
    # already pushed the engine past that tier), or it comes back *worse* — the
    # tier order runs settlement before payment plan, so descending from an
    # $800 settlement to a full-balance plan raises the ask. Neither is
    # something to present as giving ground.
    standing = context.standing_offer
    stepped = context.state.advance_ladder()
    offer = build_counter(stepped, context.policy, preferred_cadence=cadence)
    while (
        not _is_concession(offer, standing)
        and stepped.can_concede
        and stepped.ladder_floor is not _LAST_TIER
    ):
        stepped = stepped.advance_ladder()
        offer = build_counter(stepped, context.policy, preferred_cadence=cadence)

    moved = _is_concession(offer, standing)
    # Nothing better was available, so the offer on the table is the one that
    # was already there. Reporting the candidate instead would have the agent
    # read back terms it is not authorized to agree to — and confirm_agreement
    # would then close on the standing offer, not the one just described.
    on_table = offer if moved else standing
    # When the step wasn't a concession, ``stepped`` still carries a spent
    # refusal and a moved ladder floor from advance_ladder() — committing it
    # anyway would consume the consumer's refusal for zero benefit. Keep the
    # pristine input state instead, so the refusal is still on record for a
    # concession that actually helps them (ADVERSARIAL_TESTING.md L1). The
    # round itself must still be recorded either way: a non-concession is
    # still an exchange, and once the ladder is bottomed out every further
    # refusal would otherwise never advance the round count, leaving the
    # round cap unreachable and the call unable to close (badgering with no
    # backstop).
    state = (stepped if moved else context.state).record_round(None, on_table, None)
    return ToolResult(
        name=name,
        payload={
            "ok": True,
            "moved": moved,
            "offer_on_the_table": _offer_payload(on_table),
            "you_may_say": _sayable(on_table),
        },
        context=context._with(state),
        offer=on_table,
    )


# Bottom of the ladder: there is nothing below a payment plan to step down to.
_LAST_TIER = Tier.PAYMENT_PLAN


def _is_concession(offer: Offer, standing: Offer | None) -> bool:
    """Is ``offer`` genuinely easier for the consumer than what they refused?

    Easier means less money, or the same money in smaller pieces. Anything else
    is a lateral move at best, and the consumer hears it as one.
    """
    if standing is None:
        return True
    if offer.total != standing.total:
        return offer.total < standing.total
    return offer.smallest_payment < standing.smallest_payment


def _confirm_agreement(args: JsonDict, context: ToolContext) -> ToolResult:
    name = "confirm_agreement"

    # A dropped call replays exactly (module docstring) — including a replayed
    # confirm_agreement after the call already closed. _step() raises on any
    # transition attempted from a terminal state, so that must be checked
    # here rather than fallen into; a plain second call to confirm the same
    # agreement is not an error, it is exactly what a replay looks like.
    if context.state.is_terminal:
        if context.state.outcome is CallOutcome.AGREED and context.state.agreed_offer is not None:
            agreed = context.state.agreed_offer
            verdict = validate_offer(_as_proposal(agreed), context.state, context.policy)
            return ToolResult(
                name=name,
                payload={
                    "ok": True,
                    "agreed": True,
                    "agreement": _offer_payload(agreed),
                    "conditions": [to_jsonable(c) for c in verdict.conditions],
                    "you_may_say": _sayable(agreed),
                },
                context=context,
                verdict=verdict,
                offer=agreed,
                agreed_offer=agreed,
                ends_call=True,
            )
        return _error(
            name,
            context,
            f"the call is already closed ({context.state.outcome.value}); nothing further",
        )

    offer = context.standing_offer
    if offer is None:
        return _error(name, context, "there is no offer on the table to agree to")

    # Run the final terms back through the engine one last time. Every offer
    # here was engine-authored, so this should always pass — which is exactly
    # why it is worth asserting, and it gives the agreement record the
    # evaluated-condition trail for the terms actually agreed to rather than
    # for some earlier proposal.
    verdict = validate_offer(_as_proposal(offer), context.state, context.policy)
    if verdict.outcome != "accept":
        return _error(
            name,
            context,
            "those terms do not pass policy and cannot be agreed to "
            f"({verdict.rationale_code.value}); propose_offer for a legal arrangement",
        )

    state = context.state.agree(offer)
    return ToolResult(
        name=name,
        payload={
            "ok": True,
            "agreed": True,
            "agreement": _offer_payload(offer),
            "conditions": [to_jsonable(c) for c in verdict.conditions],
            "you_may_say": _sayable(offer),
        },
        context=context._with(state),
        verdict=verdict,
        offer=offer,
        agreed_offer=offer,
        ends_call=True,
    )


def _as_proposal(offer: Offer) -> ConsumerProposal:
    """The offer restated as something the engine can rule on."""
    return ConsumerProposal(
        total=offer.total,
        payment_count=offer.payment_count,
        cadence=offer.cadence,
        signaled_capacity=offer.smallest_payment,
    )


def _end_call(args: JsonDict, context: ToolContext) -> ToolResult:
    state = context.state
    if not state.is_terminal:
        state = state.end_without_agreement()
    return ToolResult(
        name="end_call",
        payload={"ok": True, "outcome": state.outcome.value},
        context=context._with(state),
        ends_call=True,
    )


_DISPATCH = {
    "validate_consumer_offer": _validate_consumer_offer,
    "propose_offer": _propose_offer,
    "record_refusal": _record_refusal,
    "concede": _concede,
    "confirm_agreement": _confirm_agreement,
    "end_call": _end_call,
}


def execute(call: ToolCall, context: ToolContext) -> ToolResult:
    """Run one tool call. The only way the model can affect anything.

    Arguments clear the schema layer before a handler sees them, so every
    handler below can assume its arguments are already the types it declared.
    A malformed argument comes back as an ``ok: false`` payload naming the
    parameter, which the model can read and correct on the next round trip.
    """
    handler = _DISPATCH.get(call.name)
    schema = _SCHEMAS.get(call.name)
    if handler is None or schema is None:
        return _error(
            call.name,
            context,
            f"{call.name!r} is not an available action. Available: {', '.join(sorted(TOOL_NAMES))}",
        )
    if context.state.is_terminal and call.name not in {"end_call", "confirm_agreement"}:
        return _error(
            call.name,
            context,
            f"the call is already closed ({context.state.outcome.value}); nothing further",
        )
    try:
        arguments = schema.parse(call.arguments)
    except ArgumentError as exc:
        return _error(call.name, context, f"could not read that call: {exc}")
    return handler(arguments, context)


# -- rendering for the model ----------------------------------------------


def _offer_payload(offer: Offer | None) -> JsonDict | None:
    if offer is None:
        return None
    return {
        "tier": offer.tier.label,
        "total": str(offer.total.amount),
        "payment_count": offer.payment_count,
        "cadence": offer.cadence.value,
        "schedule": [
            {"amount": str(i.amount.amount), "due_in_days": i.due_day_offset}
            for i in offer.installments
        ],
        "duration_days": offer.duration_days,
    }


def _sayable(offer: Offer | None) -> list[str]:
    """The figures from this offer, pre-formatted the way they should be spoken.

    Not decoration: the numeric guard blocks anything not in the authorized set,
    and this is the model's copy of that set for the offer just computed. Read
    from here and the turn passes; paraphrase a number and it does not.
    """
    if offer is None:
        return []
    said = [str(offer.total), str(offer.payment_count)]
    said.extend(str(i.amount) for i in offer.installments)
    return list(dict.fromkeys(said))
