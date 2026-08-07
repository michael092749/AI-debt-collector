"""The action whitelist — SPEC §3, §5.2.

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

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
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


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    input_schema: JsonDict


_CADENCE_ENUM = [c.value for c in Cadence]

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
        input_schema={
            "type": "object",
            "properties": {
                "total": {
                    "type": "string",
                    "description": (
                        "Total they offered to pay, as a decimal string, e.g. '500.00'. "
                        "Omit it when they proposed only a structure and no sum — "
                        "'weekly for a year' — and the full balance is assumed."
                    ),
                },
                "payment_count": {
                    "type": "integer",
                    "description": "How many payments to split it into. 1 for a lump sum.",
                    "minimum": 1,
                    "maximum": 1000,
                },
                "cadence": {
                    "type": "string",
                    "enum": _CADENCE_ENUM,
                    "description": "How often they proposed to pay.",
                },
                "signaled_capacity": {
                    "type": "string",
                    "description": (
                        "What they said they can afford at one time, if they said it "
                        "at all, as a decimal string. Omit rather than guessing."
                    ),
                },
            },
            "required": ["payment_count", "cadence"],
        },
    ),
    ToolSchema(
        name="propose_offer",
        description=(
            "Ask for the offer to put on the table now. Use it to open the "
            "negotiation and any time you need the current terms restated."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "preferred_cadence": {
                    "type": "string",
                    "enum": _CADENCE_ENUM,
                    "description": "Cadence the consumer asked for, honoured where possible.",
                }
            },
            "required": [],
        },
    ),
    ToolSchema(
        name="record_refusal",
        description=(
            "Record that the consumer turned down the offer on the table or pushed "
            "back on it. This is what earns the right to concede later; without it "
            "concede will refuse."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
    ),
    ToolSchema(
        name="concede",
        description=(
            "Move to the next arrangement down. Returns the new offer, which is the "
            "only one you may now describe. Fails if nothing has been refused."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "preferred_cadence": {
                    "type": "string",
                    "enum": _CADENCE_ENUM,
                    "description": "Cadence the consumer asked for, honoured where possible.",
                }
            },
            "required": [],
        },
    ),
    ToolSchema(
        name="confirm_agreement",
        description=(
            "The consumer has accepted the offer currently on the table. Closes the "
            "negotiation and returns the agreed schedule for you to read back."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
    ),
    ToolSchema(
        name="end_call",
        description="End the call without an agreement.",
        input_schema={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Short note on why the call is ending, for the log.",
                }
            },
            "required": [],
        },
    ),
)

TOOL_NAMES: frozenset[str] = frozenset(s.name for s in TOOL_SCHEMAS)


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


# -- argument coercion -----------------------------------------------------


def _money(value: object) -> Money:
    """Parse an amount from tool arguments.

    JSON has one number type and it decodes to ``float``, which ``Money``
    refuses on purpose (SPEC §9). Routing through ``str`` at this boundary
    keeps the exact digits the model actually emitted — ``str(50.5)`` is
    "50.5" — without letting a float any further into the system.
    """
    if isinstance(value, Money):
        return value
    if isinstance(value, float | int | str | Decimal):
        return Money(Decimal(str(value)))
    raise ValueError(f"expected an amount, got {type(value).__name__}")


def _cadence(value: object, default: Cadence = Cadence.MONTHLY) -> Cadence:
    if value is None:
        return default
    try:
        return Cadence(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"cadence must be one of {', '.join(_CADENCE_ENUM)}; got {value!r}"
        ) from exc


# Far above any legal payment count (the ladder tops out at 4), but low
# enough that Money.allocate() on it is still cheap. Rejected here, before
# ConsumerProposal.smallest_payment ever calls allocate() on the raw value —
# a payment_count in the millions took 3+ seconds to reject otherwise
# (ADVERSARIAL_TESTING.md M4).
_MAX_SANE_PAYMENT_COUNT = 1000


def _payment_count(value: object) -> int:
    if not isinstance(value, int | float | str | Decimal):
        raise ValueError(f"expected a payment count, got {type(value).__name__}")
    count = int(Decimal(str(value)))
    if count < 1:
        raise ValueError("payment_count must be at least 1")
    if count > _MAX_SANE_PAYMENT_COUNT:
        raise ValueError(f"payment_count must be at most {_MAX_SANE_PAYMENT_COUNT}")
    return count


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
    try:
        capacity_arg = args.get("signaled_capacity")
        total_arg = args.get("total")
        proposal = ConsumerProposal(
            # No sum named means they proposed a shape, not a discount: the
            # balance stands, and the structure is what gets ruled on.
            total=_money(total_arg) if total_arg is not None else context.policy.original_balance,
            payment_count=_payment_count(args["payment_count"]),
            cadence=_cadence(args.get("cadence")),
            signaled_capacity=_money(capacity_arg) if capacity_arg is not None else None,
        )
    except (KeyError, ValueError, TypeError, ArithmeticError, InvalidOperation) as exc:
        return _error(name, context, f"could not read the proposal: {exc}")

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
    try:
        cadence = _cadence(args.get("preferred_cadence"))
    except ValueError as exc:
        return _error(name, context, str(exc))

    offer = build_counter(context.state, context.policy, preferred_cadence=cadence)
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
    try:
        cadence = _cadence(args.get("preferred_cadence"))
    except ValueError as exc:
        return _error(name, context, str(exc))

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
    # for some earlier proposal (SPEC §4.1, §6).
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
    """Run one tool call. The only way the model can affect anything."""
    handler = _DISPATCH.get(call.name)
    if handler is None:
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
    return handler(call.arguments, context)


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
