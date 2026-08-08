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
from collector.guardrails.confirmation import confirmation_line
from collector.llm.base import ToolCall
from collector.money import Money
from collector.negotiation import CallOutcome, NegotiationState
from collector.offers import Cadence, ConsumerProposal, Installment, Offer, Tier
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
                    "The one overall sum they offered to pay, as a decimal string, "
                    "e.g. '850.00'. Only for a sum they named as the whole amount. "
                    "If they sized the payments instead — 'two payments of five "
                    "hundred' — pass amount_each and leave this out; never multiply "
                    "it out yourself. Omit both when they proposed only a structure "
                    "and no figure — 'weekly for a year' — and the full balance is "
                    "assumed."
                ),
            ),
            Param(
                name="amount_each",
                kind=ParamKind.MONEY,
                description=(
                    "What they offered per payment, as a decimal string, when they "
                    "named the size of each payment rather than an overall sum. "
                    "Relay their figure verbatim; the engine does the multiplying. "
                    "Never pass both this and total."
                ),
            ),
            Param(
                name="payment_count",
                kind=ParamKind.COUNT,
                description=(
                    "How many payments they said to split it into. 1 for a lump "
                    "sum. Omit when they did not say — a bare figure like 'three "
                    "hundred dollars' names no count, and inventing one is what "
                    "turned $150 into a refusal for overpaying."
                ),
                minimum=1,
                maximum=MAX_PROPOSED_PAYMENTS,
            ),
            Param(
                name="cadence",
                kind=ParamKind.CADENCE,
                description="How often they proposed to pay, if they said.",
            ),
            Param(
                name="signaled_capacity",
                kind=ParamKind.MONEY,
                description=(
                    "The largest single payment they said they can make, as a "
                    "decimal string. Two shapes of sentence carry one, and both "
                    "belong here: a ceiling they named ('I can only manage four "
                    "hundred at a time'), and the first payment of a schedule "
                    "they laid out — 'four hundred today, and then six hundred "
                    "next month' signals 400. The second is the one that gets "
                    "missed, because the sentence also names a total and reads "
                    "as a finished plan rather than a limit. Their schedule is "
                    "built to start from this figure, so omitting it re-splits "
                    "their own plan evenly and asks them for more on the day "
                    "than they offered. Omit only when they named no figure."
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


def _proposed_cadence(args: JsonDict, payment_count: int) -> Cadence:
    """The rhythm they named, or the one their shape implies where they named none.

    ``cadence`` is no longer required: a consumer answering "what can you
    manage?" with "a hundred and fifty dollars" named no rhythm, and a model
    forced to supply one supplies a guess.
    """
    cadence = args.get("cadence")
    if isinstance(cadence, Cadence):
        return cadence
    return Cadence.MONTHLY if payment_count > 1 else Cadence.IMMEDIATE


def _schedule_for(each: Money, balance: Money) -> tuple[Money, int]:
    """The plan a per-payment figure describes: that much at a time until clear.

    The payment count that arrives beside a per-payment figure is the model's,
    not the consumer's — they said "a hundred and fifty dollars", not "seven
    payments" — so multiplying the two assembled a total nobody proposed and
    the engine then ruled on it. $150 x 7 is $1,050, which trips the
    over-collection ceiling, and the consumer was told their $150 would collect
    more than they owed (live call, 2026-08-07). Worse, the answer moved with
    the invented number: one utterance of "$300" produced three rationales and
    two outcomes depending on the count the model happened to pick.

    Deriving the count from the balance makes the verdict a function of the one
    figure the consumer actually said. A schedule that overshoots the balance is
    a plan whose last payment is trimmed, not an illegal proposal, so the total
    is the balance and never more.
    """
    if each.amount <= 0:
        return each, 1
    # ceil division in cents; Money is quantized, so these are whole numbers.
    covered = -(-int(balance.amount * 100) // int(each.amount * 100))
    return balance, max(1, covered)


def _consumer_led_schedule(
    proposal: ConsumerProposal, policy: PolicyConfig
) -> tuple[Money, ...] | None:
    """The schedule the consumer described, when they sized its first payment.

    ``even_installments`` is the right reading of "two payments" — a shape with
    no figures in it. It is the wrong reading of "four hundred today, and then
    six hundred next month", which has both: accepted as an even split that
    closed at $500 and $500, $100 more due on the day than was offered (live
    call, 2026-08-08). The engine's own ``_allocate`` already leads a *counter*
    with a capacity named out loud; this is the same rule for the proposal we
    are agreeing to.

    ``None`` whenever their split cannot be written as a legal schedule, and the
    even one stands. Lives here rather than in ``Offer.from_proposal`` because
    the payment floor decides that, and ``offers.py`` cannot see the policy —
    the engine ruled MIN_PAYMENT against ``smallest_payment``, which is the even
    split's figure, so a front-loaded schedule is unchecked until here. "$800
    today and the rest next month" leaves $200 against a $250 floor.
    """
    down = proposal.signaled_capacity
    if down is None or proposal.payment_count < 2 or down < policy.min_payment:
        return None
    remainder = proposal.total - down
    if remainder.amount <= 0:
        return None
    rest = remainder.allocate(proposal.payment_count - 1)
    if min(rest) < policy.min_payment:
        return None
    return (down, *rest)


def _consumer_proposal(args: JsonDict, policy: PolicyConfig) -> ConsumerProposal:
    """What the consumer proposed, out of whichever way the model relayed it.

    Three channels, in descending order of how directly the consumer authored
    the number: a total they named, a per-payment figure they named, and
    nothing at all — in which case the balance stands and only the structure is
    in question. Only the first is a total the consumer uttered, and only that
    one may be ruled on as one (``ConsumerProposal.total_stated``).

    A capacity with no sum beside it belongs to the second channel: "I can only
    do three hundred" sizes the payments, it does not offer $300 for the debt.
    Read as a total it made the engine accept the full balance today on behalf
    of a consumer who had just said they could not manage it — an agreement the
    engine authored, so nothing above the engine could have caught it.
    """
    stated_total: Money | None = args.get("total")
    signaled: Money | None = args.get("signaled_capacity")
    each: Money | None = args.get("amount_each")
    if each is None and stated_total is None:
        each = signaled

    if stated_total is not None:
        # Membership, not truthiness. "They proposed $0" and "they named no sum
        # at all" are different negotiation states — the first is a lowball to
        # rule on, the second means the balance stands — and a falsy-zero would
        # silently merge them.
        count: int = args.get("payment_count", 1)
        return ConsumerProposal(
            total=stated_total,
            payment_count=count,
            cadence=_proposed_cadence(args, count),
            signaled_capacity=signaled,
        )

    if each is not None:
        total, count = _schedule_for(each, policy.original_balance)
        cadence = _proposed_cadence(args, count)
        if count > 1 and cadence is Cadence.IMMEDIATE:
            # They named the rhythm for the lump sum the model assumed. A
            # multi-payment schedule cannot be immediate: it would fall due in
            # its entirety today.
            cadence = Cadence.MONTHLY
        return ConsumerProposal(
            total=total,
            payment_count=count,
            cadence=cadence,
            signaled_capacity=signaled,
            amount_each=each,
            total_stated=False,
        )

    count = args.get("payment_count", 1)
    return ConsumerProposal(
        total=policy.original_balance,
        payment_count=count,
        cadence=_proposed_cadence(args, count),
        signaled_capacity=signaled,
        total_stated=False,
    )


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
    # Two ways to name the money, because that is how it arrives: "eight
    # fifty all in" is a total, "two payments of five hundred" is a size of
    # each. The model relays whichever was said verbatim — it is forbidden
    # arithmetic — so the multiplication happens here. Both at once is a
    # contradiction to send back, not a guess to make.
    if "amount_each" in args and "total" in args:
        return _error(
            name,
            context,
            "total and amount_each together are ambiguous: pass the one figure "
            "the consumer actually said, and let the engine do the rest",
        )
    proposal = _consumer_proposal(args, context.policy)

    verdict = validate_offer(proposal, context.state, context.policy)

    # An accepted proposal becomes the standing offer: it is what the consumer
    # will be asked to confirm, so it has to be a real schedule and not a
    # remembered sentence.
    accepted = (
        Offer.from_proposal(proposal, verdict.tier)
        if verdict.outcome == "accept" and verdict.tier is not None
        else None
    )
    # Their own front-loading, where they sized the first payment themselves.
    consumer_led = False
    if accepted is not None:
        schedule = _consumer_led_schedule(proposal, context.policy)
        if schedule is not None and schedule != tuple(i.amount for i in accepted.installments):
            interval = accepted.cadence.interval_days
            accepted = Offer(
                tier=accepted.tier,
                installments=tuple(
                    Installment(amount, i * interval) for i, amount in enumerate(schedule)
                ),
                cadence=accepted.cadence,
            )
            consumer_led = True
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
    #
    # And only over a split we assembled. The mis-statement reading rests on the
    # instalments being ours — "two payments" is a shape, and the $500s that
    # come back from it are arithmetic on the balance, so there is nothing of
    # the consumer's to overwrite. A per-payment figure they said out loud is a
    # proposal of their own (``ConsumerProposal.amount_each``), and answering it
    # with a schedule they never named agrees to something nobody offered:
    # "two payments of five hundred each" was confirmed back as $250 today and
    # $750 in thirty days (live call, 2026-08-08).
    standing = context.standing_offer
    if (
        accepted is not None
        and proposal.amount_each is None
        and not consumer_led
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
    if verdict.outcome != "accept":
        # They named terms we cannot meet; that is a refusal of ours in all but
        # wording, and it is what makes a later concession legitimate.
        state = state.record_refusal()
        # And a refusal has to buy something. Recording one without ever moving
        # the ladder was the whole of the effect: `advance_ladder` was reachable
        # only through `concede`, so a consumer could put the same terms up
        # eight times, be refused eight times, and hear pay-in-full every time —
        # the tiers below it existed on paper only.
        #
        # Held-to terms are the trigger, not any refusal. Someone who repeats an
        # arrangement we have already answered has heard our counter and said no
        # to it, which is a refusal in all but wording and earns exactly one step
        # (A7) — the same reading `_held_to` takes of a repeat in the engine.
        # A consumer naming *new* terms is still negotiating, and the model's
        # own `concede` is what answers that; stepping there too would spend two
        # tiers on one exchange.
        standing = context.standing_offer
        if standing is not None and _first_repeat_of(context.state, proposal):
            state, stepped, moved = _step_down(state, context.policy, standing, proposal.cadence)
            if moved:
                on_table = stepped
    state = state.record_round(proposal, on_table, verdict.rationale_code)

    payload: JsonDict = {
        "ok": True,
        "outcome": verdict.outcome,
        "rationale_code": verdict.rationale_code.value,
        "tier": verdict.tier.label if verdict.tier else None,
        "conditions": [to_jsonable(c) for c in verdict.conditions],
        "offer_on_the_table": _offer_payload(on_table),
        "you_may_say": _sayable(on_table),
        "you_must_confirm": _confirmation(on_table),
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
            "you_must_confirm": _confirmation(offer),
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
    standing = context.standing_offer
    state, offer, moved = _step_down(context.state, context.policy, standing, _cadence_arg(args))
    # Nothing better was available, so the offer on the table is the one that
    # was already there. Reporting the candidate instead would have the agent
    # read back terms it is not authorized to agree to — and confirm_agreement
    # would then close on the standing offer, not the one just described.
    on_table = offer if moved else standing
    # The round is recorded whether or not the ladder moved: a non-concession is
    # still an exchange, and once the ladder is bottomed out every further
    # refusal would otherwise never advance the round count, leaving the round
    # cap unreachable and the call unable to close (badgering with no backstop).
    state = state.record_round(None, on_table, None)
    return ToolResult(
        name=name,
        payload={
            "ok": True,
            "moved": moved,
            "offer_on_the_table": _offer_payload(on_table),
            "you_may_say": _sayable(on_table),
            "you_must_confirm": _confirmation(on_table),
        },
        context=context._with(state),
        offer=on_table,
    )


# Bottom of the ladder: there is nothing below a payment plan to step down to.
_LAST_TIER = Tier.PAYMENT_PLAN


def _first_repeat_of(state: NegotiationState, proposal: ConsumerProposal) -> bool:
    """Are these exact terms being put to us again, for the first time?

    Rounds are recorded *after* this is consulted, so the count is of earlier
    utterances only: nothing on a first hearing, one on the first repeat, more
    after that.

    Answering every repeat alike is what made the ladder a free elevator. A
    consumer who says "a hundred dollars" four times was walked from pay-in-full
    to the bottom of the ladder without ever moving, and the settlement floor --
    the most we are authorized to discount -- was spoken on the third utterance
    of the same number.

    One repeat still buys its step: they heard the counter and said no to it,
    which is a refusal in all but wording and earns exactly one (A7). The second
    identical repeat says nothing the first did not, and a concession ladder
    that steps per turn rather than per counterparty move is the standard shape
    a held position is designed to exploit. Refusals still accumulate either
    way, so ``concede`` still has something to spend when the model judges the
    negotiation has genuinely moved.
    """
    return sum(1 for r in state.rounds if r.proposal == proposal) == 1


def _step_down(
    state: NegotiationState,
    policy: PolicyConfig,
    standing: Offer | None,
    cadence: Cadence,
) -> tuple[NegotiationState, Offer, bool]:
    """Walk the ladder down until the terms actually improve for the consumer.

    Shared by ``concede`` and by the refusal recorded inside
    ``validate_consumer_offer``, so both inherit the same two safeguards rather
    than one of them getting a second, subtly different ladder walk.

    Two ways a step can fail to be a concession: the offer comes back identical
    (capacity had already pushed the engine past that tier), or it comes back
    worse. The loop steps past both while there is ladder left.

    When no step improved on what was already up, the *input* state is returned.
    ``advance_ladder`` spends a refusal and moves the floor, and committing that
    would consume the consumer's refusal for zero benefit; kept pristine, it is
    still on record for a concession that actually helps them
    (ADVERSARIAL_TESTING.md L1).
    """
    stepped = state.advance_ladder()
    offer = build_counter(stepped, policy, preferred_cadence=cadence)
    while (
        not _is_concession(offer, standing)
        and stepped.can_concede
        and stepped.ladder_floor is not _LAST_TIER
    ):
        stepped = stepped.advance_ladder()
        offer = build_counter(stepped, policy, preferred_cadence=cadence)

    moved = _is_concession(offer, standing)
    return (stepped if moved else state), offer, moved


def _is_concession(offer: Offer, standing: Offer | None) -> bool:
    """Is ``offer`` genuinely easier for the consumer than what they refused?

    On the same rung, easier means less money, or the same money in smaller
    pieces.

    Across rungs the raw total is the wrong measure, and comparing it was why
    the bottom of the ladder was dead code: a settlement is $800 and a payment
    plan is the full $1,000, so stepping from one to the other never read as
    giving ground and ``concede`` returned ``moved: False`` for ever. But the
    brief ranks a payment plan *below* a settlement precisely because it costs
    more and is still worth holding back — it is easier to fund, in smaller
    pieces and over more time. So a step down the preference order is the
    concession, provided it lands somewhere genuinely easier: less to produce at
    one time, or less money overall.

    Kept honest at the edges. A step that improves neither is lateral or worse
    and reports no movement, and a step back *up* the order is never a
    concession however cheap it looks (A4).
    """
    if standing is None:
        return True
    if offer.tier is standing.tier:
        if offer.total != standing.total:
            return offer.total < standing.total
        return offer.smallest_payment < standing.smallest_payment
    if offer.tier < standing.tier:
        return False
    return offer.largest_payment < standing.largest_payment or offer.total < standing.total


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
                # The terms being closed on, carried so the ruling that
                # authorized them reaches the `decisions` table. Without it the
                # agent loop's `result.proposal is not None` gate drops the
                # write, and a call that *agreed* could leave `decisions`
                # empty — the one query an auditor is most likely to run.
                proposal=_as_proposal(agreed),
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
        # See the replay branch above: the agreed terms are the proposal the
        # engine just ruled on, and the decision record is the deliverable.
        proposal=_as_proposal(offer),
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
        # The only name for this arrangement the agent may say out loud. The
        # label above is an internal identifier and is not sayable; handed only
        # that, the model invented a name and picked another tier's.
        "call_it": offer.tier.spoken_name,
        "total": str(offer.total.amount),
        "payment_count": offer.payment_count,
        "cadence": offer.cadence.value,
        "schedule": [
            {"amount": str(i.amount.amount), "due_in_days": i.due_day_offset}
            for i in offer.installments
        ],
        "duration_days": offer.duration_days,
    }


def _confirmation(offer: Offer | None) -> str | None:
    """The line that has to be said before these terms can be agreed to.

    Handed over with the offer rather than only at the point of refusal, so the
    model reads it while it is deciding what to say — the gate in ``agent.py``
    is then a backstop for the turn that skipped it, not the first the model
    hears of the requirement.
    """
    return None if offer is None else confirmation_line(offer)


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
