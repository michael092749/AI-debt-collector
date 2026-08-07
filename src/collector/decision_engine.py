"""The decision engine — SPEC §4. The graded core.

    The LLM talks. Deterministic code decides.

Pure by construction: no I/O, no LLM, no network, no clock. A function of
(proposal, state, policy) and nothing else, so it is testable as a table of
inputs and expected verdicts. The import-purity test in T6 enforces this
structurally rather than by convention.

Every verdict carries the full set of conditions evaluated — passing ones
included. That is deliberate: the research report's vendor test is "show me the
decision record; if they show you a transcript, the model decided; if they show
you evaluated conditions and a policy path, the engine did."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from collector.money import Money
from collector.negotiation import NegotiationState
from collector.offers import Cadence, ConsumerProposal, Installment, Offer, Tier
from collector.policy import PolicyConfig

Outcome = Literal["accept", "counter", "reject"]


class RuleId(StrEnum):
    """Stable identifiers. These appear in the audit log and must not churn."""

    TOTAL_FLOOR = "TOTAL_FLOOR"
    NO_UNAUTHORIZED_DISCOUNT = "NO_UNAUTHORIZED_DISCOUNT"
    NO_OVER_COLLECTION = "NO_OVER_COLLECTION"
    MIN_PAYMENT = "MIN_PAYMENT"
    PAYMENT_COUNT = "PAYMENT_COUNT"
    CADENCE = "CADENCE"
    MAX_DURATION = "MAX_DURATION"
    LADDER = "LADDER"


class RationaleCode(StrEnum):
    """Closed set. The LLM phrases these; it never invents one (SPEC §10)."""

    ACCEPTED = "ACCEPTED"
    BELOW_SETTLEMENT_FLOOR = "BELOW_SETTLEMENT_FLOOR"
    BELOW_MIN_PAYMENT = "BELOW_MIN_PAYMENT"
    TOO_MANY_PAYMENTS = "TOO_MANY_PAYMENTS"
    SCHEDULE_TOO_LONG = "SCHEDULE_TOO_LONG"
    CADENCE_NOT_OFFERED = "CADENCE_NOT_OFFERED"
    DISCOUNT_NOT_AUTHORIZED = "DISCOUNT_NOT_AUTHORIZED"
    PREFERRED_TIER_AVAILABLE = "PREFERRED_TIER_AVAILABLE"
    ABOVE_BALANCE_OWED = "ABOVE_BALANCE_OWED"


@dataclass(frozen=True)
class Condition:
    """One rule, evaluated. Self-describing so the audit trail needs no decoder."""

    rule_id: RuleId
    passed: bool
    actual: str
    limit: str


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    tier: Tier | None
    conditions: tuple[Condition, ...]
    counter: Offer | None
    rationale_code: RationaleCode


# -- tier classification ---------------------------------------------------


def classify(proposal: ConsumerProposal, policy: PolicyConfig) -> Tier:
    """Which tier is the consumer actually asking for?

    Classification is structural, not charitable: it reads what they proposed.
    Whether that proposal is *allowed* is the rules' job, below.
    """
    if proposal.total < policy.original_balance:
        return Tier.SETTLEMENT
    if proposal.payment_count <= 1:
        return Tier.PAY_IN_FULL
    if proposal.payment_count == 2:
        return Tier.DOWNPAYMENT_PLUS_ONE
    return Tier.PAYMENT_PLAN


# -- rule evaluation -------------------------------------------------------


def _evaluate(
    proposal: ConsumerProposal,
    tier: Tier,
    policy: PolicyConfig,
    ladder_floor: Tier,
    repeated: bool,
) -> tuple[Condition, ...]:
    max_payments = policy.max_payments_for(tier)
    smallest = proposal.smallest_payment

    return (
        Condition(
            RuleId.TOTAL_FLOOR,
            proposal.total >= policy.settlement_floor,
            str(proposal.total),
            f">= {policy.settlement_floor}",
        ),
        Condition(
            RuleId.NO_UNAUTHORIZED_DISCOUNT,
            tier is Tier.SETTLEMENT or proposal.total >= policy.original_balance,
            str(proposal.total),
            f">= {policy.original_balance} unless settlement",
        ),
        # The ceiling. SPEC §4.3 says the total is `== ORIGINAL_BALANCE` unless
        # the tier is settlement, and the rule above only ever enforced the
        # lower half of that equality — nothing anywhere capped it. A consumer
        # who misheard the balance and offered "a thousand a month for three
        # months" was *accepted* at $3,000 on a $1,000 debt, and $1,000,000
        # was accepted too.
        #
        # This is the worst shape a defect can take here: the model did not
        # originate the figure, the engine did. `you_may_say` published
        # "$3,000.00" as engine-authorized, so the numeric guard passed it
        # correctly and the agreement record was written. No guardrail above
        # the engine can catch an illegal number the engine itself authorized.
        Condition(
            RuleId.NO_OVER_COLLECTION,
            proposal.total <= policy.original_balance,
            str(proposal.total),
            f"<= {policy.original_balance}",
        ),
        Condition(
            RuleId.MIN_PAYMENT,
            smallest >= policy.min_payment,
            str(smallest),
            f">= {policy.min_payment}",
        ),
        Condition(
            RuleId.PAYMENT_COUNT,
            1 <= proposal.payment_count <= max_payments,
            str(proposal.payment_count),
            f"<= {max_payments} for {tier.label}",
        ),
        Condition(
            RuleId.CADENCE,
            proposal.cadence in policy.allowed_cadences,
            proposal.cadence.value,
            "one of " + ", ".join(sorted(c.value for c in policy.allowed_cadences)),
        ),
        Condition(
            RuleId.MAX_DURATION,
            proposal.duration_days <= policy.max_plan_days,
            f"{proposal.duration_days} days",
            f"<= {policy.max_plan_days} days",
        ),
        # Last, so a proposal that also breaks a real limit is answered on that
        # limit rather than on our preferences. This one is not a policy floor:
        # the terms are legal, they are simply further down the list than the
        # negotiation has reached — so we ask once for better, and take them at
        # their word when they hold to it.
        Condition(
            RuleId.LADDER,
            tier <= ladder_floor or repeated,
            tier.label,
            f"no worse than {ladder_floor.label}, or terms held to through a counter",
        ),
    )


_FAILURE_CODES: dict[RuleId, RationaleCode] = {
    RuleId.TOTAL_FLOOR: RationaleCode.BELOW_SETTLEMENT_FLOOR,
    RuleId.NO_UNAUTHORIZED_DISCOUNT: RationaleCode.DISCOUNT_NOT_AUTHORIZED,
    RuleId.NO_OVER_COLLECTION: RationaleCode.ABOVE_BALANCE_OWED,
    RuleId.MIN_PAYMENT: RationaleCode.BELOW_MIN_PAYMENT,
    RuleId.PAYMENT_COUNT: RationaleCode.TOO_MANY_PAYMENTS,
    RuleId.CADENCE: RationaleCode.CADENCE_NOT_OFFERED,
    RuleId.MAX_DURATION: RationaleCode.SCHEDULE_TOO_LONG,
    RuleId.LADDER: RationaleCode.PREFERRED_TIER_AVAILABLE,
}


def _first_failure(conditions: tuple[Condition, ...]) -> RationaleCode:
    """Rules are ordered most- to least-fundamental, so the first failure is the
    one worth explaining to the consumer."""
    for c in conditions:
        if not c.passed:
            return _FAILURE_CODES[c.rule_id]
    raise AssertionError("no failing condition")


# -- public entry point ----------------------------------------------------


def validate_offer(
    proposal: ConsumerProposal,
    state: NegotiationState,
    policy: PolicyConfig,
) -> Verdict:
    """Validate a consumer's proposal and, where it fails, counter it.

    This is the tool the agent calls mid-call. The agent phrases the result; it
    never computes one.

    Legal is not the same as agreeable. The tiers are a preference order, and a
    proposal below where the ladder currently stands is countered at the ladder
    rather than taken — otherwise the first thing a consumer says decides the
    outcome and there was never a negotiation (A7).

    Countered once, though, not forever. Someone who hears the better ask and
    repeats their own terms has said no to it, and their arrangement is legal;
    holding out for a tier they have twice declined loses an account we could
    have closed. It is also the only way down to a payment plan, since raising
    the ask from a settlement is not something the agent may do (``_concede``).
    """
    tier = classify(proposal, policy)
    conditions = _evaluate(
        proposal, tier, policy, state.ladder_floor, _held_to(proposal, tier, state, policy)
    )

    if all(c.passed for c in conditions):
        return Verdict(
            outcome="accept",
            tier=tier,
            conditions=conditions,
            counter=None,
            rationale_code=RationaleCode.ACCEPTED,
        )

    # A failing proposal always leaves with a counter. "No" on its own is not a
    # negotiation move, and the brief requires the counter to come from here.
    counter = build_counter(
        state,
        policy,
        capacity=effective_capacity(proposal, state),
        preferred_cadence=proposal.cadence,
    )
    failed = {c.rule_id for c in conditions if not c.passed}
    outcome: Outcome = "reject" if failed & _HARD_FLOORS else "counter"

    return Verdict(
        outcome=outcome,
        tier=None,
        conditions=conditions,
        counter=counter,
        rationale_code=_first_failure(conditions),
    )


# Violating one of these means the *amount* is unacceptable -> reject.
# Anything else means the amount is fine but the *structure* is wrong -> counter.
_HARD_FLOORS = frozenset({RuleId.TOTAL_FLOOR, RuleId.MIN_PAYMENT, RuleId.NO_UNAUTHORIZED_DISCOUNT})


def _held_to(
    proposal: ConsumerProposal, tier: Tier, state: NegotiationState, policy: PolicyConfig
) -> bool:
    """Have they put these terms up before and heard the better ask?

    Rounds are recorded *after* the verdict is computed (``tools.py``), so the
    first time terms are named this is False and they get countered; a consumer
    who holds to them across that counter finds it True on the way back.

    Two clauses do the real work, and both close a door back to the leapfrog:

    ``PREFERRED_TIER_AVAILABLE`` — the earlier round must have been countered
    for the ladder and nothing else. Otherwise an *illegal* proposal unlocks its
    tier for a later legal one: "can I pay $77 a week?" is a T4 shape, refused
    on the payment floor, and the "$250 a month" that follows would walk into
    the worst outcome with no counter ever made at the ladder.

    ``total >= ...`` — and they must have held to the terms, not sharpened them.
    Without it "settle at $900" answered by a counter becomes "$800, then", and
    the ladder would have talked us *down* $100.
    """
    return any(
        r.proposal is not None
        and classify(r.proposal, policy) is tier
        and r.rationale_code == RationaleCode.PREFERRED_TIER_AVAILABLE
        and proposal.total >= r.proposal.total
        for r in state.rounds
    )


def effective_capacity(proposal: ConsumerProposal, state: NegotiationState) -> Money:
    """What this proposal reveals the consumer thinks they can pay at a time.

    An explicit signal wins outright: it is the latest thing they actually said,
    and it may move the figure in either direction. Otherwise their own smallest
    instalment is the honest read — someone asking for 13 weekly payments of $77
    has told us $77 without ever saying a capacity out loud.

    But a figure we inferred may only *lower* one they stated. Inference reads a
    ceiling, never a floor: a lump sum they are asking us to accept says nothing
    about what they can produce, and where no sum was named at all the shape is
    scored against the balance, so the "capacity" is arithmetic on a default.
    Left unclamped, both talk over the number the consumer gave us — $300 said
    three times became a counter built on $500 — and the next counter comes back
    harder than the one they just refused.
    """
    if proposal.signaled_capacity is not None:
        return proposal.signaled_capacity
    inferred = proposal.smallest_payment
    if state.signaled_capacity is not None:
        return min(inferred, state.signaled_capacity)
    return inferred


# -- counter construction --------------------------------------------------


def _tier_total(tier: Tier, policy: PolicyConfig) -> Money:
    """Settlements are the only tier that discounts (A3)."""
    return policy.settlement_floor if tier is Tier.SETTLEMENT else policy.original_balance


def select_tier(state: NegotiationState, policy: PolicyConfig, capacity: Money | None) -> Tier:
    """The tier the ladder currently stands on, and only that one.

    Every tier below the floor is a concession that has not been earned yet, so
    there is nothing to choose between: the offer is the floor. ``capacity``
    shapes the schedule *inside* the tier — how large a downpayment, how many
    instalments — and is deliberately not allowed to choose the tier itself.

    It used to. The engine skipped any tier whose whole total could not be paid
    at the signalled capacity, which meant naming "$77 a week" on the first turn
    collected all three concessions at once and put the worst outcome on the
    table before anything had been refused. A tier the consumer cannot fund is
    answered by them refusing it, which earns exactly one step down (A7).
    """
    return state.ladder_floor


def _allocate(
    total: Money, parts: int, floor: Money, capacity: Money | None = None
) -> tuple[Money, ...]:
    """Split ``total`` into ``parts`` instalments that each clear ``floor``.

    A capacity they named out loud is taken at its word and leads the schedule:
    someone who says "$400" gets $400 now and $300/$300 after, not three even
    payments of $333.34. Money sooner is worth more than a tidy schedule, and
    it is the same instinct as maximizing the downpayment one tier up.

    Otherwise even - "three payments of $266.67" is easier to agree to than a
    lopsided schedule. Only when an even split would dip below the floor do we
    pin instalments at the floor and put the remainder on the last one.
    """
    even = total.allocate(parts)
    if capacity is not None and parts > 1 and capacity > even[0]:
        rest = (total - capacity).allocate(parts - 1)
        if min(rest) >= floor:
            return (capacity, *rest)
    if min(even) >= floor:
        return even
    remainder = total - floor * (parts - 1)
    return tuple([floor] * (parts - 1) + [remainder])


def _counter_cadence(preferred: Cadence, parts: int, policy: PolicyConfig) -> Cadence:
    """Honour the consumer's cadence preference where the calendar allows it."""
    if parts <= 1:
        return Cadence.IMMEDIATE
    candidate = preferred
    if candidate is Cadence.IMMEDIATE or candidate not in policy.allowed_cadences:
        candidate = Cadence.MONTHLY
    # Step to a tighter cadence if the schedule would run past the 3-month cap.
    for fallback in (candidate, Cadence.BIWEEKLY, Cadence.WEEKLY):
        if (parts - 1) * fallback.interval_days <= policy.max_plan_days:
            return fallback
    return Cadence.WEEKLY


def build_counter(
    state: NegotiationState,
    policy: PolicyConfig,
    *,
    tier: Tier | None = None,
    capacity: Money | None = None,
    payments: int | None = None,
    preferred_cadence: Cadence = Cadence.MONTHLY,
) -> Offer:
    """Compute the offer the agent should put back to the consumer.

    Deterministic and total: for any inputs this returns a policy-legal Offer.

    An explicit ``capacity`` wins, but with none given the state's own signal
    stands in. Without that fallback a concession can hand back terms harder
    than the ones just refused — the consumer says $300, we counter $300/$700,
    they refuse, and the "concession" comes back $750/$250.
    """
    capacity = capacity if capacity is not None else state.signaled_capacity
    tier = tier if tier is not None else select_tier(state, policy, capacity)
    total = _tier_total(tier, policy)
    amounts: tuple[Money, ...]

    if tier is Tier.DOWNPAYMENT_PLUS_ONE:
        # "Highest-amount downpayment": take everything they can put down now,
        # capped so the remaining payment still clears the floor.
        ceiling = total - policy.min_payment
        down = capacity if capacity is not None else ceiling
        down = min(max(down, policy.min_payment), ceiling)
        amounts = (down, total - down)
    else:
        headroom = int(total.amount / policy.min_payment.amount)
        allowed = min(policy.max_payments_for(tier), headroom)
        wanted = payments if payments is not None else _payments_for(total, capacity)
        parts = max(1, min(wanted, allowed))
        amounts = _allocate(total, parts, policy.min_payment, capacity)

    cadence = _counter_cadence(preferred_cadence, len(amounts), policy)
    installments = tuple(
        Installment(amount=a, due_day_offset=i * cadence.interval_days)
        for i, a in enumerate(amounts)
    )
    return Offer(tier=tier, installments=installments, cadence=cadence)


def _payments_for(total: Money, capacity: Money | None) -> int:
    """Fewest instalments that fit the stated capacity - fewer is higher value."""
    if capacity is None or capacity.amount <= 0:
        return 1
    return -(-int(total.amount * 100) // int(capacity.amount * 100))  # ceil division
