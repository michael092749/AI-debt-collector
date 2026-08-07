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


class RationaleCode(StrEnum):
    """Closed set. The LLM phrases these; it never invents one (SPEC §10)."""

    ACCEPTED = "ACCEPTED"
    BELOW_SETTLEMENT_FLOOR = "BELOW_SETTLEMENT_FLOOR"
    BELOW_MIN_PAYMENT = "BELOW_MIN_PAYMENT"
    TOO_MANY_PAYMENTS = "TOO_MANY_PAYMENTS"
    SCHEDULE_TOO_LONG = "SCHEDULE_TOO_LONG"
    CADENCE_NOT_OFFERED = "CADENCE_NOT_OFFERED"
    DISCOUNT_NOT_AUTHORIZED = "DISCOUNT_NOT_AUTHORIZED"
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
    proposal: ConsumerProposal, tier: Tier, policy: PolicyConfig
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
    )


_FAILURE_CODES: dict[RuleId, RationaleCode] = {
    RuleId.TOTAL_FLOOR: RationaleCode.BELOW_SETTLEMENT_FLOOR,
    RuleId.NO_UNAUTHORIZED_DISCOUNT: RationaleCode.DISCOUNT_NOT_AUTHORIZED,
    RuleId.NO_OVER_COLLECTION: RationaleCode.ABOVE_BALANCE_OWED,
    RuleId.MIN_PAYMENT: RationaleCode.BELOW_MIN_PAYMENT,
    RuleId.PAYMENT_COUNT: RationaleCode.TOO_MANY_PAYMENTS,
    RuleId.CADENCE: RationaleCode.CADENCE_NOT_OFFERED,
    RuleId.MAX_DURATION: RationaleCode.SCHEDULE_TOO_LONG,
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
    """
    tier = classify(proposal, policy)
    conditions = _evaluate(proposal, tier, policy)

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
        capacity=effective_capacity(proposal),
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


def effective_capacity(proposal: ConsumerProposal) -> Money:
    """What this proposal reveals the consumer thinks they can pay at a time.

    An explicit signal wins; otherwise their own smallest instalment is the
    honest read. Someone asking for 13 weekly payments of $77 has told us $77,
    even though they never said a capacity out loud.
    """
    return proposal.signaled_capacity or proposal.smallest_payment


# -- counter construction --------------------------------------------------


def _tier_total(tier: Tier, policy: PolicyConfig) -> Money:
    """Settlements are the only tier that discounts (A3)."""
    return policy.settlement_floor if tier is Tier.SETTLEMENT else policy.original_balance


def _feasible(tier: Tier, capacity: Money | None, policy: PolicyConfig) -> bool:
    if capacity is None:
        return True  # no signal: stay optimistic and ask for the best tier left
    if tier is Tier.PAY_IN_FULL:
        return capacity >= policy.original_balance
    return capacity >= policy.min_payment


def select_tier(state: NegotiationState, policy: PolicyConfig, capacity: Money | None) -> Tier:
    """Highest-preference tier still both feasible and permitted.

    Never better than ``ladder_floor`` - concessions only move one way (A4).
    If nothing is feasible the consumer cannot meet the floor at all; we return
    the most affordable legal structure and let the agent say so plainly rather
    than inventing something below policy.
    """
    for tier in Tier:
        if tier >= state.ladder_floor and _feasible(tier, capacity, policy):
            return tier
    return Tier.PAYMENT_PLAN


def _allocate(total: Money, parts: int, floor: Money) -> tuple[Money, ...]:
    """Split ``total`` into ``parts`` instalments that each clear ``floor``.

    Even where possible - "three payments of $266.67" is easier to agree to
    than a lopsided schedule. Only when an even split would dip below the floor
    do we pin instalments at the floor and put the remainder on the last one.
    """
    even = total.allocate(parts)
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
        amounts = _allocate(total, parts, policy.min_payment)

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
