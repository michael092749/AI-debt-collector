"""Negotiation state — the ladder, the concessions, and the record of both.

Division of labour with the decision engine: the engine rules on whether a
given arrangement is *legal*; this module governs how fast the agent may give
ground and preserves what was exchanged. Both are deterministic, and neither
lets the model decide.

Every transition returns a new state. Nothing mutates in place, so a dropped
call replays exactly and the audit trail cannot drift from what happened.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum

from collector.money import Money
from collector.offers import ConsumerProposal, Offer, Tier
from collector.policy import PolicyConfig


class CallOutcome(StrEnum):
    """How the call ended. Defined once here and imported by the audit layer.

    ``ABANDONED`` is never reached by a transition on this class — a call that
    drops mid-negotiation leaves its state at ``IN_PROGRESS`` forever, and only
    the audit layer, closing the record out, can say the call ended that way.
    """

    IN_PROGRESS = "in_progress"
    AGREED = "agreed"
    ESCALATED = "escalated"
    NO_AGREEMENT = "no_agreement"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class Round:
    """One exchange: what they proposed, what we put back, and why."""

    proposal: ConsumerProposal | None
    counter: Offer | None
    rationale_code: str | None


@dataclass(frozen=True)
class NegotiationState:
    """What the engine needs to know about the call so far.

    ``ladder_floor`` enforces A4 monotonicity: the best tier still on the table.
    The agent walks down T1 -> T4 and never back up, so a consumer cannot talk
    their way into a settlement and then re-open at pay-in-full terms, nor can
    the model concede faster than the ladder allows.
    """

    ladder_floor: Tier
    signaled_capacity: Money | None
    offers_made: tuple[Offer, ...]
    rounds: tuple[Round, ...] = ()
    outcome: CallOutcome = CallOutcome.IN_PROGRESS
    agreed_offer: Offer | None = None
    escalation_reason: str | None = None
    pending_refusals: int = 0
    max_rounds: int = 8

    # -- construction ------------------------------------------------------

    @classmethod
    def opening(cls, policy: PolicyConfig) -> NegotiationState:
        return cls(
            ladder_floor=Tier.PAY_IN_FULL,
            signaled_capacity=None,
            offers_made=(),
            max_rounds=policy.max_negotiation_rounds,
        )

    # -- status ------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.outcome is not CallOutcome.IN_PROGRESS

    @property
    def round_count(self) -> int:
        return len(self.rounds)

    @property
    def is_exhausted(self) -> bool:
        """Past the round cap. An unbounded haggle becomes badgering, which is
        a compliance problem regardless of how politely it is phrased."""
        return self.round_count >= self.max_rounds

    @property
    def can_concede(self) -> bool:
        """Concessions are earned. Without a refusal on the record the agent is
        giving away money for nothing."""
        return self.pending_refusals > 0

    # -- transitions -------------------------------------------------------

    def _step(self, **changes: object) -> NegotiationState:
        if self.is_terminal:
            raise ValueError(f"call is terminal ({self.outcome.value}); no further negotiation")
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]

    def with_capacity(self, capacity: Money | None) -> NegotiationState:
        """Consumers revise what they can afford; the latest statement wins."""
        return self._step(signaled_capacity=capacity)

    def record_refusal(self) -> NegotiationState:
        return self._step(pending_refusals=self.pending_refusals + 1)

    def record_round(
        self,
        proposal: ConsumerProposal | None,
        counter: Offer | None,
        rationale_code: str | None,
    ) -> NegotiationState:
        return self._step(
            rounds=(*self.rounds, Round(proposal, counter, rationale_code)),
            offers_made=(*self.offers_made, counter) if counter else self.offers_made,
        )

    def advance_ladder(self) -> NegotiationState:
        """Step down exactly one tier, spending one refusal.

        Silently a no-op when nothing has been refused or when already at the
        bottom — holding position is a legitimate move, not an error.
        """
        if self.is_terminal:
            raise ValueError(f"call is terminal ({self.outcome.value}); no further negotiation")
        if not self.can_concede or self.ladder_floor is Tier.PAYMENT_PLAN:
            return self
        return dataclasses.replace(
            self,
            ladder_floor=Tier(self.ladder_floor + 1),
            pending_refusals=self.pending_refusals - 1,
        )

    def spend_refusal(self) -> NegotiationState:
        """Consume a refusal while holding the tier.

        Movement does not have to cost a rung. A settlement that was put up and
        turned down re-prices at ``settlement_floor``, which is a real
        concession made *inside* the tier — the consumer earned it, so the
        refusal is spent, but the ladder keeps its position and the tiers below
        stay unspent.

        A no-op with nothing on record, matching ``advance_ladder``: giving
        ground for free is the thing ``can_concede`` exists to prevent.
        """
        if not self.can_concede:
            return self
        return self._step(pending_refusals=self.pending_refusals - 1)

    def conceded_to(self, tier: Tier) -> NegotiationState:
        """Move the ladder down. Never back up (A4)."""
        return self._step(ladder_floor=max(self.ladder_floor, tier))

    # -- terminal transitions ---------------------------------------------

    def agree(self, offer: Offer) -> NegotiationState:
        return self._step(outcome=CallOutcome.AGREED, agreed_offer=offer)

    def end_without_agreement(self) -> NegotiationState:
        return self._step(outcome=CallOutcome.NO_AGREEMENT)

    def escalate(self, reason: str) -> NegotiationState:
        """Hand off to a human.

        Deliberately not guarded by ``_step``: escalation is the safety valve
        and must remain reachable from any state. Per A6 it also ends the call,
        and it never carries an agreement — pressing someone who disputed the
        debt is precisely what the guardrails exist to stop.
        """
        return dataclasses.replace(
            self,
            outcome=CallOutcome.ESCALATED,
            escalation_reason=reason,
            agreed_offer=None,
        )
