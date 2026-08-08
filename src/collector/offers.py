"""Value objects for offers and consumer proposals.

Schedules carry day *offsets* from the call, never absolute dates: the decision
engine must stay clock-free so it is testable as a pure input/output table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from collector.money import Money


class Tier(IntEnum):
    """Outcome tiers in the brief's preference order. Lower value = preferred."""

    PAY_IN_FULL = 1
    DOWNPAYMENT_PLUS_ONE = 2
    SETTLEMENT = 3
    PAYMENT_PLAN = 4

    @property
    def label(self) -> str:
        """The internal name. For the audit trail and the engine's own payloads."""
        return self.name.replace("_", " ").lower()

    @property
    def spoken_name(self) -> str:
        """What this arrangement is called out loud.

        ``label`` is derived from the enum member, which makes it exact and
        makes it unsayable: nobody says "downpayment plus one". Handed that as
        the only name for the offer it had just authored, the model reached for
        the natural English words and called it "a payment plan" — the label of
        a different, worse tier — while the decision record said
        DOWNPAYMENT_PLUS_ONE (live call, 2026-08-07). The transcript and the
        record disagreeing about what was offered is the one thing the decision
        record exists to prevent, so the sayable name is defined here, once,
        beside the tier it names.
        """
        return {
            Tier.PAY_IN_FULL: "paying the balance in full",
            Tier.DOWNPAYMENT_PLUS_ONE: "a payment today and one more after it",
            Tier.SETTLEMENT: "a reduced settlement",
            Tier.PAYMENT_PLAN: "a payment plan",
        }[self]


class Cadence(StrEnum):
    IMMEDIATE = "immediate"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"

    @property
    def interval_days(self) -> int:
        return {"immediate": 0, "weekly": 7, "biweekly": 14, "monthly": 30}[self.value]


@dataclass(frozen=True)
class Installment:
    amount: Money
    due_day_offset: int  # days from the call; 0 = today


@dataclass(frozen=True)
class Offer:
    """A fully-structured, engine-authored offer."""

    tier: Tier
    installments: tuple[Installment, ...]
    cadence: Cadence

    @property
    def total(self) -> Money:
        return sum((i.amount for i in self.installments), Money.zero())

    @property
    def payment_count(self) -> int:
        return len(self.installments)

    @property
    def smallest_payment(self) -> Money:
        return min(i.amount for i in self.installments)

    @property
    def largest_payment(self) -> Money:
        """The most the consumer has to produce at one time.

        The binding figure when comparing two arrangements for how hard they
        are to fund: a schedule is affordable at the payment they cannot make,
        not at the one they can.
        """
        return max(i.amount for i in self.installments)

    @property
    def duration_days(self) -> int:
        return max(i.due_day_offset for i in self.installments)

    @classmethod
    def from_proposal(cls, proposal: ConsumerProposal, tier: Tier) -> Offer:
        """Give an accepted proposal a schedule, so it can be agreed to and logged.

        Only meaningful once the engine has ruled the proposal legal — it is
        the consumer's own arrangement written down, not a new offer. The split
        is the even one the engine validated, on the cadence they asked for.
        """
        interval = proposal.cadence.interval_days
        return cls(
            tier=tier,
            installments=tuple(
                Installment(amount, i * interval)
                for i, amount in enumerate(proposal.even_installments)
            ),
            cadence=proposal.cadence,
        )


@dataclass(frozen=True)
class ConsumerProposal:
    """What the consumer said they'd do. Unvalidated by construction.

    ``total`` is the sum being ruled on, but it is not always a sum the
    consumer uttered: where they sized the payments instead, or named nothing
    at all, the tool layer assembles one. The two provenance fields record
    which, because a figure the consumer named and a figure we derived are not
    interchangeable evidence:

    ``amount_each`` — the per-payment figure they actually said. It is what the
    payment floor must be answered on, so the condition trail quotes their
    number rather than an even split of a total they never named. Relaying
    "$150" with an invented count of 7 was ruled on $1,050 and answered "that
    is more than you owe" (live call, 2026-08-07).

    ``total_stated`` — False when the tool layer supplied the total. An
    assembled total is arithmetic on a shape and may not stand in for money the
    consumer said they could produce.
    """

    total: Money
    payment_count: int
    cadence: Cadence
    signaled_capacity: Money | None = None
    amount_each: Money | None = None
    total_stated: bool = True

    @property
    def duration_days(self) -> int:
        return (self.payment_count - 1) * self.cadence.interval_days

    @property
    def even_installments(self) -> tuple[Money, ...]:
        return self.total.allocate(self.payment_count)

    @property
    def smallest_payment(self) -> Money:
        """The figure they would actually have to produce at one time.

        Their own per-payment figure when they gave one — an even split of an
        assembled total puts a number nobody said into the decision record.
        Safe against under-reading the floor: the count is derived as
        ``ceil(balance / amount_each)``, so the even split can only be smaller
        than ``amount_each`` when it is still at or above the payment floor.
        """
        if self.amount_each is not None:
            return self.amount_each
        return min(self.even_installments)

    @property
    def lump_sum(self) -> Money | None:
        """A total they named and offered to hand over in one go, if that is
        what this is.

        Money they said they could produce on the day, rather than a figure
        read off a shape — but only some of the time, which is
        ``decision_engine.effective_capacity``'s call to make, not this one's.
        """
        return self.total if self.total_stated and self.payment_count == 1 else None
