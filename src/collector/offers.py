"""Value objects for offers and consumer proposals (SPEC §2.2).

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
        return self.name.replace("_", " ").lower()


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
    def duration_days(self) -> int:
        return max(i.due_day_offset for i in self.installments)


@dataclass(frozen=True)
class ConsumerProposal:
    """What the consumer said they'd do. Unvalidated by construction."""

    total: Money
    payment_count: int
    cadence: Cadence
    signaled_capacity: Money | None = None

    @property
    def duration_days(self) -> int:
        return (self.payment_count - 1) * self.cadence.interval_days

    @property
    def even_installments(self) -> tuple[Money, ...]:
        return self.total.allocate(self.payment_count)

    @property
    def smallest_payment(self) -> Money:
        return min(self.even_installments)
