"""Policy constants — the single source of truth.

Every limit is derived from the brief. Nothing here is duplicated in a prompt:
compliance rules live in code with stable ids, never as prose.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from collector.money import Money
from collector.offers import Cadence, Tier


@dataclass(frozen=True)
class PolicyConfig:
    original_balance: Money
    min_payment_pct: Decimal
    max_settlement_discount: Decimal
    max_plan_days: int
    max_settlement_payments: int
    allowed_cadences: frozenset[Cadence]
    # Caps how long the agent may keep haggling. Past this the call closes out:
    # an unbounded negotiation is badgering however politely it is phrased.
    max_negotiation_rounds: int = 8
    # Where the settlement tier *opens*, as against the most it may ever give
    # up. Necessarily smaller than ``max_settlement_discount``: a tier whose
    # first offer is already its floor has nothing left to concede, and the one
    # number we least want to disclose is the first one we say.
    opening_settlement_discount: Decimal = Decimal("0.10")

    @classmethod
    def default(cls) -> PolicyConfig:
        """The brief: $1,000 debt, 180+ days delinquent."""
        return cls(
            original_balance=Money("1000.00"),
            min_payment_pct=Decimal("0.25"),
            max_settlement_discount=Decimal("0.20"),
            max_plan_days=92,  # "over 3 months max"
            max_settlement_payments=3,
            allowed_cadences=frozenset(
                {Cadence.IMMEDIATE, Cadence.WEEKLY, Cadence.BIWEEKLY, Cadence.MONTHLY}
            ),
        )

    # -- derived limits ----------------------------------------------------

    @property
    def min_payment(self) -> Money:
        """A1 RESOLVED: 25% of the ORIGINAL BALANCE, not of the agreed total.

        Fixed at $250 for every tier. This is the strict reading and the only
        one implemented.
        """
        return self.original_balance * self.min_payment_pct

    @property
    def settlement_floor(self) -> Money:
        """Most we may discount is 20% off, so the least we may accept is $800."""
        return self.original_balance * (Decimal(1) - self.max_settlement_discount)

    @property
    def opening_settlement(self) -> Money:
        """The first settlement figure a consumer hears: $900, not the floor.

        Clamped to the floor so a misconfiguration can only ever ask for *more*
        than the floor, never authorize a deeper discount than
        ``max_settlement_discount`` allows.
        """
        opening = self.original_balance * (Decimal(1) - self.opening_settlement_discount)
        return max(opening, self.settlement_floor)

    @property
    def max_installments(self) -> int:
        """Falls out of the floor: $1,000 / $250 = 4, never more.

        This is why a 'weekly over 3 months' plan (~13 payments) is structurally
        impossible and must be countered rather than accommodated.
        """
        return int(self.original_balance.amount / self.min_payment.amount)

    def max_payments_for(self, tier: Tier) -> int:
        return {
            Tier.PAY_IN_FULL: 1,
            Tier.DOWNPAYMENT_PLUS_ONE: 2,
            Tier.SETTLEMENT: self.max_settlement_payments,
            Tier.PAYMENT_PLAN: self.max_installments,
        }[tier]

    def replace(self, **changes: Any) -> PolicyConfig:
        return dataclasses.replace(self, **changes)
