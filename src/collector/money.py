"""Exact money. Floats are a bug in this codebase (SPEC §9)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

CENTS = Decimal("0.01")


@dataclass(frozen=True, order=True, init=False)
class Money:
    """A Decimal amount quantized to cents.

    Constructing from ``float`` raises: binary floats cannot represent cents
    exactly, and a half-cent of drift in a payment schedule is a compliance
    defect, not a rounding nit.
    """

    amount: Decimal

    def __init__(self, value: Decimal | int | str) -> None:
        if isinstance(value, float):
            raise TypeError(
                f"Money rejects float ({value!r}); use Decimal, int, or str. SPEC §9."
            )
        object.__setattr__(self, "amount", Decimal(value).quantize(CENTS, rounding=ROUND_HALF_UP))

    def __str__(self) -> str:
        return f"${self.amount:,.2f}"

    def __repr__(self) -> str:
        return f"Money('{self.amount}')"

    def __add__(self, other: Money) -> Money:
        return Money(self.amount + other.amount)

    def __radd__(self, other: Money | int) -> Money:
        # Supports sum(), which seeds with int 0.
        if other == 0:
            return self
        raise TypeError("Money can only be summed with Money")

    def __sub__(self, other: Money) -> Money:
        return Money(self.amount - other.amount)

    def __mul__(self, factor: Decimal | int) -> Money:
        if isinstance(factor, float):
            raise TypeError("Money cannot be scaled by a float. SPEC §9.")
        return Money(self.amount * Decimal(factor))

    def __neg__(self) -> Money:
        return Money(-self.amount)

    def ratio_to(self, other: Money) -> Decimal:
        return self.amount / other.amount

    def allocate(self, parts: int) -> tuple[Money, ...]:
        """Split into ``parts`` amounts that sum back to exactly this value.

        Largest-remainder: the leftover cents land on the earliest installments,
        so the schedule is front-loaded rather than short by a cent.
        """
        if parts < 1:
            raise ValueError("parts must be >= 1")
        total_cents = int(self.amount / CENTS)
        base, leftover = divmod(total_cents, parts)
        return tuple(
            Money(Decimal(base + (1 if i < leftover else 0)) * CENTS) for i in range(parts)
        )

    @classmethod
    def zero(cls) -> Money:
        return cls(0)
