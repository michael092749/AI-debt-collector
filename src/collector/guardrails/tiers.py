"""Tier naming — the agent may not call an offer by another tier's name.

The numeric guard settles what *figures* the agent may say. Nothing settled what
it may call the arrangement those figures make up, and the two are not the same
control: every amount in the sentence below was engine-authored and passed the
numeric guard verbatim, while the offer standing on the table was
``DOWNPAYMENT_PLUS_ONE`` (live call, 2026-08-07).

    "I can offer a payment plan for the one thousand dollars. It would be two
    monthly payments, starting with two hundred fifty dollars today and then
    seven hundred fifty dollars in thirty days."

"payment plan" is the brief's fourth acceptable outcome — no discount, up to
three months, four instalments. The consumer was offered the second. She can
agree to something she was never offered, and the decision record will disagree
with the transcript about which arrangement she agreed to. That disagreement is
the precise failure the decision record exists to prevent, so it is closed here
with a stable rule id rather than in the prompt.

Deliberately narrow, in the same spirit as the disclosure detectors: only the
tier's own names count, and only while an offer of a *different* tier is
standing. Naming a tier that is not on the table is caught for the same reason
the prompt already forbids hinting at unearned concessions — the agent has no
business floating an arrangement the engine has not authorized.
"""

from __future__ import annotations

import re
from enum import StrEnum

from collector.guardrails.prohibited import Severity, Violation
from collector.offers import Tier


class TierRuleId(StrEnum):
    """Stable identifiers. These appear in the audit log and must not churn."""

    TIER_MISNAMED = "TIER_MISNAMED"


def _names(tier: Tier) -> tuple[str, ...]:
    """Every phrase that identifies ``tier`` to a listener.

    Both the internal label and the sayable name, because the model sees the
    label in the tool payload and has been observed reading it back.
    """
    return (tier.label, tier.spoken_name)


# One pattern per tier, matched on word boundaries so "settlement" does not fire
# inside "settlements" handling text and "pay in full" does not fire on
# "repay in full".
_TIER_PATTERNS: dict[Tier, re.Pattern[str]] = {
    tier: re.compile(
        "|".join(rf"\b{re.escape(name)}\b" for name in _names(tier)),
        re.IGNORECASE,
    )
    for tier in Tier
}


def scan_tier_names(candidate: str, standing: Tier | None) -> tuple[Violation, ...]:
    """Names of tiers other than the one currently on the table.

    ``standing`` is ``None`` before any offer has been authored, and nothing is
    flagged then: there is no arrangement to misname yet, and the opening turns
    have to be able to say the word "payment" without a tier attached to it.
    """
    if standing is None:
        return ()

    violations: list[Violation] = []
    for tier, pattern in _TIER_PATTERNS.items():
        if tier is standing:
            continue
        match = pattern.search(candidate)
        if match is None:
            continue
        violations.append(
            Violation(
                rule_id=TierRuleId.TIER_MISNAMED,
                severity=Severity.BLOCK,
                span=match.group(0),
                start=match.start(),
                end=match.end(),
                detail=(
                    f"{TierRuleId.TIER_MISNAMED}: called the offer on the table "
                    f"{match.group(0)!r}, which names {tier.label}; the standing offer is "
                    f"{standing.label} and is called {standing.spoken_name!r}"
                ),
            )
        )
    return tuple(violations)
