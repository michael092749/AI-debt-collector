"""Repeat-back confirmation — the terms must be said back before they bind.

An agreement is the one thing on this call that outlives it. Everything else
the model says is words; ``confirm_agreement`` writes a payment obligation
against a real person. So the consumer has to hear the whole arrangement, in
one piece, and assent to *that* — not to "so we're all set?" trailing a
negotiation they may have lost the thread of three turns ago.

The same argument the disclosure module makes applies here (``disclosures``
lines 10-12): a repeat-back the prompt merely *asks for* is a habit, and a
habit is not a control. This module makes it one. The engine authors the line,
the detector verifies it was actually spoken, and ``agent.py`` refuses the tool
until it was.

Renderer and detector live together, as the disclosure scripts do, so the two
cannot drift.

Clock-free, like everything downstream of ``offers``: schedules are day offsets
from the call, never absolute dates. The engine cannot say "the 7th of each
month" because the engine does not know what day it is — and a compliance
surface that reads the clock stops being a pure input/output table.
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum

from collector.guardrails.numeric import FigureKind, extract_figures
from collector.offers import Offer


class ConfirmationRuleId(StrEnum):
    """Stable identifiers. These appear in the audit log and must not churn."""

    COMMITMENT_WITHOUT_REPEAT_BACK = "COMMITMENT_WITHOUT_REPEAT_BACK"


# The confirmation idioms. Deliberately a phrase list rather than "ends in a
# question mark": "Can you manage $250 today?" is a *proposal* with a question
# mark on it, and treating that as a confirmation would let the model book an
# agreement off the same sentence it used to ask for one.
_CUE_RE = re.compile(
    r"(?:\b(?:just\s+)?to\s+confirm\b"
    r"|\blet\s+me\s+confirm\b"
    r"|\bconfirming\b"
    r"|\b(?:should|shall|can|do\s+you\s+want\s+me\s+to|want\s+me\s+to)\s+i?\s*"
    r"set\s+(?:that|this|it)\s+up\b"
    r"|\bset\s+(?:that|this|it)\s+up\b"
    r"|\bdo\s+i\s+have\s+(?:that|this)\s+right\b"
    r"|\bhave\s+i\s+got\s+(?:that|this)\s+right\b"
    r"|\bis\s+(?:that|this)\s+(?:right|correct)\b"
    r"|\bdoes\s+(?:that|this)\s+sound\s+right\b"
    r"|\bsounds?\s+right\b"
    r"|\bso\s+(?:that|this)(?:'|’)?s\b"
    r"|\bso\s+we(?:'|’)?re\s+agreed\b"
    r"|\bwe(?:'|’)?re\s+agreed\b)",
    re.IGNORECASE,
)


def _when(due_day_offset: int) -> str:
    """How an offset is spoken. ``today`` is always-allowed temporal deixis
    (``numeric.ALWAYS_ALLOWED_DATES``); every other offset is an authorized
    duration, because ``_from_offer`` authorizes each instalment's own offset."""
    if due_day_offset == 0:
        return "today"
    if due_day_offset == 1:
        return "in 1 day"
    return f"in {due_day_offset} days"


def confirmation_line(offer: Offer) -> str:
    """The canonical repeat-back for an engine-authored offer.

    Every figure in it is one ``numeric._from_offer`` authorizes for this same
    offer, so the line always clears the pre-TTS guard. That is the property
    that makes it safe to tell the model to say it exactly: a canonical string
    the guard would block is worse than no canonical string, because it trains
    the model to paraphrase its way around a control.
    """
    parts = [f"{i.amount} {_when(i.due_day_offset)}" for i in offer.installments]
    if len(parts) == 1:
        schedule = parts[0]
    else:
        schedule = f"{parts[0]}, then " + ", then ".join(parts[1:])
        schedule += f" — {offer.total} in total"
    return f"Just to confirm: {schedule}. Should I set that up?"


def repeats_back(text: str, offer: Offer) -> bool:
    """Did this utterance read *these* terms back and ask the consumer to agree?

    Lenient on prose, exact on numbers. The model will not reproduce
    ``confirmation_line`` verbatim and should not have to — but it does not get
    to drop an instalment, and it does not get to slip in a figure that is not
    part of the deal. The amounts come from the shared ``extract_figures``, so
    "$250.00", "250" and "two fifty" all read as the same commitment.

    The utterance must state no money figure beyond the offer's own. A stricter
    demand than the numeric guard makes of an ordinary turn, and deliberately
    so: this is the one sentence whose numbers become an obligation, so an
    unrelated amount sitting inside it is ambiguity in the record of what was
    agreed to.
    """
    if _CUE_RE.search(text) is None:
        return False

    spoken = {f.value for f in extract_figures(text) if f.kind is FigureKind.MONEY}
    required = {i.amount.amount for i in offer.installments}
    permitted: set[Decimal] = required | {offer.total.amount}
    return required <= spoken and spoken <= permitted
