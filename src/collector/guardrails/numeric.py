"""Numeric authorization — SPEC §5.2, during-call check 2. The important one.

    The model may never originate a number.

That sentence is a prompt suggestion until something mechanical enforces it.
This module is the mechanism: it extracts *every* figure from a candidate agent
utterance — digits, spelled-out numbers, percentages, durations, dates — and
asserts each one appears in the set the decision engine currently authorizes.
Anything else blocks the turn.

Spelled-out numbers matter as much as digits. A model that has learned to route
figures through a tool still says "eight hundred dollars" when it is improvising,
and a digit-only guard would wave that straight through to TTS.

Two deliberate classification rules, both about false positives and negatives:

* A bare number with no unit is treated as money once it reaches three figures.
  In a call about a $1,000 debt, "I can do two fifty" is an amount, not a count.
* A bare number below three figures is WARN, not BLOCK. "Let me ask you two
  things" is not a policy statement, and blocking it would make the guard
  something operators disable.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from collector.decision_engine import Verdict
from collector.guardrails.prohibited import Severity, Violation
from collector.money import Money
from collector.offers import Offer
from collector.policy import PolicyConfig


class NumericRuleId(StrEnum):
    """Stable identifiers. These appear in the audit log and must not churn."""

    UNAUTHORIZED_AMOUNT = "UNAUTHORIZED_AMOUNT"
    UNAUTHORIZED_PERCENT = "UNAUTHORIZED_PERCENT"
    UNAUTHORIZED_DURATION = "UNAUTHORIZED_DURATION"
    UNAUTHORIZED_PAYMENT_COUNT = "UNAUTHORIZED_PAYMENT_COUNT"
    UNAUTHORIZED_DATE = "UNAUTHORIZED_DATE"
    UNVERIFIED_NUMBER = "UNVERIFIED_NUMBER"


class FigureKind(StrEnum):
    MONEY = "amount"
    PERCENT = "percent"
    DURATION = "duration"
    PAYMENT_COUNT = "payment count"
    DATE = "date"
    BARE = "number"


class DurationUnit(StrEnum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

    @property
    def days(self) -> int:
        return {"day": 1, "week": 7, "month": 30, "year": 365}[self.value]


@dataclass(frozen=True)
class Figure:
    """One numeric claim found in an utterance, with the span that produced it."""

    kind: FigureKind
    text: str
    start: int
    end: int
    value: Decimal | None = None
    unit: DurationUnit | None = None
    token: str | None = None  # normalized form, for dates

    @property
    def value_in_days(self) -> Decimal:
        if self.value is None or self.unit is None:
            raise ValueError(f"{self.kind} figure has no duration")
        return self.value * self.unit.days


_BLOCKING_RULES: dict[FigureKind, NumericRuleId] = {
    FigureKind.MONEY: NumericRuleId.UNAUTHORIZED_AMOUNT,
    FigureKind.PERCENT: NumericRuleId.UNAUTHORIZED_PERCENT,
    FigureKind.DURATION: NumericRuleId.UNAUTHORIZED_DURATION,
    FigureKind.PAYMENT_COUNT: NumericRuleId.UNAUTHORIZED_PAYMENT_COUNT,
    FigureKind.DATE: NumericRuleId.UNAUTHORIZED_DATE,
}

# Temporal deixis, not an account fact: an agent asking "can you pay today?"
# is not originating a figure, and blocking it would break every opening turn.
ALWAYS_ALLOWED_DATES = frozenset({"today", "tonight", "now", "right now"})


# -- authorized set --------------------------------------------------------


@dataclass(frozen=True)
class AuthorizedFigures:
    """Every figure the agent is currently permitted to say out loud.

    Durations are stored as day-equivalents so "three months" and "90 days" are
    the same authorization; the agent phrasing an engine figure in friendlier
    units is not the failure mode this guard exists to catch.
    """

    money: frozenset[Decimal] = frozenset()
    percents: frozenset[Decimal] = frozenset()
    counts: frozenset[Decimal] = frozenset()
    durations: frozenset[Decimal] = frozenset()
    dates: frozenset[str] = frozenset()

    @classmethod
    def empty(cls) -> AuthorizedFigures:
        return cls()

    def merged_with(self, other: AuthorizedFigures) -> AuthorizedFigures:
        return AuthorizedFigures(
            money=self.money | other.money,
            percents=self.percents | other.percents,
            counts=self.counts | other.counts,
            durations=self.durations | other.durations,
            dates=self.dates | other.dates,
        )

    def with_offer(self, offer: Offer) -> AuthorizedFigures:
        return self.merged_with(_from_offer(offer))

    def with_verdict(self, verdict: Verdict) -> AuthorizedFigures:
        return self.merged_with(_from_verdict(verdict))

    def permits(self, figure: Figure) -> bool:
        if figure.kind is FigureKind.DATE:
            token = figure.token or ""
            return token in ALWAYS_ALLOWED_DATES or token in self.dates
        if figure.value is None:
            return True
        if figure.kind is FigureKind.MONEY:
            return figure.value in self.money
        if figure.kind is FigureKind.PERCENT:
            return figure.value in self.percents
        if figure.kind is FigureKind.PAYMENT_COUNT:
            return figure.value in self.counts
        if figure.kind is FigureKind.DURATION:
            return figure.value_in_days in self.durations
        # BARE: unattached, so any authorized reading of the value clears it.
        return (
            figure.value in self.money
            or figure.value in self.counts
            or figure.value in self.durations
            or figure.value in self.percents
        )


def _amounts(values: Iterable[Money]) -> frozenset[Decimal]:
    return frozenset(v.amount for v in values)


def _from_offer(offer: Offer) -> AuthorizedFigures:
    installments = [i.amount for i in offer.installments]
    durations = [Decimal(i.due_day_offset) for i in offer.installments]
    durations.append(Decimal(offer.duration_days))
    durations.append(Decimal(offer.cadence.interval_days))
    return AuthorizedFigures(
        money=_amounts([*installments, offer.total]),
        # 1..n so "the second payment" is sayable once the offer exists.
        counts=frozenset(Decimal(n) for n in range(1, offer.payment_count + 1)),
        durations=frozenset(durations),
    )


def _from_verdict(verdict: Verdict) -> AuthorizedFigures:
    """What the engine *offered*, which is narrower than what it printed.

    A verdict states three kinds of figure and only one of them is a term the
    engine put on the table. SPEC §5.2 authorizes an "offer set", so the test is
    provenance — was this figure ever offered — not whether a condition passed:

    * ``verdict.counter`` is an ``Offer``, built by the engine. This is the only
      thing here that was offered, and it is the whole of what we authorize.
    * ``condition.limit`` is the policy threshold the proposal was *measured
      against*: ">= $800.00", "<= 92 days", "<= 3 for settlement". Harvesting it
      meant a *rejection* unlocked the company's private floors — lowball $400,
      get refused, and the agent could then say "I could settle this at $800"
      with no settlement anywhere in the call. The agent explains a refusal from
      the rationale code and ``you_may_say``, which is precisely why those
      exist; it never needs the threshold to say "that's below what I can take".
    * ``condition.actual`` is the *consumer's* proposal restated back. Harvesting
      it let the consumer choose the agent's numbers: propose $25/week × 40, be
      rejected under the minimum-payment floor, and "$25 a week", "40 payments"
      and "273 days" all became sayable — a tenth of the floor and ten times the
      payment cap, none of it in any offer.

    Neither is redeemed by its condition passing, so per-condition filtering is
    the wrong unit: a $999 proposal *passes* TOTAL_FLOOR while being exactly the
    unauthorized discount. Nor is ``actual`` redeemed by an accepting verdict,
    tempting as that is — an accepted proposal is a real deal, but ``tools.py``
    already turns it into an ``Offer`` (``Offer.from_proposal``) and records it
    in ``offers_made``, which authorizes the whole schedule rather than the one
    smallest instalment ``MIN_PAYMENT.actual`` happens to name. Reading it back
    off the conditions would be both redundant and less complete.
    """
    if verdict.counter is None:
        return AuthorizedFigures.empty()
    return _from_offer(verdict.counter)


def authorized_for(
    policy: PolicyConfig,
    *,
    offers: Iterable[Offer] = (),
    verdicts: Iterable[Verdict] = (),
    extra_money: Iterable[Money] = (),
    extra_percents: Iterable[Decimal | int] = (),
    extra_counts: Iterable[int] = (),
    extra_durations: Iterable[tuple[int, DurationUnit]] = (),
    extra_dates: Iterable[str] = (),
) -> AuthorizedFigures:
    """Build the authorized set for *the current point in the call*.

    SPEC §5.2 says a figure must appear in "the engine's currently-authorized
    offer set", and the emphasis is on *currently*. Only two things go in the
    base set:

    * **The balance.** An account fact the agent was handed with the file, not
      a term it negotiated. It has to be sayable before any offer exists —
      "the balance on the account is $1,000" is the opening of the call.
    * Nothing else.

    The policy *limits* — the $250 minimum payment, the $800 settlement floor,
    the 20% discount ceiling, the 92-day maximum — are deliberately excluded.
    They are the engine's private thresholds, and an agent that says "the most
    I can discount is 20%" before the engine has surfaced a discounted offer is
    negotiating against the company with a number it was never given.

    A limit becomes sayable only by appearing in an *offer* — through ``offers``,
    or through the counter a verdict carries (``_from_verdict``, which reads
    nothing else: a verdict's evaluated conditions restate the policy threshold
    and the consumer's own proposal, and neither was ever offered). So the $800
    settlement floor is unsayable until an $800 settlement is on the table, and
    refusing a proposal surfaces nothing at all — being told a figure is illegal
    is not the same as being offered it.

    Payment counts are on the same footing: ``_from_offer`` authorizes 1..n for
    the offer actually on the table. Before there is an offer there is no
    "second payment" to refer to.

    Account facts beyond the balance — days delinquent, a due date — enter
    through the ``extra_*`` arguments, where the caller states them explicitly
    rather than the guard assuming them.
    """
    base = AuthorizedFigures(money=_amounts([policy.original_balance]))
    extras = AuthorizedFigures(
        money=_amounts(extra_money),
        percents=frozenset(Decimal(p) for p in extra_percents),
        counts=frozenset(Decimal(c) for c in extra_counts),
        durations=frozenset(Decimal(value) * unit.days for value, unit in extra_durations),
        dates=frozenset(d.strip().lower() for d in extra_dates),
    )
    authorized = base.merged_with(extras)
    for offer in offers:
        authorized = authorized.with_offer(offer)
    for verdict in verdicts:
        authorized = authorized.with_verdict(verdict)
    return authorized


# -- extraction ------------------------------------------------------------

_UNITS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES: dict[str, int] = {"hundred": 100, "thousand": 1000}
_NUMBER_WORDS = {**_UNITS, **_TENS, **_SCALES}

_WORD_ALT = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_WORD_NUMBER_RE = re.compile(
    rf"\b(?:an?\s+(?=(?:{_WORD_ALT})\b))?(?:{_WORD_ALT})"
    rf"(?:[\s-]+(?:and[\s-]+)?(?:{_WORD_ALT}))*\b",
    re.IGNORECASE,
)
_DIGIT_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th)?\b")

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec"
)
_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"

_DATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+of\s+(?:{_MONTHS})\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b",
        rf"\b(?:next|this|last)\s+(?:week|month|year|{_WEEKDAYS})\b",
        rf"\b(?:{_WEEKDAYS})\b",
        r"\bthe\s+\d{1,2}(?:st|nd|rd|th)\b",
        r"\b(?:19|20)\d{2}\b",
        r"\b(?:today|tomorrow|tonight)\b",
        r"\bend\s+of\s+(?:the\s+)?(?:month|week)\b",
    )
)

_MONEY_PREFIX_RE = re.compile(r"\$\s*$")
_IDENTIFIER_PREFIX_RE = re.compile(
    r"\b(?:account|reference|confirmation|case|file|ending\s+in|extension|number|#)\s*#?\s*$",
    re.IGNORECASE,
)
_MONEY_SUFFIX_RE = re.compile(r"^[\s-]*(?:dollars?|bucks?|usd)\b", re.IGNORECASE)
_CENTS_SUFFIX_RE = re.compile(r"^[\s-]*cents?\b", re.IGNORECASE)
_PERCENT_SUFFIX_RE = re.compile(r"^\s*(?:%|percent\b)", re.IGNORECASE)
_DURATION_SUFFIX_RE = re.compile(r"^[\s-]*(day|week|month|year)s?\b", re.IGNORECASE)
_PAYMENT_SUFFIX_RE = re.compile(
    r"^[\s-]*(?:payments?|installments?|checks?|cheques?)\b", re.IGNORECASE
)

_ORDINAL_SUFFIX_RE = re.compile(r"(?:st|nd|rd|th)$", re.IGNORECASE)
_CONTEXT_CHARS = 24

# Below this, an unattached number is chatter ("two things"); at or above it, in
# a call about a $1,000 balance, it is an amount.
_MONEY_INFERENCE_FLOOR = Decimal(100)


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < t_end and t_start < end for t_start, t_end in taken)


def _words_to_number(raw: str) -> Decimal | None:
    tokens = [t for t in re.split(r"[\s-]+", raw.lower()) if t and t not in {"and", "a", "an"}]
    if not tokens or any(t not in _NUMBER_WORDS for t in tokens):
        return None

    # Colloquial hundreds: "two fifty" is $250, not 52. Only applies when no
    # scale word is present, which is exactly how people say amounts aloud.
    colloquial = (
        2 <= len(tokens) <= 3
        and tokens[0] in _UNITS
        and 1 <= _UNITS[tokens[0]] <= 9
        and tokens[1] in _TENS
    )
    if colloquial:
        remainder = _TENS[tokens[1]]
        if len(tokens) == 3:
            if tokens[2] not in _UNITS:
                return None
            remainder += _UNITS[tokens[2]]
        return Decimal(_UNITS[tokens[0]] * 100 + remainder)

    total = 0
    current = 0
    for token in tokens:
        if token in _UNITS:
            current += _UNITS[token]
        elif token in _TENS:
            current += _TENS[token]
        elif token == "hundred":
            current = (current or 1) * 100
        else:  # thousand
            total += (current or 1) * 1000
            current = 0
    return Decimal(total + current)


def _digits_to_number(raw: str) -> Decimal | None:
    cleaned = _ORDINAL_SUFFIX_RE.sub("", raw.replace(",", ""))
    try:
        return Decimal(cleaned)
    except ArithmeticError:
        return None


def _classify(
    text: str, start: int, end: int, value: Decimal
) -> tuple[FigureKind, DurationUnit | None, int, int]:
    """Decide what a number *is* from the words around it, and widen its span."""
    before = text[max(0, start - _CONTEXT_CHARS) : start]
    after = text[end : end + _CONTEXT_CHARS]

    if (prefix := _MONEY_PREFIX_RE.search(before)) is not None:
        return FigureKind.MONEY, None, start - len(prefix.group(0)), end
    if (suffix := _MONEY_SUFFIX_RE.match(after)) is not None:
        return FigureKind.MONEY, None, start, end + len(suffix.group(0))
    if (suffix := _CENTS_SUFFIX_RE.match(after)) is not None:
        return FigureKind.MONEY, None, start, end + len(suffix.group(0))
    if (suffix := _PERCENT_SUFFIX_RE.match(after)) is not None:
        return FigureKind.PERCENT, None, start, end + len(suffix.group(0))
    if (suffix := _DURATION_SUFFIX_RE.match(after)) is not None:
        unit = DurationUnit(suffix.group(1).lower())
        return FigureKind.DURATION, unit, start, end + len(suffix.group(0))
    if (suffix := _PAYMENT_SUFFIX_RE.match(after)) is not None:
        return FigureKind.PAYMENT_COUNT, None, start, end + len(suffix.group(0))
    if _IDENTIFIER_PREFIX_RE.search(before) is not None:
        return FigureKind.BARE, None, start, end
    if value >= _MONEY_INFERENCE_FLOOR:
        return FigureKind.MONEY, None, start, end
    return FigureKind.BARE, None, start, end


def extract_figures(text: str) -> tuple[Figure, ...]:
    """Every monetary figure, percentage, count, duration and date in ``text``."""
    figures: list[Figure] = []
    taken: list[tuple[int, int]] = []

    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            if _overlaps(match.span(), taken):
                continue
            taken.append(match.span())
            figures.append(
                Figure(
                    kind=FigureKind.DATE,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    token=" ".join(match.group(0).lower().split()),
                )
            )

    parsers: tuple[tuple[re.Pattern[str], Callable[[str], Decimal | None]], ...] = (
        (_DIGIT_NUMBER_RE, _digits_to_number),
        (_WORD_NUMBER_RE, _words_to_number),
    )
    for regex, parse in parsers:
        for match in regex.finditer(text):
            if _overlaps(match.span(), taken):
                continue
            value = parse(match.group(0))
            if value is None:
                continue
            kind, unit, start, end = _classify(text, match.start(), match.end(), value)
            if _CENTS_SUFFIX_RE.match(text[match.end() : match.end() + _CONTEXT_CHARS]):
                value = value / 100
            taken.append((start, end))
            figures.append(
                Figure(
                    kind=kind,
                    text=text[start:end],
                    start=start,
                    end=end,
                    value=value,
                    unit=unit,
                )
            )

    return tuple(sorted(figures, key=lambda f: f.start))


# -- the check -------------------------------------------------------------


def check_numeric(candidate: str, authorized: AuthorizedFigures) -> tuple[Violation, ...]:
    """Block any figure in a candidate agent utterance the engine did not authorize."""
    violations: list[Violation] = []
    for figure in extract_figures(candidate):
        if authorized.permits(figure):
            continue
        rule = _BLOCKING_RULES.get(figure.kind, NumericRuleId.UNVERIFIED_NUMBER)
        severity = Severity.BLOCK if figure.kind in _BLOCKING_RULES else Severity.WARN
        violations.append(
            Violation(
                rule_id=rule,
                severity=severity,
                span=figure.text,
                start=figure.start,
                end=figure.end,
                detail=(
                    f"{rule}: {figure.text!r} is not an engine-authorized "
                    f"{figure.kind}; the model may never originate a number"
                ),
            )
        )
    return tuple(violations)
