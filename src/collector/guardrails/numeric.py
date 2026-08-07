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
import unicodedata
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
    INVISIBLE_CHARACTER = "INVISIBLE_CHARACTER"
    SUSPICIOUS_DIGIT_BOUNDARY = "SUSPICIOUS_DIGIT_BOUNDARY"


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
    # A digit run touching a letter or underscore ("950k", "9_5_0") with no
    # recognized suffix is exactly the shape a boundary-evasion attempt takes
    # — flagged rather than silently dropped (ADVERSARIAL_TESTING.md H2).
    suspicious: bool = False

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
    """Whatever the engine itself stated is by definition engine-authored.

    Conditions carry their actual and limit as display strings ("$250.00",
    "<= 92 days"), so the same extractor that polices the agent is reused to
    harvest them. That keeps the authorized set derived from the decision
    record rather than restated by hand.
    """
    authorized = AuthorizedFigures.empty()
    if verdict.counter is not None:
        authorized = authorized.with_offer(verdict.counter)
    for condition in verdict.conditions:
        authorized = authorized.merged_with(_harvest(f"{condition.actual} {condition.limit}"))
    return authorized


def _harvest(text: str) -> AuthorizedFigures:
    money: set[Decimal] = set()
    percents: set[Decimal] = set()
    counts: set[Decimal] = set()
    durations: set[Decimal] = set()
    for figure in extract_figures(text):
        if figure.value is None:
            continue
        if figure.kind is FigureKind.MONEY:
            money.add(figure.value)
        elif figure.kind is FigureKind.PERCENT:
            percents.add(figure.value)
        elif figure.kind is FigureKind.DURATION:
            durations.add(figure.value_in_days)
        else:
            counts.add(figure.value)
    return AuthorizedFigures(
        money=frozenset(money),
        percents=frozenset(percents),
        counts=frozenset(counts),
        durations=frozenset(durations),
    )


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
    negotiating against the company with a number it was never given. Once the
    engine does surface one, it arrives through ``offers`` (the schedule) or
    ``verdicts`` (the evaluated-condition trail, whose ``actual`` and ``limit``
    strings are harvested), and it is authorized from that moment on.

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


# -- invisible-character defense --------------------------------------------
#
# Unicode format characters (category ``Cf``: zero-width space, zero-width
# joiner, soft hyphen, byte-order mark, ...) have no legitimate reason to
# appear in text bound for TTS. Left alone, one placed inside a figure
# fragments it into two separately-authorized numbers that read as one
# unauthorized amount when spoken — "4[ZWSP]800" extracts as "4" and "800"
# individually, never as "$4,800" (ADVERSARIAL_TESTING.md C6). NFKC
# normalization does not help here: it does not touch ``Cf`` characters.
#
# The same fragmentation reproduces with non-breaking space (category ``Zs``,
# not ``Cf``) and a bare tab (category ``Cc``) standing in for the ZWSP, so
# the net is cast over whitespace/format/control characters generally. Plain
# ASCII space is always exempt; newline and carriage return are too — a real
# model turn can legitimately be multi-line ("Here's what I can do:\n$250
# today..."), and a blanket block there would burn regeneration strikes on
# ordinary formatting rather than an evasion attempt.
_SUSPICIOUS_CATEGORIES = frozenset({"Cf", "Cc", "Zs", "Zl", "Zp"})
_ALWAYS_ALLOWED_WHITESPACE = frozenset({" ", "\n", "\r"})


def _is_suspicious_char(ch: str) -> bool:
    if ch in _ALWAYS_ALLOWED_WHITESPACE:
        return False
    return unicodedata.category(ch) in _SUSPICIOUS_CATEGORIES


def _has_format_chars(text: str) -> bool:
    return any(_is_suspicious_char(ch) for ch in text)


def _strip_format_chars(text: str) -> str:
    return "".join(ch for ch in text if not _is_suspicious_char(ch))


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
# "grand"/"stack"/"dozen" are scale words exactly like "thousand" is: "two
# grand" and "two thousand" compose the same way. A model that has learned to
# route figures through a tool still says these when improvising, same as
# any other spelled-out number (ADVERSARIAL_TESTING.md H1).
_SCALES: dict[str, int] = {
    "hundred": 100,
    "thousand": 1000,
    "grand": 1000,
    "stack": 1000,
    "dozen": 12,
}
_NUMBER_WORDS = {**_UNITS, **_TENS, **_SCALES}

_WORD_ALT = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_WORD_NUMBER_RE = re.compile(
    rf"\b(?:an?\s+(?=(?:{_WORD_ALT})\b))?(?:{_WORD_ALT})"
    rf"(?:[\s-]+(?:and[\s-]+)?(?:{_WORD_ALT}))*\b",
    re.IGNORECASE,
)
# Boundaries are digit-adjacency, not \b: a plain word boundary treats a
# letter or underscore as the same word-class as a digit, so a run touching
# either ("950k", "settle950today", "9_5_0") previously failed to match at
# all rather than being flagged (ADVERSARIAL_TESTING.md H2).
_DIGIT_NUMBER_RE = re.compile(r"(?<!\d)\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th)?(?!\d)")

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


# Idioms with no digit or cardinal-number word in them at all: a fraction
# framed as a discount ("half of that", "a third off"), a duration noun with
# no numeral ("a fortnight"), and a bare single-letter money slang ("a G").
# These require no adversarial intent at all — exactly the failure mode this
# guard exists to catch, per its own docstring (ADVERSARIAL_TESTING.md H1).
@dataclass(frozen=True)
class _Idiom:
    pattern: re.Pattern[str]
    kind: FigureKind
    value: Decimal
    unit: DurationUnit | None = None


_FRACTION_TAIL = r"(?:of\s+(?:that|it|this|the\s+balance)|off)\b"
_IDIOMS: tuple[_Idiom, ...] = (
    _Idiom(
        re.compile(rf"\bhalf\s+{_FRACTION_TAIL}", re.IGNORECASE),
        FigureKind.PERCENT,
        Decimal(50),
    ),
    _Idiom(
        re.compile(rf"\ba\s+third\s+{_FRACTION_TAIL}", re.IGNORECASE),
        FigureKind.PERCENT,
        Decimal(100) / Decimal(3),
    ),
    _Idiom(
        re.compile(rf"\ba\s+quarter\s+{_FRACTION_TAIL}", re.IGNORECASE),
        FigureKind.PERCENT,
        Decimal(25),
    ),
    _Idiom(
        re.compile(rf"\ba\s+fifth\s+{_FRACTION_TAIL}", re.IGNORECASE),
        FigureKind.PERCENT,
        Decimal(20),
    ),
    _Idiom(
        re.compile(r"\ba\s+fortnight\b", re.IGNORECASE),
        FigureKind.DURATION,
        Decimal(14),
        DurationUnit.DAY,
    ),
    _Idiom(re.compile(r"\ba\s+[gG]\b"), FigureKind.MONEY, Decimal(1000)),
)

# Bare Roman numerals ("CD dollars", "IX dollars") only when at least two
# characters and immediately followed by a recognized money/percent suffix —
# a single-letter numeral ("I", "V", "X") collides with far too much ordinary
# English ("I can pay...") to ever treat on its own (ADVERSARIAL_TESTING.md
# H5).
_ROMAN_NUMERAL_RE = re.compile(r"\bM{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b")
_ROMAN_VALUES: dict[str, int] = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(token: str) -> int:
    total = 0
    prev = 0
    for ch in reversed(token.upper()):
        value = _ROMAN_VALUES[ch]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total


_MONEY_PREFIX_RE = re.compile(r"\$\s*$")
_IDENTIFIER_PREFIX_RE = re.compile(
    r"\b(?:account|reference|confirmation|case|file|ending\s+in|extension|number|#)\s*#?\s*$",
    re.IGNORECASE,
)
_MONEY_SUFFIX_RE = re.compile(r"^[\s-]*(?:dollars?|bucks?|usd)\b", re.IGNORECASE)
_CENTS_SUFFIX_RE = re.compile(r"^[\s-]*cents?\b", re.IGNORECASE)
_PERCENT_SUFFIX_RE = re.compile(r"^\s*(?:%|per\s*cent\b)", re.IGNORECASE)
_DURATION_SUFFIX_RE = re.compile(r"^[\s-]*(day|week|month|year)s?\b", re.IGNORECASE)
_THOUSAND_SUFFIX_RE = re.compile(r"^k\b", re.IGNORECASE)
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
        else:  # thousand-like scale word: thousand, grand, stack, dozen
            total += (current or 1) * _SCALES[token]
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


def _touches_letter_or_underscore(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return any(ch.isalpha() or ch == "_" for ch in (before, after) if ch)


def extract_figures(text: str) -> tuple[Figure, ...]:
    """Every monetary figure, percentage, count, duration and date in ``text``."""
    text = _strip_format_chars(text)
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

    for idiom in _IDIOMS:
        for match in idiom.pattern.finditer(text):
            if _overlaps(match.span(), taken):
                continue
            taken.append(match.span())
            figures.append(
                Figure(
                    kind=idiom.kind,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    value=idiom.value,
                    unit=idiom.unit,
                )
            )

    for match in _ROMAN_NUMERAL_RE.finditer(text):
        token = match.group(0)
        if len(token) < 2 or _overlaps(match.span(), taken):
            continue
        after = text[match.end() : match.end() + _CONTEXT_CHARS]
        roman_kind: FigureKind | None = None
        suffix_match: re.Match[str] | None = None
        if (money_suffix := _MONEY_SUFFIX_RE.match(after)) is not None:
            roman_kind, suffix_match = FigureKind.MONEY, money_suffix
        elif (percent_suffix := _PERCENT_SUFFIX_RE.match(after)) is not None:
            roman_kind, suffix_match = FigureKind.PERCENT, percent_suffix
        if roman_kind is None or suffix_match is None:
            continue
        end = match.end() + len(suffix_match.group(0))
        taken.append((match.start(), end))
        figures.append(
            Figure(
                kind=roman_kind,
                text=text[match.start() : end],
                start=match.start(),
                end=end,
                value=Decimal(_roman_to_int(token)),
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
            after = text[match.end() : match.end() + _CONTEXT_CHARS]
            if _CENTS_SUFFIX_RE.match(after):
                value = value / 100
            elif regex is _DIGIT_NUMBER_RE and (thousand := _THOUSAND_SUFFIX_RE.match(after)):
                # "950k" — a digit run with a bare 'k' suffix, no space. Not
                # handled by _classify's suffix table since it isn't a unit
                # word; folded in here the same way the cents suffix is.
                value = value * 1000
                kind = FigureKind.MONEY
                end = match.end() + len(thousand.group(0))
            taken.append((start, end))
            figures.append(
                Figure(
                    kind=kind,
                    text=text[start:end],
                    start=start,
                    end=end,
                    value=value,
                    unit=unit,
                    suspicious=(
                        regex is _DIGIT_NUMBER_RE
                        and _touches_letter_or_underscore(text, start, end)
                    ),
                )
            )

    return tuple(sorted(figures, key=lambda f: f.start))


# -- the check -------------------------------------------------------------


def check_numeric(candidate: str, authorized: AuthorizedFigures) -> tuple[Violation, ...]:
    """Block any figure in a candidate agent utterance the engine did not authorize."""
    violations: list[Violation] = []
    if _has_format_chars(candidate):
        violations.append(
            Violation(
                rule_id=NumericRuleId.INVISIBLE_CHARACTER,
                severity=Severity.BLOCK,
                span=candidate,
                start=0,
                end=len(candidate),
                detail=(
                    f"{NumericRuleId.INVISIBLE_CHARACTER}: candidate text contains a "
                    "Unicode format character (zero-width space/joiner, soft hyphen, "
                    "BOM, ...); there is no legitimate reason for one in spoken text"
                ),
            )
        )
    for figure in extract_figures(candidate):
        if figure.suspicious:
            # A digit run sitting directly against a letter or underscore
            # with no recognized unit ("950k", "9_5_0") is exactly the shape
            # a boundary-evasion attempt takes — blocked outright rather than
            # judged by whether its raw value happens to be authorized
            # (ADVERSARIAL_TESTING.md H2).
            violations.append(
                Violation(
                    rule_id=NumericRuleId.SUSPICIOUS_DIGIT_BOUNDARY,
                    severity=Severity.BLOCK,
                    span=figure.text,
                    start=figure.start,
                    end=figure.end,
                    detail=(
                        f"{NumericRuleId.SUSPICIOUS_DIGIT_BOUNDARY}: {figure.text!r} sits "
                        "directly against a letter or underscore with no recognized unit"
                    ),
                )
            )
            continue
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
