"""Required disclosures — SPEC §5.2, during-call check 3.

Two disclosures gate the call:

* **Mini-Miranda** — "this is an attempt to collect a debt, any information
  obtained will be used for that purpose" — before any substantive collection
  discussion.
* **AI disclosure** — at the open, and again whenever the consumer asks.

The state is exposed rather than hidden so the negotiation state machine can
refuse to advance until they have fired: a disclosure that is merely *checked*
after the fact is a log entry, while one that gates the next turn is a control.

Ordering is checked inside a single utterance too. An agent that opens with
"you owe eleven hundred dollars — and this is an attempt to collect a debt" has
technically said the words and still disclosed too late.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum

from collector.guardrails.prohibited import Severity, Violation


class DisclosureId(StrEnum):
    MINI_MIRANDA = "MINI_MIRANDA"
    AI_DISCLOSURE = "AI_DISCLOSURE"


class DisclosureRuleId(StrEnum):
    """Stable identifiers. These appear in the audit log and must not churn."""

    MINI_MIRANDA_NOT_FIRED = "MINI_MIRANDA_NOT_FIRED"
    MINI_MIRANDA_OUT_OF_ORDER = "MINI_MIRANDA_OUT_OF_ORDER"
    AI_DISCLOSURE_MISSING_AT_OPEN = "AI_DISCLOSURE_MISSING_AT_OPEN"
    AI_DISCLOSURE_REQUEST_IGNORED = "AI_DISCLOSURE_REQUEST_IGNORED"


# The scripts. Kept here, next to their detectors, so the two cannot drift.
MINI_MIRANDA_TEXT = (
    "This is an attempt to collect a debt, and any information obtained "
    "will be used for that purpose."
)
AI_DISCLOSURE_TEXT = (
    "Before we go further: I'm an AI assistant calling on behalf of the creditor, "
    "and you can ask for a human at any time."
)

_MINI_MIRANDA_ATTEMPT_RE = re.compile(r"attempt\s+to\s+collect\s+a\s+debt", re.IGNORECASE)
_MINI_MIRANDA_PURPOSE_RE = re.compile(
    r"(?:information|info)[^.!?]{0,80}?(?:will\s+be\s+)?used\s+for\s+(?:that|this)\s+purpose",
    re.IGNORECASE,
)

_AI_DISCLOSURE_RE = re.compile(
    r"\b(?:i(?:'m|’m|\s+am)\s+(?:an?\s+)?(?:a\.i\.|ai|artificial\s+intelligence|automated|"
    r"virtual|digital|synthetic)\b"
    # Third-person "this is ..." disclosures (live-testing found this is the
    # model's most common opening phrasing). "ai"/"a.i."/"artificial
    # intelligence" are specific enough on their own to need no trailing noun;
    # the generic adjectives below do, or "this is a virtual card number" and
    # its like would false-fire and permanently satisfy the disclosure.
    r"|this\s+is\s+(?:an?\s+)?(?:a\.i\.|ai|artificial\s+intelligence)\b"
    r"|this\s+is\s+(?:an?\s+)?(?:automated|virtual|digital|synthetic)\s+"
    r"(?:assistant|agent|system|voice|call(?:er)?)\b"
    r"|\ba\.?i\.?\s+(?:assistant|agent|voice|system)\b"
    r"|automated\s+(?:assistant|agent|system|voice)\b"
    r"|virtual\s+(?:assistant|agent)\b"
    r"|i(?:'m|’m|\s+am)\s+(?:a\s+)?(?:bot|not\s+a\s+human|not\s+a\s+person)\b)",
    re.IGNORECASE,
)

_AI_ADJ = r"(?:an?\s+)?(?:(?:real|actual|live)\s+)?"
_AI_REQUEST_RE = re.compile(
    rf"\b(?:are\s+you\s+{_AI_ADJ}(?:real|human|person|robot|bot|ai|recording|machine|"
    r"computer)"
    rf"|am\s+i\s+(?:talking|speaking)\s+(?:to|with)\s+{_AI_ADJ}(?:real|human|person|machine|bot|"
    r"robot|computer)"
    rf"|is\s+this\s+being\s+recorded"
    rf"|is\s+this\s+{_AI_ADJ}(?:recording|robot|bot|ai|real\s+person|human|machine|computer|"
    r"automated)"
    r"|you(?:'re|’re|\s+are)\s+(?:a\s+)?(?:robot|bot|ai|machine|recording)"
    r"|(?:get|put)\s+me\s+(?:a|to\s+a)\s+(?:real\s+person|human)"
    r"|(?:human|person)\s+or\s+(?:bot|ai|robot|machine))\b",
    re.IGNORECASE,
)

# What counts as "substantive collection discussion". Deliberately narrow:
# greeting and identity verification must be able to happen first. A dollar
# figure counts on its own — quoting the balance *is* the discussion.
_SUBSTANTIVE_RE = re.compile(
    r"(?:\$\s?\d"
    r"|\b(?:balance|owes?|owed?|owing|amount\s+due|past\s+due|delinquent|dollars?|payments?|"
    r"paying|pay|paid|settle|settlement|installments?|debt|collect|arrangement|offer|offering|"
    r"afford|charge|charges?)\b)",
    re.IGNORECASE,
)


def fires_mini_miranda(text: str) -> int | None:
    """Start offset of a complete Mini-Miranda, or ``None``.

    Both halves are required. Half of the disclosure is not the disclosure.
    """
    attempt = _MINI_MIRANDA_ATTEMPT_RE.search(text)
    if attempt is None or _MINI_MIRANDA_PURPOSE_RE.search(text) is None:
        return None
    return attempt.start()


def fires_ai_disclosure(text: str) -> bool:
    return _AI_DISCLOSURE_RE.search(text) is not None


def requests_ai_disclosure(text: str) -> bool:
    return _AI_REQUEST_RE.search(text) is not None


# Deliberately narrow. "Who's asking?" and "What is this about?" are not
# confirmations, and neither is silence; being conservative here costs one
# extra question, while being generous discloses a debt to whoever picked up.
#
# "go ahead" and "you've got (him|her|them|me)" were dropped: both false-
# positive on stray phrases that were never meant as an identity confirmation
# at all — most severely, an answering-machine greeting ("...go ahead when
# you're ready") and a third party about to hand the phone off
# (ADVERSARIAL_TESTING.md C2).
_IDENTITY_CONFIRMED_RE = re.compile(
    r"(?:\byes\b|\byeah\b|\byep\b|\byup\b|\bspeaking\b|\bthat'?s me\b|\bthis is (?:he|she|they|"
    r"them|me)\b)",
    re.IGNORECASE,
)
# A third party identifying themselves by relation ("this is his brother") is
# a denial in substance even without the word "no": the speaker has just said,
# in plain words, they are someone other than the account holder.
#
# Deliberately includes bare "no": within confirms_identity this only clears
# an affirmation elsewhere in the *same* utterance ("yes, but no, that's not
# me") — a bounded, low-risk use. It must not also drive denies_identity
# (below), which re-checks on every later turn once identity is confirmed:
# "no" is the single most common word in an ordinary negotiation refusal
# ("No, too much."), and revoking identity on it would end the call's ability
# to discuss terms at all (ADVERSARIAL_TESTING.md C1 follow-up).
_IDENTITY_DENIED_RE = re.compile(
    r"(?:\bno\b|\bwrong number\b|\bnot (?:me|him|her|them)\b|\bnever heard of\b|"
    r"\bdoesn'?t live here\b|\bthey'?re not here\b|"
    r"\bthis is (?:his|her|their|my)\s+(?:brother|sister|son|daughter|wife|husband|"
    r"roommate|mother|father|mom|dad|assistant|secretary)\b)",
    re.IGNORECASE,
)
# The subset above that is an unambiguous, explicit denial regardless of
# where in the call it appears — no bare "no", which is ordinary negotiation
# vocabulary, not an identity claim.
_IDENTITY_EXPLICIT_DENIAL_RE = re.compile(
    r"(?:\bwrong number\b|\bnot (?:me|him|her|them)\b|\bnever heard of\b|"
    r"\bdoesn'?t live here\b|\bthey'?re not here\b|"
    r"\bthis is (?:his|her|their|my)\s+(?:brother|sister|son|daughter|wife|husband|"
    r"roommate|mother|father|mom|dad|assistant|secretary)\b)",
    re.IGNORECASE,
)


def confirms_identity(text: str) -> bool:
    """Did the consumer just confirm they are the account holder? (SPEC §5.1)

    Code decides this, not the model: it gates every substantive word that
    follows, so it cannot be a judgement call made inside a generated turn. A
    denial anywhere in the utterance wins over an affirmation — "yes, but no,
    that's not me" is not a confirmation.
    """
    if _IDENTITY_DENIED_RE.search(text):
        return False
    return _IDENTITY_CONFIRMED_RE.search(text) is not None


def denies_identity(text: str) -> bool:
    """Did the consumer just say, explicitly, that they are not the account
    holder? Checked on every turn, not just before identity is confirmed: a
    confirmation is not a one-way latch — "hold on, someone's at the door"
    followed later by "sorry, that's not me" must revoke it, not leave the
    balance sayable to whoever is now on the line (ADVERSARIAL_TESTING.md C1).

    Uses a narrower pattern than the negation cue inside ``confirms_identity``
    on purpose: this is checked on *every* turn once identity is confirmed,
    including plain negotiation refusals ("No, too much.") — a bare "no"
    revoking identity there would end the call's ability to discuss terms.
    """
    return _IDENTITY_EXPLICIT_DENIAL_RE.search(text) is not None


def substantive_span(text: str) -> tuple[int, int] | None:
    match = _SUBSTANTIVE_RE.search(text)
    return None if match is None else match.span()


def is_substantive(text: str) -> bool:
    return substantive_span(text) is not None


@dataclass(frozen=True)
class DisclosureState:
    """Which disclosures have fired, and what is owed right now."""

    fired: frozenset[DisclosureId] = frozenset()
    ai_disclosure_requested: bool = False
    agent_turns: int = 0

    @classmethod
    def opening(cls) -> DisclosureState:
        return cls()

    # -- what the negotiation state machine reads --------------------------

    @property
    def may_discuss_debt(self) -> bool:
        return DisclosureId.MINI_MIRANDA in self.fired

    @property
    def pending(self) -> tuple[DisclosureId, ...]:
        """Disclosures the very next agent turn must carry."""
        owed: list[DisclosureId] = []
        at_open = self.agent_turns == 0 and DisclosureId.AI_DISCLOSURE not in self.fired
        if at_open or self.ai_disclosure_requested:
            owed.append(DisclosureId.AI_DISCLOSURE)
        if not self.may_discuss_debt:
            owed.append(DisclosureId.MINI_MIRANDA)
        return tuple(owed)

    @property
    def missing(self) -> tuple[DisclosureId, ...]:
        return tuple(d for d in DisclosureId if d not in self.fired)

    # -- transitions -------------------------------------------------------

    def observe_consumer(self, utterance: str) -> DisclosureState:
        if requests_ai_disclosure(utterance):
            return replace(self, ai_disclosure_requested=True)
        return self

    def observe_agent(self, utterance: str) -> DisclosureState:
        fired = set(self.fired)
        answered = fires_ai_disclosure(utterance)
        if answered:
            fired.add(DisclosureId.AI_DISCLOSURE)
        if fires_mini_miranda(utterance) is not None:
            fired.add(DisclosureId.MINI_MIRANDA)
        return DisclosureState(
            fired=frozenset(fired),
            ai_disclosure_requested=self.ai_disclosure_requested and not answered,
            agent_turns=self.agent_turns + 1,
        )

    # -- the check ---------------------------------------------------------

    def check_agent_turn(self, candidate: str) -> tuple[Violation, ...]:
        violations: list[Violation] = []
        whole = (0, len(candidate))

        if (
            self.agent_turns == 0
            and DisclosureId.AI_DISCLOSURE not in self.fired
            and not fires_ai_disclosure(candidate)
        ):
            violations.append(
                _violation(
                    DisclosureRuleId.AI_DISCLOSURE_MISSING_AT_OPEN,
                    candidate,
                    whole,
                    "the first agent turn must disclose that the caller is an AI",
                )
            )
        elif self.ai_disclosure_requested and not fires_ai_disclosure(candidate):
            violations.append(
                _violation(
                    DisclosureRuleId.AI_DISCLOSURE_REQUEST_IGNORED,
                    candidate,
                    whole,
                    "the consumer asked whether they are speaking to a human; answer it",
                )
            )

        span = substantive_span(candidate)
        if span is not None and not self.may_discuss_debt:
            fired_at = fires_mini_miranda(candidate)
            if fired_at is None:
                violations.append(
                    _violation(
                        DisclosureRuleId.MINI_MIRANDA_NOT_FIRED,
                        candidate,
                        span,
                        "substantive collection discussion before the Mini-Miranda",
                    )
                )
            elif fired_at > span[0]:
                violations.append(
                    _violation(
                        DisclosureRuleId.MINI_MIRANDA_OUT_OF_ORDER,
                        candidate,
                        span,
                        "the Mini-Miranda must precede the collection discussion, not trail it",
                    )
                )

        return tuple(violations)


def _violation(
    rule_id: DisclosureRuleId, candidate: str, span: tuple[int, int], detail: str
) -> Violation:
    start, end = span
    return Violation(
        rule_id=rule_id,
        severity=Severity.BLOCK,
        span=candidate[start:end],
        start=start,
        end=end,
        detail=f"{rule_id}: {detail}",
    )
