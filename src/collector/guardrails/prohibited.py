"""Prohibited persuasion — SPEC §5.2, during-call check 1.

The brief is explicit: persuasion that relies on threats, false urgency, or
invented consequences fails *regardless of how well it converts*. So this runs
on every candidate agent utterance before TTS and a trip blocks the turn.

Detection is deliberately conservative about negated phrasing. An agent
answering "no, nobody is going to sue you" is doing exactly the right thing, and
flagging it would train the model out of the reassurance consumers most need to
hear. A matched phrase is therefore cleared when a negation cue appears earlier
in the same clause.

This module also carries the shared violation vocabulary (``Severity``,
``Violation``) used by the other three checks: SPEC §3 fixes this package at
four modules, so the common types live with the first check rather than in a
fifth file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# -- shared violation vocabulary -------------------------------------------


class Severity(StrEnum):
    """BLOCK stops the turn. WARN is recorded for the audit log and nothing more."""

    BLOCK = "block"
    WARN = "warn"


@dataclass(frozen=True)
class Violation:
    """One guardrail trip, self-describing so the audit trail needs no decoder.

    ``rule_id`` is a plain ``str`` because every check module contributes its
    own ``StrEnum``; the ids are stable and appear verbatim in the log.
    """

    rule_id: str
    severity: Severity
    span: str
    start: int
    end: int
    detail: str

    @property
    def blocking(self) -> bool:
        return self.severity is Severity.BLOCK


# -- rule identifiers ------------------------------------------------------


class ProhibitedRuleId(StrEnum):
    """Stable identifiers. These appear in the audit log and must not churn."""

    THREAT_LEGAL_ACTION = "THREAT_LEGAL_ACTION"
    THREAT_ARREST = "THREAT_ARREST"
    THREAT_GARNISHMENT = "THREAT_GARNISHMENT"
    THREAT_CREDIT_REPORT = "THREAT_CREDIT_REPORT"
    FALSE_URGENCY = "FALSE_URGENCY"
    INVENTED_CONSEQUENCE = "INVENTED_CONSEQUENCE"
    ABUSIVE_LANGUAGE = "ABUSIVE_LANGUAGE"
    UNAUTHORIZED_ADVICE = "UNAUTHORIZED_ADVICE"


@dataclass(frozen=True)
class _Rule:
    rule_id: ProhibitedRuleId
    pattern: re.Pattern[str]
    detail: str
    negatable: bool = True


def _rule(
    rule_id: ProhibitedRuleId, detail: str, *alternatives: str, negatable: bool = True
) -> _Rule:
    return _Rule(rule_id, re.compile("|".join(alternatives), re.IGNORECASE), detail, negatable)


_RULES: tuple[_Rule, ...] = (
    _rule(
        ProhibitedRuleId.THREAT_LEGAL_ACTION,
        "asserts or implies litigation the collector has not decided on",
        r"\bsue\s+you\b",
        r"\bsuing\s+you\b",
        r"\btak(?:e|ing)\s+you\s+to\s+court\b",
        r"\blaw\s?suits?\b",
        r"\blegal\s+action\b",
        r"\blegal\s+(?:proceedings|remedies|consequences)\b",
        r"\bfil(?:e|ing)\s+(?:a\s+)?(?:suit|claim|case)\s+against\s+you\b",
        r"\bserv(?:e|ing)\s+you\s+(?:with\s+)?papers\b",
        r"\bjudgment\s+against\s+you\b",
        r"\btak(?:e|ing)\s+legal\b",
        r"\bturn(?:ing)?\s+(?:this|it|your\s+account)\s+over\s+to\s+(?:our\s+)?"
        r"(?:legal|attorneys?|lawyers?)\b",
        r"\bour\s+legal\s+(?:department|team)\b",
    ),
    _rule(
        ProhibitedRuleId.THREAT_ARREST,
        "asserts a criminal consequence; non-payment of a consumer debt is not a crime",
        r"\barrest(?:ed|ing)?\b",
        r"\bjail\b",
        r"\bprison\b",
        r"\bwarrant\b",
        r"\bcriminal\s+(?:charges?|record|case)\b",
        r"\bfraud\s+charges?\b",
        r"\bpress\s+charges\b",
        r"\bcall\s+the\s+police\b",
    ),
    _rule(
        ProhibitedRuleId.THREAT_GARNISHMENT,
        "asserts an involuntary collection remedy that requires a court order",
        r"\bgarnish(?:ment|ing|ed|ee)?\b",
        r"\blev(?:y|ying|ied)\s+(?:your\s+)?(?:bank|account|wages)\b",
        r"\bseiz(?:e|ing|ure)\s+(?:your\s+)?(?:assets|property|bank|accounts?|wages)\b",
        r"\bfreeze\s+(?:your\s+)?(?:bank\s+)?accounts?\b",
        r"\bput\s+a\s+lien\s+on\b",
        r"\brepossess(?:ion|ing|ed)?\b",
        r"\battach\s+your\s+(?:wages|assets)\b",
    ),
    _rule(
        ProhibitedRuleId.THREAT_CREDIT_REPORT,
        "asserts a credit-reporting outcome the agent cannot know or promise",
        r"\bcredit\s+(?:report(?:ing|ed|s)?|bureaus?|score|file|rating)\b",
        r"\breport(?:ing|ed)?\s+(?:this|it|you|the\s+account)\s+to\s+(?:the\s+)?credit\b",
        r"\bruin\s+your\s+credit\b",
        r"\bhurt\s+your\s+credit\b",
        r"\btank\s+your\s+credit\b",
        r"\bequifax|experian|transunion\b",
    ),
    _rule(
        ProhibitedRuleId.FALSE_URGENCY,
        "manufactures a deadline the policy does not actually impose",
        r"\btoday\s+only\b",
        r"\bonly\s+(?:good\s+|available\s+)?(?:for\s+)?today\b",
        r"\bfinal\s+(?:notice|warning|chance|opportunity|demand)\b",
        r"\blast\s+(?:chance|opportunity|call)\b",
        r"\bnow\s+or\s+never\b",
        r"\bthis\s+offer\s+(?:expires|ends|disappears|goes\s+away)\b",
        r"\bexpires?\s+(?:today|tonight|at\s+midnight|in\s+\w+\s+minutes)\b",
        r"\bbefore\s+it'?s\s+too\s+late\b",
        r"\bact\s+now\b",
        r"\blimited[- ]time\b",
        r"\bone[- ]time\s+(?:only|offer|deal)\b",
        r"\boff\s+the\s+table\b",
        r"\bwon'?t\s+be\s+(?:available|here|around)\s+(?:again|tomorrow|later)\b",
        r"\bif\s+you\s+hang\s+up\b",
    ),
    _rule(
        ProhibitedRuleId.INVENTED_CONSEQUENCE,
        "invents a consequence that is not in the policy and may not exist at all",
        r"\bpermanent\s+record\b",
        r"\b(?:contact|call|notify)(?:ing)?\s+your\s+"
        r"(?:employer|boss|work|family|neighbors?|references)\b",
        r"\b(?:show\s+up|come)\s+(?:at|to)\s+your\s+(?:house|home|door|work|job)\b",
        r"\bsend\s+someone\s+(?:to|out)\b",
        r"\byou'?(?:ll|re\s+going\s+to)\s+lose\s+your\s+"
        r"(?:house|home|car|job|license|benefits)\b",
        r"\bsuspend\s+your\s+(?:driver'?s\s+)?license\b",
        r"\bblack\s?list(?:ed)?\b",
        r"\bfollow\s+you\s+(?:around\s+)?for(?:ever|\s+the\s+rest\s+of\s+your\s+life)\b",
        r"\bnever\s+be\s+able\s+to\s+(?:get|borrow|rent|buy|finance)\b",
        r"\bnotify\s+(?:the\s+)?(?:irs|police|authorities|immigration)\b",
        r"\bdeport(?:ed|ation)?\b",
    ),
    _rule(
        ProhibitedRuleId.ABUSIVE_LANGUAGE,
        "demeans the consumer; harassment is a compliance failure on its own",
        r"\bdeadbeat\b",
        r"\byou\s+people\b",
        r"\bshould\s+be\s+ashamed\b",
        r"\bstop\s+lying\b",
        r"\byou'?re\s+(?:a\s+)?(?:liar|stupid|pathetic|worthless)\b",
        r"\bwast(?:e|ing)\s+my\s+time\b",
        # "don't be a deadbeat" is still an insult. Unlike a threat, an
        # abusive word is not made acceptable by the grammar around it.
        negatable=False,
    ),
    _rule(
        ProhibitedRuleId.UNAUTHORIZED_ADVICE,
        "gives legal or financial advice the agent is not licensed to give (SPEC §10)",
        r"\byou\s+should\s+(?:file\s+for\s+|declare\s+)?bankrupt(?:cy)?\b",
        r"\byou\s+should\s+(?:take\s+out|get)\s+a\s+loan\b",
        r"\byou\s+should\s+borrow\b",
        r"\byou\s+should\s+(?:use|max\s+out)\s+your\s+credit\s+card\b",
        r"\byou\s+should\s+sell\s+your\b",
        r"\bmy\s+(?:legal\s+)?advice\s+(?:to\s+you\s+)?is\b",
        r"\bas\s+your\s+(?:lawyer|attorney|financial\s+advisor)\b",
    ),
)

# A negation cue this far back, in the same clause, clears the match.
_NEGATION_WINDOW = 70

_NEGATION_RE = re.compile(
    r"\b(?:not|no|nobody|none|never|cannot|can't|won't|wouldn't|shouldn't|isn't|aren't|"
    r"don't|doesn't|didn't|haven't|hasn't|unable|nothing|without|neither)\b",
    re.IGNORECASE,
)

_CLAUSE_BREAK_RE = re.compile(r"[,;:.!?]")


def _normalize_apostrophes(text: str) -> str:
    """Curly quotes in, straight quotes out — same length, so spans stay valid."""
    return text.replace("’", "'")


def is_negated(text: str, start: int) -> bool:
    """Does a negation cue precede ``start`` *in the same clause*?

    Public because the escalation detector needs the same test and had none:
    without it, "I'm not giving up on this" reads as distress and ends the call
    under A6. Splitting on clause breaks first is what stops a negation in one
    clause laundering a match in the next.
    """
    window = text[max(0, start - _NEGATION_WINDOW) : start]
    clause = _CLAUSE_BREAK_RE.split(window)[-1]
    return _NEGATION_RE.search(clause) is not None


_is_negated = is_negated


def scan_prohibited(candidate: str) -> tuple[Violation, ...]:
    """Scan a candidate *agent* utterance for prohibited persuasion.

    Never run this over consumer speech: a consumer asking "are you going to sue
    me?" is not a threat, and treating it as one would block the agent for the
    consumer's words.
    """
    haystack = _normalize_apostrophes(candidate)
    violations: list[Violation] = []
    for rule in _RULES:
        for match in rule.pattern.finditer(haystack):
            if rule.negatable and _is_negated(haystack, match.start()):
                continue
            violations.append(
                Violation(
                    rule_id=rule.rule_id,
                    severity=Severity.BLOCK,
                    span=candidate[match.start() : match.end()],
                    start=match.start(),
                    end=match.end(),
                    detail=f"{rule.rule_id}: {rule.detail}",
                )
            )
    return tuple(sorted(violations, key=lambda v: (v.start, v.rule_id)))


def is_prohibited(candidate: str) -> bool:
    """Convenience predicate. Callers that need the audit detail use ``scan_prohibited``."""
    return bool(scan_prohibited(candidate))
