"""Prohibited persuasion — during-call check 1.

The brief is explicit: persuasion that relies on threats, false urgency, or
invented consequences fails *regardless of how well it converts*. So this runs
on every candidate agent utterance before TTS and a trip blocks the turn.

Detection is deliberately conservative about negated phrasing. An agent
answering "no, nobody is going to sue you" is doing exactly the right thing, and
flagging it would train the model out of the reassurance consumers most need to
hear. A matched phrase is therefore cleared when a negation cue appears earlier
in the same clause.

This module also carries the shared violation vocabulary (``Severity``,
``Violation``) used by the other three checks: the package is fixed at four
modules, so the common types live with the first check rather than in a fifth
file.
"""

from __future__ import annotations

import re
import unicodedata
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
        r"\bdon'?t\s+stick\s+around\b",
        r"\bcan'?t\s+promise\s+(?:this|that|it)?\s*(?:option|offer|deal|rate)?\s*"
        r"will\s+(?:still\s+)?be\s+here\b",
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
        r"\bescalate\s+for\s+you\b",
        r"\bconsequences\s+to\s+(?:letting\s+this\s+go\s+further|this|not\s+paying)\b",
        r"\byou\s+won'?t\s+like\s+(?:them|it|what\s+happens)\b",
        r"\b(?:affects?|hurts?|damages?)\s+your\s+standing\s+with\s+lenders\b",
        r"\bnext\s+step\s+isn'?t\s+one\s+you\s+want\b",
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
        "gives legal or financial advice the agent is not licensed to give",
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

# A clause boundary for negation-scoping purposes is not just punctuation.
# Two run-on constructions launder a threat past the punctuation-only check:
#
#   "This isn't me being difficult, but if you can't work with me today
#    we will garnish your wages."
#
# The negation cue ("isn't"/"can't") sits in a *different* clause from the
# threat ("we will garnish") even though nothing but a comma-free coordinating
# or subordinating conjunction separates them. And within an "if"-clause
# itself, a new-subject assertion ("we will", "I'll", "they're going to")
# hands the sentence off to a different actor's promise, which is exactly the
# threat this rule exists to catch — so that handoff is a clause break too,
# even mid-clause.
_CLAUSE_BREAK_RE = re.compile(
    r"[,;:.!?]"
    r"|\b(?:and|but|if|when|unless|while)\b"
    # won'?t is deliberately excluded here: it is the negation cue _NEGATION_RE
    # relies on for this same branch, not a new-subject clause boundary, and
    # including it lets .split() consume and discard it before is_negated
    # ever sees it (e.g. "We won't sue you." would wrongly block).
    r"|\b(?:we|i|you|they|it)\s+(?:will|are\s+going\s+to|is\s+going\s+to)\b"
    r"|\b(?:we|i|you|they|it)'ll\b",
    re.IGNORECASE,
)


def _normalize_apostrophes(text: str) -> str:
    """Curly quotes in, straight quotes out — same length, so spans stay valid."""
    return text.replace("’", "'")


# -- evasion normalization --------------------------------------------------
#
# Two ways an otherwise verbatim-matching threat slips past every rule above:
# a Unicode format/invisible character breaking word adjacency ("sue[ZWSP]
# you"), or a visually-identical character from another script standing in
# for a Latin letter ("we will ѕue you", Cyrillic U+0455). Neither trips
# anything until the candidate is normalized first (ADVERSARIAL_TESTING.md
# H3). NFKC alone does not catch the second case: Cyrillic 's' is not a
# compatibility decomposition of Latin 's', so it is folded by an explicit
# lookalike table instead.

_SUSPICIOUS_CATEGORIES = frozenset({"Cf", "Cc", "Zs", "Zl", "Zp"})

# The classic IDN-homograph set: Cyrillic and Greek letters visually
# indistinguishable from a Latin one at TTS/reading speed.
_CONFUSABLES: dict[str, str] = {
    "а": "a",
    "А": "A",
    "в": "b",
    "В": "B",
    "е": "e",
    "Е": "E",
    "і": "i",
    "І": "I",
    "к": "k",
    "К": "K",
    "м": "m",
    "М": "M",
    "н": "h",
    "Н": "H",
    "о": "o",
    "О": "O",
    "р": "p",
    "Р": "P",
    "с": "c",
    "С": "C",
    "ѕ": "s",
    "Ѕ": "S",
    "т": "t",
    "Т": "T",
    "у": "y",
    "У": "Y",
    "х": "x",
    "Х": "X",
    "α": "a",
    "β": "b",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "υ": "u",
    "χ": "x",
}


def _strip_invisible(text: str) -> str:
    """Neutralize evasion whitespace — replaced with a space, never deleted.

    Deleting "we will[ZWSP]garnish" would join it into "we willgarnish" and
    kill the rule's own \\b word-boundary anchors, opening a *worse* bypass
    than the one this normalization exists to close. A space in its place
    keeps "garnish" a real word while still breaking whatever adjacency the
    evasion character was relying on.
    """
    return "".join(
        " " if ch != " " and unicodedata.category(ch) in _SUSPICIOUS_CATEGORIES else ch
        for ch in text
    )


def _fold_confusables(text: str) -> str:
    return "".join(_CONFUSABLES.get(ch, ch) for ch in text)


def _normalize_for_matching(text: str) -> str:
    text = _normalize_apostrophes(text)
    text = unicodedata.normalize("NFKC", text)
    text = _strip_invisible(text)
    return _fold_confusables(text)


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


def scan_prohibited(candidate: str) -> tuple[Violation, ...]:
    """Scan a candidate *agent* utterance for prohibited persuasion.

    Never run this over consumer speech: a consumer asking "are you going to sue
    me?" is not a threat, and treating it as one would block the agent for the
    consumer's words.
    """
    # ``haystack`` may differ in length from ``candidate`` once invisible
    # characters are stripped, so spans below are reported against the
    # normalized text, not resliced from the original — still an accurate,
    # readable record of what was actually said once evasion is undone.
    haystack = _normalize_for_matching(candidate)
    violations: list[Violation] = []
    for rule in _RULES:
        for match in rule.pattern.finditer(haystack):
            if rule.negatable and is_negated(haystack, match.start()):
                continue
            violations.append(
                Violation(
                    rule_id=rule.rule_id,
                    severity=Severity.BLOCK,
                    span=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    detail=f"{rule.rule_id}: {rule.detail}",
                )
            )
    return tuple(sorted(violations, key=lambda v: (v.start, v.rule_id)))


def is_prohibited(candidate: str) -> bool:
    """Convenience predicate. Callers that need the audit detail use ``scan_prohibited``."""
    return bool(scan_prohibited(candidate))
