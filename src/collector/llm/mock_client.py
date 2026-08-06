"""A scripted stand-in for the model — SPEC §3, §7.1.

Rule-based and deterministic, so the whole agent loop runs in CI with an empty
``.env``. It is not a simulation of Claude and does not try to be; it is a
conversational partner faithful enough to exercise the parts that matter — the
tool round trip, the disclosure ordering, the regeneration path, and the rule
that every spoken figure came back from the engine.

It holds no state. Everything it decides comes from the message list it is
handed, the same input a real client gets, so the agent loop cannot come to
depend on a client remembering anything.

Two things it does deliberately, because they are the behaviours under test:

- It reads amounts out of ``you_may_say`` in the tool result, never out of its
  own arithmetic. Ask it to say something the engine did not return and it
  cannot.
- On a regeneration note it falls back to figure-free phrasing rather than
  rewording the same blocked sentence, which is what a guarded retry should do.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal

from collector.guardrails.disclosures import (
    AI_DISCLOSURE_TEXT,
    MINI_MIRANDA_TEXT,
    confirms_identity,
    fires_mini_miranda,
)
from collector.guardrails.numeric import Figure, FigureKind, extract_figures
from collector.llm.base import LLMResponse, Message, ToolCall

# -- reading the consumer --------------------------------------------------

_AGREE_RE = re.compile(
    r"\b(yes|yeah|yep|ok(ay)?|sure|fine|deal|agreed?|that works|i can do that|"
    r"let'?s do (that|it)|sounds good|i'?ll take it|sign me up)\b",
    re.IGNORECASE,
)
_REFUSE_RE = re.compile(
    r"\b(no|nope|can'?t|cannot|won'?t|not going to|too much|too high|forget it|"
    r"impossible|absurd|ridiculous|out of the question|that doesn'?t work|"
    r"no way|not happening)\b",
    re.IGNORECASE,
)
_DONE_RE = re.compile(
    r"\b(goodbye|bye|hang(ing)? up|i'?m done|stop calling|leave me alone|"
    r"we'?re done|end the call)\b",
    re.IGNORECASE,
)
_PER_PAYMENT_RE = re.compile(
    r"(\b(a|per|every|each)\s+(week|month|paycheck|fortnight)\b|"
    r"\b(weekly|monthly|biweekly|bi-weekly|fortnightly)\b|/\s*(wk|mo))",
    re.IGNORECASE,
)
_CADENCE_WORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(bi-?weekly|fortnight|every (other|two) weeks?|paycheck)\b", re.I), "biweekly"),
    (re.compile(r"\b(week(ly)?|a week|per week|/\s*wk)\b", re.I), "weekly"),
    (re.compile(r"\b(month(ly)?|a month|per month|/\s*mo)\b", re.I), "monthly"),
    (re.compile(r"\b(now|today|right away|up ?front|lump sum|one (payment|shot|go))\b", re.I),
     "immediate"),
)
_COUNT_RE = re.compile(
    r"\b(?:in|over|across|split (?:it )?(?:in)?to|make it)\s+"
    r"(\d{1,2}|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:payments?|installments?|instalments?|chunks?|parts?)",
    re.IGNORECASE,
)
_WORD_COUNTS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
# "for a year" carries no numeral, so the guardrail's figure extractor does not
# see it — correctly, since it exists to catch figures in *agent* speech. Here
# it is the whole proposal, so the mock reads the article as "one".
_INDEFINITE_DURATION_RE = re.compile(
    r"\b(?:for|over|across|about|next)\s+(?:a|an|one)\s+(week|month|year)\b", re.IGNORECASE
)
_UNIT_DAYS = {"week": 7, "month": 30, "year": 365}
_INTERVALS = {"weekly": 7, "biweekly": 14, "monthly": 30}


@dataclass(frozen=True)
class ParsedProposal:
    """What the consumer appears to be proposing. Never a ruling on it."""

    total: Decimal | None
    payment_count: int
    cadence: str
    signaled_capacity: Decimal | None


def parse_proposal(text: str) -> ParsedProposal | None:
    """Read a proposal out of ordinary speech, or return ``None``.

    Uses the guardrail's own figure extractor, so what the mock can hear and
    what the guard can see are the same thing.
    """
    figures = extract_figures(text)
    amounts = [f.value for f in figures if f.kind is FigureKind.MONEY and f.value is not None]
    span_days = _span_days(text, figures)
    stated_cadence = _cadence_of(text)
    per_payment = _PER_PAYMENT_RE.search(text) is not None

    count = _explicit_count(text)
    if count is None and span_days is not None and stated_cadence in _INTERVALS:
        # "for a year", on a cadence: how many payments that actually implies.
        count = max(1, int(span_days // _INTERVALS[stated_cadence]))
    if count is None:
        count = 1

    # Only a schedule has a cadence. "I can give you $50" is a lump sum, and
    # calling it monthly would put a recurrence in the record they never said.
    cadence = stated_cadence or ("monthly" if count > 1 or per_payment else "immediate")

    if not amounts:
        # A shape with no sum — "weekly for a year". Still a proposal, and one
        # the engine has plenty to say about: they named no discount, so the
        # balance stands and only the structure is in question.
        if span_days is None and _explicit_count(text) is None:
            return None
        return ParsedProposal(
            total=None, payment_count=count, cadence=cadence, signaled_capacity=None
        )

    amount = amounts[0]
    if per_payment or count > 1:
        return ParsedProposal(
            total=amount * count,
            payment_count=count,
            cadence=cadence,
            signaled_capacity=amount,
        )
    return ParsedProposal(
        total=amount, payment_count=1, cadence=cadence, signaled_capacity=amount
    )


def _span_days(text: str, figures: tuple[Figure, ...]) -> Decimal | None:
    """How long they said this would run, in days, however they phrased it."""
    for figure in figures:
        if figure.kind is FigureKind.DURATION:
            return figure.value_in_days
    match = _INDEFINITE_DURATION_RE.search(text)
    return Decimal(_UNIT_DAYS[match.group(1).lower()]) if match else None


def _cadence_of(text: str) -> str | None:
    """The cadence they actually named, or ``None`` if they named none."""
    for pattern, cadence in _CADENCE_WORDS:
        if pattern.search(text):
            return cadence
    return None


def _explicit_count(text: str) -> int | None:
    match = _COUNT_RE.search(text)
    if match is None:
        return None
    token = match.group(1).lower()
    return _WORD_COUNTS.get(token, int(token) if token.isdigit() else None)


# -- the client ------------------------------------------------------------


class MockLLMClient:
    """Deterministic ``LLMClient``. Same messages in, same response out, always."""

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        last = messages[-1] if messages else None

        if last is not None and last.role == "system" and len(messages) > 1:
            return LLMResponse(text=self._after_block(last.content))
        if last is not None and last.role == "tool":
            return LLMResponse(text=self._speak_from_tool(last.content))
        if not self._has_spoken(messages):
            return LLMResponse(text=self._opening())
        if last is not None and last.role == "consumer":
            return self._on_consumer(last.content, messages)
        return LLMResponse(text="Take your time.")

    # -- turns -------------------------------------------------------------

    def _opening(self) -> str:
        """Identity first, and the AI disclosure with it. No account details:
        nothing substantive may be said before they confirm who they are."""
        return (
            f"Hi, this is Avery calling from Meridian Recovery Services. {AI_DISCLOSURE_TEXT} "
            "Am I speaking with the account holder?"
        )

    def _on_consumer(self, said: str, messages: tuple[Message, ...]) -> LLMResponse:
        # Identity, then the Mini-Miranda, then anything about the account. The
        # order is load-bearing: the disclosure itself says "collect a debt",
        # which is substantive, so firing it before identity is confirmed is
        # blocked just as surely as quoting the balance would be.
        if not self._identity_confirmed(messages):
            return LLMResponse(text="I'm sorry — am I speaking with the account holder?")
        if not self._disclosed(messages):
            return LLMResponse(
                text=(
                    f"Thank you. {MINI_MIRANDA_TEXT} I'm calling about an overdue "
                    "account, and I'd like to find something that works for you."
                )
            )

        if _DONE_RE.search(said):
            return LLMResponse(tool_calls=(ToolCall(name="end_call", arguments={
                "reason": "consumer ended the call"
            }),))

        proposal = parse_proposal(said)
        if proposal is not None:
            arguments: dict[str, object] = {
                "payment_count": proposal.payment_count,
                "cadence": proposal.cadence,
            }
            if proposal.total is not None:
                arguments["total"] = str(proposal.total)
            if proposal.signaled_capacity is not None:
                arguments["signaled_capacity"] = str(proposal.signaled_capacity)
            return LLMResponse(tool_calls=(
                ToolCall(name="validate_consumer_offer", arguments=arguments),
            ))

        offer_standing = self._standing_offer(messages) is not None
        if _AGREE_RE.search(said) and offer_standing:
            return LLMResponse(tool_calls=(ToolCall(name="confirm_agreement"),))
        if _REFUSE_RE.search(said):
            if offer_standing:
                return LLMResponse(tool_calls=(
                    ToolCall(name="record_refusal"),
                    ToolCall(name="concede"),
                ))
            return LLMResponse(tool_calls=(ToolCall(name="record_refusal"),))
        if not offer_standing:
            return LLMResponse(tool_calls=(ToolCall(name="propose_offer"),))
        return LLMResponse(text="I hear you. What would work on your side?")

    def _after_block(self, note: str) -> str:
        """A blocked turn is not an invitation to reword the same claim.

        Dropping every figure is the one retry guaranteed to clear the numeric
        guard, and handing the turn back to the consumer is better collections
        practice than talking through a compliance problem.
        """
        return "Let me put that a different way. What would be manageable for you?"

    # -- reading the conversation back -------------------------------------

    def _speak_from_tool(self, content: str) -> str:
        payload = _load(content)
        if not payload.get("ok"):
            return "Give me one moment — let me check what I can do here."

        if payload.get("agreed"):
            agreement = payload.get("agreement") or {}
            return f"{self._describe(agreement)} I've got that recorded. Thank you."

        offer = payload.get("offer_on_the_table")
        if payload.get("outcome") == "accept":
            return f"{self._describe(offer)} Can I get your okay on that?"
        if payload.get("moved") is False:
            # Nothing below this is available. Restating the same terms as a
            # concession is worse than saying plainly that this is the floor.
            return (
                f"I've gone as far as I'm able to on this one. {self._describe(offer)} "
                "Is there any way that works?"
            )
        if offer:
            return f"{self._reason(payload.get('rationale_code'))} {self._describe(offer)}"
        return "Thank you for that. Let me see what I can do."

    def _describe(self, offer: object) -> str:
        """Read back an engine-authored offer using only its own figures."""
        if not isinstance(offer, dict):
            return "Here's where we are."
        total = _dollars(offer.get("total"))
        count = int(offer.get("payment_count") or 1)
        cadence = str(offer.get("cadence", "monthly"))
        schedule = offer.get("schedule") or []

        if count == 1:
            return f"That's {total}, in one payment."
        first = _dollars(schedule[0].get("amount")) if schedule else total
        if len({str(i.get("amount")) for i in schedule}) == 1:
            return f"That's {total}, as {count} {cadence} payments of {first}."
        rest = ", then ".join(_dollars(i.get("amount")) for i in schedule[1:])
        return f"That's {total}: {first} to start, then {rest}, {cadence}."

    def _reason(self, code: object) -> str:
        """Plain words for a rationale code. The engine decided; this only says so."""
        return {
            "BELOW_SETTLEMENT_FLOOR": "I can't go that low on the balance, unfortunately.",
            "BELOW_MIN_PAYMENT": "Those payments are smaller than I'm able to set up.",
            "TOO_MANY_PAYMENTS": "That's more payments than I'm able to split it into.",
            "SCHEDULE_TOO_LONG": "I can't stretch it out that far, I'm afraid.",
            "CADENCE_NOT_OFFERED": "I can't set it up on that schedule.",
            "DISCOUNT_NOT_AUTHORIZED": "I'm not able to reduce the balance on that arrangement.",
        }.get(str(code), "Here's what I can do instead.")

    def _has_spoken(self, messages: tuple[Message, ...]) -> bool:
        return any(m.role == "agent" for m in messages)

    def _identity_confirmed(self, messages: tuple[Message, ...]) -> bool:
        """Confirmed once they have answered the opening question affirmatively."""
        return any(
            m.role == "consumer" and confirms_identity(m.content)
            for m in messages
        )

    def _disclosed(self, messages: tuple[Message, ...]) -> bool:
        """Has the Mini-Miranda already been spoken? Asked of the transcript
        rather than remembered, since this client keeps no state."""
        return any(
            m.role == "agent" and fires_mini_miranda(m.content) is not None for m in messages
        )

    def _standing_offer(self, messages: tuple[Message, ...]) -> dict[str, object] | None:
        for message in reversed(messages):
            if message.role != "tool":
                continue
            payload = _load(message.content)
            offer = payload.get("offer_on_the_table") or payload.get("agreement")
            if isinstance(offer, dict):
                return offer
        return None


def _load(content: str) -> dict[str, object]:
    try:
        loaded = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _dollars(value: object) -> str:
    """Format an engine amount for speech. Formatting only — no arithmetic."""
    return f"${Decimal(str(value)):,.2f}"
