"""A scripted stand-in for the model.

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
    requests_ai_disclosure,
)
from collector.guardrails.numeric import Figure, FigureKind, extract_figures
from collector.llm.base import LLMResponse, Message, ToolCall

# -- reading the consumer --------------------------------------------------

_AGREE_RE = re.compile(
    r"\b(yes|yeah|yep|ok(ay)?|sure|fine|deal|agreed?|that works|i can do that|"
    r"let'?s do (that|it)|sounds good|i'?ll take it|sign me up)\b",
    re.IGNORECASE,
)
# _AGREE_RE matches its bare tokens ("sure", "fine", "deal", "agreed") on word
# boundary alone, so "I am not sure" reads as consent — which previously
# closed the call AGREED off pure uncertainty. Mirrors
# guardrails.disclosures.confirms_identity: a negation elsewhere in the
# utterance wins over the bare affirmation it's attached to. Bare "no" is
# deliberately not a blanket negator here — it's ordinary negotiation
# vocabulary ("no, that's fine, go ahead") and would suppress more genuine
# agreement than it would catch false positives — "no deal" is the one bare-
# "no" idiom worth catching explicitly.
_NEGATED_AGREE_RE = re.compile(
    r"\b(?:not|never|don'?t|doesn'?t|didn'?t|haven'?t)\b[^.!?,]{0,20}?"
    r"\b(?:sure|fine|ok(?:ay)?|deal|agreed?)\b"
    r"|\bno\s+deal\b",
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
# A plain factual question, not a proposal — parse_proposal correctly returns
# None for it (no figures, no span, no explicit count), and it must not be
# treated as generic filler either: it deserves a direct answer, not "here's
# what I can do instead" or a re-prompt that ignores what was actually asked.
_BALANCE_INQUIRY_RE = re.compile(
    r"\bhow much (?:do|does|would) i (?:owe|need to pay)\b"
    r"|\bwhat (?:do|does) i owe\b"
    r"|\bwhat'?s (?:my|the) balance\b"
    r"|\bwhat is (?:my|the) balance\b"
    r"|\bhow much is owed\b"
    r"|\bwhat'?s the amount\b",
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
    (
        re.compile(r"\b(now|today|right away|up ?front|lump sum|one (payment|shot|go))\b", re.I),
        "immediate",
    ),
)
_COUNT_RE = re.compile(
    r"\b(?:in|over|across|split (?:it )?(?:in)?to|make it)\s+"
    r"(\d{1,2}|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:payments?|installments?|instalments?|chunks?|parts?)",
    re.IGNORECASE,
)
_WORD_COUNTS = {
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
    return ParsedProposal(total=amount, payment_count=1, cadence=cadence, signaled_capacity=amount)


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
            return self._speak_from_tool(last.content)
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
        # Asking who/what they're talking to is not "substantive collection
        # discussion" (that's the disclosure guard's own definition), so it
        # is answered ahead of both gates below rather than behind them. Left
        # behind the gates, an unanswered request trips
        # AI_DISCLOSURE_REQUEST_IGNORED on every later candidate — including
        # ones about something else entirely — and the call never recovers
        # since nothing here ever actually fires the disclosure again.
        if requests_ai_disclosure(said):
            if not self._identity_confirmed(messages):
                return LLMResponse(
                    text=f"{AI_DISCLOSURE_TEXT} Am I speaking with the account holder?"
                )
            if not self._disclosed(messages):
                return LLMResponse(
                    text=(
                        f"{AI_DISCLOSURE_TEXT} {MINI_MIRANDA_TEXT} I'm calling about an "
                        "overdue account, and I'd like to find something that works for you."
                    )
                )
            return LLMResponse(
                text=f"{AI_DISCLOSURE_TEXT} Now, back to your account — what would work for you?"
            )

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
            return LLMResponse(
                tool_calls=(
                    ToolCall(name="end_call", arguments={"reason": "consumer ended the call"}),
                )
            )

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
            return LLMResponse(
                tool_calls=(ToolCall(name="validate_consumer_offer", arguments=arguments),)
            )

        offer_standing = self._standing_offer(messages) is not None
        if _AGREE_RE.search(said) and offer_standing and not _NEGATED_AGREE_RE.search(said):
            return LLMResponse(tool_calls=(ToolCall(name="confirm_agreement"),))
        if _REFUSE_RE.search(said):
            if offer_standing:
                return LLMResponse(
                    tool_calls=(
                        ToolCall(name="record_refusal"),
                        ToolCall(name="concede"),
                    )
                )
            # Nothing is on the table yet, so there is nothing to refuse —
            # record_refusal alone left no offer, no change in state, and no
            # way for a repeat of the same refusal to produce anything but
            # the identical reply forever. Treated as any other opening
            # reply: put a first offer up.
            return LLMResponse(tool_calls=(ToolCall(name="propose_offer"),))
        if _BALANCE_INQUIRY_RE.search(said):
            standing = self._standing_offer(messages)
            if standing is not None:
                # A standing offer may already be a concession below the
                # original balance — reading it back as "you currently owe"
                # would misstate it as the debt itself.
                return LLMResponse(text=self._describe(standing, lead="That's still"))
            return LLMResponse(tool_calls=(ToolCall(name="propose_offer"),))
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

    def _speak_from_tool(self, content: str) -> LLMResponse:
        payload = _load(content)
        if not payload.get("ok"):
            if "end_call" in str(payload.get("error", "")):
                # Every tool refuses this way once the round cap is hit
                # (tools.py's own instruction is to close out with end_call
                # rather than keep negotiating) — without this, the call has
                # no way to ever act on that refusal and just repeats a
                # filler line forever.
                return LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="end_call",
                            arguments={"reason": "negotiation round limit reached"},
                        ),
                    )
                )
            return LLMResponse(text="Give me one moment — let me check what I can do here.")

        if payload.get("agreed"):
            agreement = payload.get("agreement") or {}
            return LLMResponse(
                text=f"{self._describe(agreement)} I've got that recorded. Thank you."
            )

        offer = payload.get("offer_on_the_table")
        if payload.get("outcome") == "accept":
            return LLMResponse(text=f"{self._describe(offer)} Can I get your okay on that?")
        if payload.get("moved") is False:
            # Nothing below this is available. Restating the same terms as a
            # concession is worse than saying plainly that this is the floor.
            return LLMResponse(
                text=(
                    f"I've gone as far as I'm able to on this one. {self._describe(offer)} "
                    "Is there any way that works?"
                )
            )
        if offer:
            if "rationale_code" not in payload and "moved" not in payload:
                # A fresh propose_offer: nothing has been refused yet to
                # concede against, so this is the first figure spoken about
                # the account (whether opening the negotiation or answering
                # "how much do I owe"). "Here's what I can do instead" is
                # only true after a refusal — reusing it here misrepresents
                # the plain balance as a fallback concession.
                return LLMResponse(text=self._describe(offer, lead="You currently owe"))
            reason = self._reason(payload.get("rationale_code"))
            return LLMResponse(text=f"{reason} {self._describe(offer)}")
        return LLMResponse(text="Thank you for that. Let me see what I can do.")

    def _describe(self, offer: object, *, lead: str = "That's") -> str:
        """Read back an engine-authored offer using only its own figures."""
        if not isinstance(offer, dict):
            return "Here's where we are."
        total = _dollars(offer.get("total"))
        count = int(offer.get("payment_count") or 1)
        cadence = str(offer.get("cadence", "monthly"))
        schedule = offer.get("schedule") or []

        if count == 1:
            return f"{lead} {total}, in one payment."
        first = _dollars(schedule[0].get("amount")) if schedule else total
        if len({str(i.get("amount")) for i in schedule}) == 1:
            return f"{lead} {total}, as {count} {cadence} payments of {first}."
        rest = ", then ".join(_dollars(i.get("amount")) for i in schedule[1:])
        return f"{lead} {total}: {first} to start, then {rest}, {cadence}."

    def _reason(self, code: object) -> str:
        """Plain words for a rationale code. The engine decided; this only says so."""
        return {
            "BELOW_SETTLEMENT_FLOOR": "I can't go that low on the balance, unfortunately.",
            "BELOW_MIN_PAYMENT": "Those payments are smaller than I'm able to set up.",
            "TOO_MANY_PAYMENTS": "That's more payments than I'm able to split it into.",
            "SCHEDULE_TOO_LONG": "I can't stretch it out that far, I'm afraid.",
            "CADENCE_NOT_OFFERED": "I can't set it up on that schedule.",
            "DISCOUNT_NOT_AUTHORIZED": "I'm not able to reduce the balance on that arrangement.",
            "PREFERRED_TIER_AVAILABLE": "I'd like to see if we can do better than that.",
        }.get(str(code), "Here's what I can do instead.")

    def _has_spoken(self, messages: tuple[Message, ...]) -> bool:
        return any(m.role == "agent" for m in messages)

    def _identity_confirmed(self, messages: tuple[Message, ...]) -> bool:
        """Confirmed once they have answered the opening question affirmatively."""
        return any(m.role == "consumer" and confirms_identity(m.content) for m in messages)

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
    except json.JSONDecodeError, TypeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _dollars(value: object) -> str:
    """Format an engine amount for speech. Formatting only — no arithmetic."""
    return f"${Decimal(str(value)):,.2f}"
