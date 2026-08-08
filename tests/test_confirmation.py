"""Repeat-back confirmation: the terms must be said back before they bind.

Three things are being proven here:

1. The engine authors the confirmation line, and every figure in it is one the
   numeric guard already authorizes for that offer. A canonical line the guard
   would block is worse than no canonical line.
2. The detector accepts paraphrase but not a wrong or missing amount. The
   model will not reproduce the line verbatim; it must not be able to skip an
   instalment either.
3. ``confirm_agreement`` does not record a commitment until the repeat-back has
   actually been spoken to the consumer.

Offline and deterministic: no API keys, no clock, no network.
"""

from __future__ import annotations

import pytest

from collector.agent import NegotiationAgent
from collector.guardrails.confirmation import agreed_line, confirmation_line, repeats_back
from collector.guardrails.disclosures import MINI_MIRANDA_TEXT
from collector.guardrails.numeric import authorized_for, check_numeric
from collector.guardrails.rings import SAFE_FALLBACK_TEXT, PreCallContext
from collector.llm.base import ToolCall
from collector.llm.mock_client import MockLLMClient
from collector.money import Money
from collector.negotiation import CallOutcome
from collector.offers import Cadence, Installment, Offer, Tier
from collector.policy import PolicyConfig
from collector.tools import ToolContext, execute

POLICY = PolicyConfig.default()


@pytest.fixture
def plan() -> Offer:
    """$800 over three monthly payments — the multi-instalment shape."""
    return Offer(
        tier=Tier.SETTLEMENT,
        installments=(
            Installment(Money("250.00"), 0),
            Installment(Money("250.00"), 30),
            Installment(Money("300.00"), 60),
        ),
        cadence=Cadence.MONTHLY,
    )


@pytest.fixture
def lump() -> Offer:
    """A single payment today — the degenerate shape."""
    return Offer(
        tier=Tier.PAY_IN_FULL,
        installments=(Installment(Money("500.00"), 0),),
        cadence=Cadence.IMMEDIATE,
    )


# --------------------------------------------------------------------------
# 1. The engine authors a line the guard permits


def test_confirmation_line_states_every_instalment(plan: Offer) -> None:
    line = confirmation_line(plan)
    assert "$250.00" in line
    assert "$300.00" in line
    assert line.rstrip().endswith("?"), "a confirmation that does not ask is not a confirmation"


def test_confirmation_line_clears_the_numeric_guard(plan: Offer) -> None:
    """Every figure in the canonical line is engine-authorized for that offer.

    This is the property that makes the line usable: the prompt can be told to
    say it exactly, and the pre-TTS guard will pass it.
    """
    authorized = authorized_for(POLICY).with_offer(plan)
    assert check_numeric(confirmation_line(plan), authorized) == ()


def test_confirmation_line_clears_the_guard_for_a_lump_sum(lump: Offer) -> None:
    authorized = authorized_for(POLICY).with_offer(lump)
    assert check_numeric(confirmation_line(lump), authorized) == ()


def test_the_canonical_line_satisfies_its_own_detector(plan: Offer, lump: Offer) -> None:
    """Renderer and detector must not drift apart."""
    assert repeats_back(confirmation_line(plan), plan)
    assert repeats_back(confirmation_line(lump), lump)


# --------------------------------------------------------------------------
# 2. Lenient on prose, exact on numbers


@pytest.mark.parametrize(
    "said",
    [
        "Just to confirm: $250.00 today, then $250.00 in 30 days, "
        "then $300.00 in 60 days. Should I set that up?",
        "So that's 250 today, 250 next month and 300 the month after — do I have that right?",
        "Let me confirm — two fifty now, two fifty in a month, three hundred after that. "
        "Is that correct?",
    ],
)
def test_paraphrase_counts_as_a_repeat_back(said: str, plan: Offer) -> None:
    assert repeats_back(said, plan)


@pytest.mark.parametrize(
    ("said", "why"),
    [
        (
            "So that's $250.00 today and $250.00 next month — is that right?",
            "drops the third instalment",
        ),
        (
            "Just to confirm: $250.00 today, $250.00 in 30 days, $350.00 in 60 days. "
            "Should I set that up?",
            "states an amount that is not in the offer",
        ),
        (
            "Your payments are $250.00, $250.00 and $300.00.",
            "recites the schedule but never asks for assent",
        ),
        (
            "Can you manage $250.00 today, $250.00 in 30 days and $300.00 in 60 days?",
            "proposes the terms rather than confirming them back",
        ),
    ],
)
def test_near_misses_are_not_a_repeat_back(said: str, why: str, plan: Offer) -> None:
    assert not repeats_back(said, plan), why


def test_a_repeat_back_of_a_different_offer_does_not_count(plan: Offer, lump: Offer) -> None:
    assert not repeats_back(confirmation_line(lump), plan)


# --------------------------------------------------------------------------
# 3. The gate: no commitment before the terms are said back


def _tabled() -> tuple[ToolContext, Offer]:
    """Put a genuinely engine-authored offer on the table.

    Built through ``propose_offer`` rather than by hand: ``confirm_agreement``
    re-validates the standing terms against policy, so a hand-rolled Offer
    fails on its own merits and proves nothing about the repeat-back gate.
    """
    result = execute(ToolCall(name="propose_offer", arguments={}), ToolContext.opening(POLICY))
    assert result.offer is not None
    return result.context, result.offer


def _ready() -> tuple[NegotiationAgent, Offer]:
    """An agent mid-call: identity confirmed, an offer on the table, nothing said."""
    agent = NegotiationAgent(llm=MockLLMClient(), policy=POLICY, store=None)
    agent.open_call(PreCallContext(account_loaded=True, within_calling_window=True))
    agent.guard = agent.guard.with_identity_confirmed()
    agent.tools, offer = _tabled()
    agent.spoken.clear()
    return agent, offer


def test_confirm_agreement_is_refused_before_the_repeat_back() -> None:
    agent, _ = _ready()

    result = agent._run_tool(ToolCall(name="confirm_agreement", arguments={}))

    assert not result.ok
    assert agent.tools.state.outcome is not CallOutcome.AGREED
    assert agent.tools.state.agreed_offer is None


def test_the_refusal_hands_back_the_line_to_say() -> None:
    agent, offer = _ready()

    result = agent._run_tool(ToolCall(name="confirm_agreement", arguments={}))

    assert result.payload.get("you_must_confirm") == confirmation_line(offer)


def test_confirm_agreement_succeeds_once_the_terms_were_spoken() -> None:
    agent, offer = _ready()
    agent._record_spoken(confirmation_line(offer))

    result = agent._run_tool(ToolCall(name="confirm_agreement", arguments={}))

    assert result.ok, result.payload
    assert agent.tools.state.outcome is CallOutcome.AGREED
    assert agent.tools.state.agreed_offer == offer


def test_a_repeat_back_of_other_terms_does_not_unlock_the_commitment(lump: Offer) -> None:
    """The gate is per-offer, not a latch. Confirming one arrangement out loud
    must not license booking a different one."""
    agent, offer = _ready()
    assert lump.total != offer.total
    agent._record_spoken(confirmation_line(lump))

    result = agent._run_tool(ToolCall(name="confirm_agreement", arguments={}))

    assert not result.ok
    assert agent.tools.state.agreed_offer is None


def test_the_gate_is_the_agents_and_not_the_tools() -> None:
    """``execute`` stays unguarded: the gate is a call-loop concern.

    ``tools.py`` is a pure function of its context and has no view of what was
    spoken, so the refusal belongs where the identity gate already lives.
    """
    context, _ = _tabled()
    result = execute(ToolCall(name="confirm_agreement", arguments={}), context)
    assert result.ok


def test_the_streaming_path_can_satisfy_the_gate() -> None:
    """The voice path speaks sentence by sentence and never calls
    ``_record_spoken``. Tracking spoken text one level above ``_record_audio``
    left ``spoken`` empty here, so the gate refused every agreement on the only
    path that carries real calls — and the unit tests above all still passed.
    """
    agent = NegotiationAgent(llm=MockLLMClient(), policy=POLICY, store=None)
    agent.open_call(PreCallContext(account_loaded=True, within_calling_window=True))

    for said in ["Yes, this is Dana.", "I could do $500 down.", "Okay, yes, let's do that."]:
        if agent.ended:
            break
        list(agent.stream_turn(said))

    assert agent.tools.state.outcome is CallOutcome.AGREED
    assert agent.tools.state.agreed_offer is not None


def test_offer_payloads_carry_the_line_to_say() -> None:
    """The model is handed the canonical string wherever an offer is tabled."""
    result = execute(ToolCall(name="propose_offer", arguments={}), ToolContext.opening(POLICY))
    offer = result.offer
    assert offer is not None
    assert result.payload["you_must_confirm"] == confirmation_line(offer)


# --------------------------------------------------------------------------
# 4. Reading a schedule aloud: equal runs group instead of repeating


@pytest.fixture
def penny_split() -> Offer:
    """$800 as 266.67/266.67/266.66 — the even-allocate shape a live call
    read out as three near-identical cents amounts in a row."""
    return Offer(
        tier=Tier.SETTLEMENT,
        installments=(
            Installment(Money("266.67"), 0),
            Installment(Money("266.67"), 30),
            Installment(Money("266.66"), 60),
        ),
        cadence=Cadence.MONTHLY,
    )


def test_equal_installments_group_instead_of_repeating(penny_split: Offer) -> None:
    """ "Two hundred sixty-six dollars and sixty-seven cents" three times in
    a row is a wall of digits to someone listening on a phone. The grouped
    form says the amount once and names the odd cent out."""
    line = confirmation_line(penny_split)
    assert line.count("$266.67") == 1, line
    assert "3 monthly payments of $266.67" in line
    assert "$266.66" in line
    assert "$800.00 in total" in line


def test_an_all_equal_schedule_groups_to_one_amount() -> None:
    offer = Offer(
        tier=Tier.PAYMENT_PLAN,
        installments=tuple(Installment(Money("250.00"), n * 30) for n in range(4)),
        cadence=Cadence.MONTHLY,
    )
    line = confirmation_line(offer)
    assert line.count("$250.00") == 1, line
    assert "4 monthly payments of $250.00" in line
    assert "$1,000.00 in total" in line


def test_a_grouped_line_still_clears_guard_and_detector(penny_split: Offer) -> None:
    assert repeats_back(confirmation_line(penny_split), penny_split)
    authorized = authorized_for(POLICY).with_offer(penny_split)
    assert check_numeric(confirmation_line(penny_split), authorized) == ()


def test_a_two_payment_schedule_stays_itemized() -> None:
    """With two unequal payments the offsets carry the story; grouping has
    nothing to compress and would only blur which amount lands when."""
    offer = Offer(
        tier=Tier.DOWNPAYMENT_PLUS_ONE,
        installments=(Installment(Money("250.00"), 0), Installment(Money("750.00"), 30)),
        cadence=Cadence.MONTHLY,
    )
    line = confirmation_line(offer)
    assert "$250.00 today" in line
    assert "$750.00 in 30 days" in line


# --------------------------------------------------------------------------
# 5. After acceptance: a blocked close must not reopen the negotiation


def test_the_agreed_line_restates_the_terms_without_reasking(penny_split: Offer) -> None:
    line = agreed_line(penny_split)
    assert "$266.67" in line
    assert "$266.66" in line
    assert not line.rstrip().endswith("?"), "the consumer already said yes"
    authorized = authorized_for(POLICY).with_offer(penny_split)
    assert check_numeric(line, authorized) == ()


def test_a_blocked_close_after_agreement_does_not_reopen_the_negotiation() -> None:
    """A live call took the consumer's "Yes.", closed the agreement, had its
    closing sentence blocked, and spoke the scripted restart — "What would
    work for you?" — over a deal that had just been made. The scripted line
    for a closed agreement is the agreement, not a reopened negotiation."""
    agent, offer = _ready()
    agent._observe_scripted(MINI_MIRANDA_TEXT)
    agent._record_spoken(confirmation_line(offer))
    result = agent._run_tool(ToolCall(name="confirm_agreement", arguments={}))
    assert result.ok
    assert agent.tools.state.outcome is CallOutcome.AGREED

    line = agent._fallback_line()
    assert line != SAFE_FALLBACK_TEXT
    assert line == agreed_line(offer)
