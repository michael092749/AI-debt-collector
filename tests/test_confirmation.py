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
from collector.guardrails.confirmation import confirmation_line, repeats_back
from collector.guardrails.numeric import authorized_for, check_numeric
from collector.guardrails.rings import PreCallContext
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
