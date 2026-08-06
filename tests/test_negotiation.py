"""Negotiation ladder and concession tracking — SPEC build step 3.

The engine decides whether a given proposal is legal. This module decides how
fast the agent is allowed to give ground, and records what was exchanged so the
audit trail can reconstruct the call.
"""

from __future__ import annotations

import pytest

from collector.decision_engine import build_counter
from collector.money import Money
from collector.negotiation import CallOutcome, NegotiationState
from collector.offers import Cadence, ConsumerProposal, Tier
from collector.policy import PolicyConfig

POLICY = PolicyConfig.default()


def _state() -> NegotiationState:
    return NegotiationState.opening(POLICY)


def _proposal(total: str = "300", count: int = 1) -> ConsumerProposal:
    return ConsumerProposal(Money(total), count, Cadence.MONTHLY)


class TestLadderAdvance:
    def test_opens_at_the_most_preferred_tier(self) -> None:
        assert _state().ladder_floor is Tier.PAY_IN_FULL
        assert _state().outcome is CallOutcome.IN_PROGRESS

    def test_will_not_concede_before_the_consumer_refuses(self) -> None:
        """Concessions are earned, not volunteered. An agent that walks itself
        down the ladder unprompted gives away money for nothing."""
        s = _state()
        assert s.can_concede is False
        assert s.advance_ladder().ladder_floor is Tier.PAY_IN_FULL

    def test_advances_exactly_one_tier_per_refusal(self) -> None:
        s = _state().record_refusal()
        assert s.can_concede is True
        s = s.advance_ladder()
        assert s.ladder_floor is Tier.DOWNPAYMENT_PLUS_ONE
        # The refusal is spent; a second step needs a second refusal.
        assert s.can_concede is False
        assert s.advance_ladder().ladder_floor is Tier.DOWNPAYMENT_PLUS_ONE

    def test_walks_the_full_ladder_in_order(self) -> None:
        s = _state()
        seen = [s.ladder_floor]
        for _ in range(5):
            s = s.record_refusal().advance_ladder()
            seen.append(s.ladder_floor)
        assert seen[:4] == [
            Tier.PAY_IN_FULL,
            Tier.DOWNPAYMENT_PLUS_ONE,
            Tier.SETTLEMENT,
            Tier.PAYMENT_PLAN,
        ]

    def test_bottoms_out_at_payment_plan(self) -> None:
        """There is nothing below a no-discount plan. The agent must hold."""
        s = _state()
        for _ in range(10):
            s = s.record_refusal().advance_ladder()
        assert s.ladder_floor is Tier.PAYMENT_PLAN

    def test_ladder_never_moves_back_up(self) -> None:
        """A4. Applies even if something tries to concede to a better tier."""
        s = _state().record_refusal().advance_ladder().record_refusal().advance_ladder()
        assert s.ladder_floor is Tier.SETTLEMENT
        assert s.conceded_to(Tier.PAY_IN_FULL).ladder_floor is Tier.SETTLEMENT


class TestConcessionTracking:
    def test_records_each_round(self) -> None:
        s = _state()
        counter = build_counter(s, POLICY, capacity=Money("300"))
        s = s.record_round(_proposal("300"), counter, "BELOW_SETTLEMENT_FLOOR")
        assert s.round_count == 1
        assert s.rounds[0].proposal is not None
        assert s.rounds[0].counter == counter
        assert s.offers_made == (counter,)

    def test_capacity_signal_is_captured(self) -> None:
        s = _state().with_capacity(Money("400"))
        assert s.signaled_capacity == Money("400")

    def test_latest_capacity_wins(self) -> None:
        """Consumers revise. The most recent statement is the operative one."""
        s = _state().with_capacity(Money("400")).with_capacity(Money("250"))
        assert s.signaled_capacity == Money("250")

    def test_state_is_immutable_between_rounds(self) -> None:
        s = _state()
        s.record_refusal()
        assert s.can_concede is False, "record_refusal must not mutate in place"


class TestTerminalStates:
    def test_agreement_is_terminal(self) -> None:
        s = _state()
        offer = build_counter(s, POLICY, tier=Tier.PAY_IN_FULL)
        s = s.agree(offer)
        assert s.outcome is CallOutcome.AGREED
        assert s.agreed_offer == offer
        assert s.is_terminal

    def test_escalation_is_terminal_and_keeps_the_reason(self) -> None:
        s = _state().escalate("consumer disputes the debt")
        assert s.outcome is CallOutcome.ESCALATED
        assert s.escalation_reason == "consumer disputes the debt"
        assert s.is_terminal
        assert s.agreed_offer is None, "an escalated call agrees to nothing"

    def test_terminal_states_refuse_further_negotiation(self) -> None:
        """Once escalated, nothing may re-open the negotiation. Continuing to
        press a consumer who disputed the debt is exactly the failure mode the
        guardrails exist to prevent."""
        s = _state().escalate("hardship")
        for move in (
            lambda: s.record_refusal(),
            lambda: s.advance_ladder(),
            lambda: s.record_round(_proposal(), None, "X"),
            lambda: s.agree(build_counter(_state(), POLICY, tier=Tier.PAY_IN_FULL)),
        ):
            with pytest.raises(ValueError, match="terminal"):
                move()

    def test_no_agreement_is_terminal(self) -> None:
        s = _state().end_without_agreement()
        assert s.outcome is CallOutcome.NO_AGREEMENT
        assert s.is_terminal


class TestRoundCap:
    def test_negotiation_is_bounded(self) -> None:
        """An unbounded haggle is its own compliance problem — it becomes
        badgering. The cap is policy, not vibes."""
        s = _state()
        for _ in range(POLICY.max_negotiation_rounds):
            assert not s.is_exhausted
            s = s.record_round(_proposal(), None, "BELOW_SETTLEMENT_FLOOR")
        assert s.is_exhausted

    def test_exhausted_call_can_still_be_closed_out(self) -> None:
        s = _state()
        for _ in range(POLICY.max_negotiation_rounds):
            s = s.record_round(_proposal(), None, "X")
        assert s.end_without_agreement().outcome is CallOutcome.NO_AGREEMENT


class TestEngineIntegration:
    def test_counter_tier_tracks_the_ladder(self) -> None:
        """The engine reads ladder_floor, so conceding here changes what the
        engine is willing to put on the table."""
        s = _state().with_capacity(Money("300"))
        first = build_counter(s, POLICY, capacity=s.signaled_capacity)
        assert first.tier is Tier.DOWNPAYMENT_PLUS_ONE

        s = s.record_refusal().advance_ladder().record_refusal().advance_ladder()
        later = build_counter(s, POLICY, capacity=s.signaled_capacity)
        assert later.tier is Tier.SETTLEMENT
        assert later.total == Money("800.00"), "settlement unlocks the 20% discount"
