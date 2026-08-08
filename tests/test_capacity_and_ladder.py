"""Regression: the offer that would not move (live call, 2026-08-07).

Two independent defects put the *same* counter back to a consumer who had just
raised what they could pay, verbatim: "$250 today and $750 in thirty days" twice.

1. ``effective_capacity`` ratchets. An inferred figure may only *lower* a stated
   one, so a consumer who says $150 and then says $300 is still scored at $150.
2. Nothing deterministic walks the ladder. ``select_tier`` returns
   ``state.ladder_floor`` and only ``concede`` ever moves it, so refusals pile up
   in ``pending_refusals`` while the engine re-proposes the same tier.

Every test here is stated behaviourally — what the consumer hears back — so a fix
is free to land in the engine or the tool layer. A4 (the ladder never walks back
up) is asserted alongside, so no fix may buy movement by giving it up.
"""

from __future__ import annotations

from collector.decision_engine import build_counter, effective_capacity
from collector.llm.base import ToolCall
from collector.money import Money
from collector.negotiation import NegotiationState
from collector.offers import Cadence, ConsumerProposal, Offer, Tier
from collector.policy import PolicyConfig
from collector.tools import ToolContext, ToolResult, execute

POLICY = PolicyConfig.default()

# The live call, as the model relayed it. The consumer's figures arrive as
# ``total`` — they named a sum, not a capacity — which is the channel both
# defects run through.
SAID_150 = {"total": "150", "payment_count": 1, "cadence": "immediate"}
SAID_300 = {"total": "300", "payment_count": 1, "cadence": "immediate"}


def _schedule(offer: Offer) -> tuple[tuple[Money, int], ...]:
    """The offer as the consumer hears it: amounts and when they are due."""
    return tuple((i.amount, i.due_day_offset) for i in offer.installments)


def _call(context: ToolContext, name: str, **arguments: object) -> ToolResult:
    result = execute(ToolCall(name=name, arguments=arguments or {}), context)
    assert result.ok, f"{name} failed: {result.payload}"
    return result


class TestCapacityRatchet:
    """A consumer who raises their number must be heard raising it."""

    def test_a_consumer_who_raises_their_figure_is_scored_at_the_new_one(self) -> None:
        """$150 on the record, then "I can only do three hundred dollars".

        ``decision_engine.py:310`` does ``min(inferred, state.signaled_capacity)``,
        so $300 is clamped back to $150 and every counter built after it is built
        on a figure the consumer has already moved past.
        """
        state = NegotiationState.opening(POLICY).with_capacity(Money("150"))
        raised = ConsumerProposal(total=Money("300"), payment_count=1, cadence=Cadence.IMMEDIATE)

        assert effective_capacity(raised, state) == Money("300"), (
            "a lump sum the consumer named is money they can produce; it is a "
            "floor under their capacity, not a ceiling over it"
        )

    def test_the_same_counter_is_not_put_back_after_the_consumer_raises(self) -> None:
        """The live call, replayed end to end through the tool layer.

        Consumer refuses $1,000, says $150, is countered $250/$750, says $300 —
        and hears $250/$750 again. The user's report, verbatim: "the agent keeps
        going to 250 today and 750 in another payment".
        """
        context = ToolContext.opening(POLICY)
        context = _call(context, "propose_offer").context
        context = _call(context, "record_refusal").context
        context = _call(context, "validate_consumer_offer", **SAID_150).context

        first = _call(context, "concede")
        context = first.context
        assert first.offer is not None
        assert _schedule(first.offer) == ((Money("250.00"), 0), (Money("750.00"), 30)), (
            "precondition drifted: this is the counter the live call produced"
        )

        second = _call(context, "validate_consumer_offer", **SAID_300)
        assert second.offer is not None
        assert _schedule(second.offer) != _schedule(first.offer), (
            "the consumer raised what they could pay from $150 to $300 and the "
            f"offer did not move: {_schedule(second.offer)}"
        )

    def test_a_three_hundred_dollar_consumer_is_asked_for_three_hundred_today(self) -> None:
        """The downpayment is "everything they can put down now" — so at a
        capacity of $300 it is $300, not the $250 floor.

        ``build_counter`` computes ``down = min(max(capacity, min_payment),
        total - min_payment)``: at capacity $150 that is ``max(150, 250) = 250``,
        and at the $300 they actually named it is $300/$700.
        """
        context = ToolContext.opening(POLICY)
        context = _call(context, "propose_offer").context
        context = _call(context, "record_refusal").context
        context = _call(context, "validate_consumer_offer", **SAID_150).context
        context = _call(context, "concede").context

        result = _call(context, "validate_consumer_offer", **SAID_300)
        assert result.offer is not None
        assert result.offer.tier is Tier.DOWNPAYMENT_PLUS_ONE
        assert _schedule(result.offer) == ((Money("300.00"), 0), (Money("700.00"), 30))

    def test_an_inferred_figure_still_may_not_talk_over_a_stated_one(self) -> None:
        """The bug the clamp was written for. Must keep passing after any fix.

        A proposal that names no sum at all is defaulted to the whole balance
        (``tools.py:438``), so its "smallest payment" is arithmetic on a default:
        $1,000 over two payments infers $500 from a consumer who said $300. That
        figure is a ceiling read off a shape, and it must not raise a stated one —
        unclamped it counters $500/$500 at someone who told us $300.
        """
        state = NegotiationState.opening(POLICY).with_capacity(Money("300"))
        shape_only = ConsumerProposal(
            total=POLICY.original_balance,  # defaulted, not named
            payment_count=2,
            cadence=Cadence.MONTHLY,
        )

        assert shape_only.smallest_payment == Money("500.00"), "precondition"
        assert effective_capacity(shape_only, state) == Money("300.00"), (
            "$300 said out loud became a counter built on $500"
        )


class TestLadderAdvancesUnderRepeatedRefusal:
    """Refusals must buy movement. Accumulating them must not be the whole
    effect of refusing."""

    def _refuse_until_exhausted(self) -> list[Offer]:
        """The consumer keeps naming terms we cannot meet and never accepts.

        ``_validate_consumer_offer`` calls ``record_refusal()`` on every
        non-accepted proposal (``tools.py:484``), so the refusals are
        unambiguously on record; nothing else in the loop is required.
        """
        context = ToolContext.opening(POLICY)
        offers: list[Offer] = []
        while not context.state.is_exhausted:
            result = _call(context, "validate_consumer_offer", **SAID_300)
            context = result.context
            assert result.offer is not None
            offers.append(result.offer)
        assert context.state.pending_refusals > 0, "precondition: refusals were recorded"
        return offers

    def test_repeated_refusal_does_not_re_offer_the_same_tier_forever(self) -> None:
        offers = self._refuse_until_exhausted()
        tiers = [o.tier for o in offers]

        assert len(set(tiers)) > 1, (
            f"{len(offers)} refusals on the record and the offer never left "
            f"{tiers[0].label}: {[_schedule(o) for o in offers[:3]]}"
        )

    def test_sustained_refusal_walks_the_ladder_down_to_a_settlement(self) -> None:
        """One refusal buys one step (A7), so a consumer who refuses through the
        round cap must be walked past pay-in-full and down to real concessions
        rather than held at the top of the ladder until the call closes out."""
        offers = self._refuse_until_exhausted()
        walked = sorted(set(o.tier for o in offers))

        assert walked[0] is Tier.PAY_IN_FULL, "the ladder must still open at the top"
        assert walked[-1] >= Tier.SETTLEMENT, (
            f"ladder stalled at {walked[-1].label} across {len(offers)} refusals"
        )

    def test_the_ladder_never_walks_back_up_while_it_descends(self) -> None:
        """A4. The fix buys movement downward; it may not buy it by re-opening a
        tier already conceded past."""
        tiers = [o.tier for o in self._refuse_until_exhausted()]

        assert tiers == sorted(tiers), f"ladder went back up: {[t.label for t in tiers]}"

    def test_a_four_by_250_plan_is_within_reach_of_a_300_dollar_consumer(self) -> None:
        """What the ladder has left to give, so "stuck at $250/$750" is a real
        loss and not a shortage of legal options.

        ``_allocate(1000, 4, floor=250, capacity=300)`` tries a capacity-led
        split first — $300 now and $700 over three — but that rest is $233.34 /
        $233.33 / $233.33, below the $250 floor, so it falls back to an even
        4 x $250 over 90 days. Every instalment clears the $300 they named.
        """
        state = NegotiationState.opening(POLICY).conceded_to(Tier.PAYMENT_PLAN)
        offer = build_counter(state, POLICY, capacity=Money("300"))

        assert offer.payment_count == 4
        assert offer.total == POLICY.original_balance
        assert all(i.amount == Money("250.00") for i in offer.installments)
        assert offer.duration_days == 90 <= POLICY.max_plan_days
        assert max(i.amount for i in offer.installments) <= Money("300.00")
