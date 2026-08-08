"""What the consumer is *told* when a per-payment figure is relayed.

The engine's rationale code is not bookkeeping: it is the sentence the consumer
hears. A code that is arithmetically reachable but factually wrong about their
money ("three hundred dollars is below the minimum I can accept", said to
someone whose figure clears the $250 floor) is a compliance defect in the same
way an illegal amount is, and it must be closed in code with a stable rule id
rather than in prose.

Every test here is a table of (what the consumer said, what the model relayed,
what they must not be told). No mocks, no I/O, no clock: the tool layer is
driven through ``execute`` and the engine through ``validate_offer``.

Reproduces the 2026-08-07 live call, where "a hundred and fifty dollars" was
answered with the over-collection rationale and "three hundred dollars" with the
settlement-floor rationale. Neither is true of the figure the consumer named.
"""

from __future__ import annotations

import pytest

from collector.decision_engine import RationaleCode, RuleId, validate_offer
from collector.llm.base import ToolCall
from collector.money import Money
from collector.negotiation import NegotiationState
from collector.offers import Cadence, ConsumerProposal
from collector.policy import PolicyConfig
from collector.tools import ToolContext, execute

# The two codes that tell a consumer their *money* is too small. Saying either
# one about a figure that clears the floor is the defect under test.
TOO_SMALL_CODES = frozenset({RationaleCode.BELOW_SETTLEMENT_FLOOR, RationaleCode.BELOW_MIN_PAYMENT})

# A count the consumer never said. The schema makes payment_count required, so
# a bare per-payment figure forces the model to invent one of these; the
# rationale must not turn on which.
INVENTED_COUNTS = (1, 2, 3, 4)


@pytest.fixture
def policy() -> PolicyConfig:
    return PolicyConfig.default()


@pytest.fixture
def fresh(policy: PolicyConfig) -> NegotiationState:
    return NegotiationState.opening(policy)


def _relay(
    policy: PolicyConfig, *, amount_each: str, payment_count: int, cadence: str = "monthly"
) -> tuple[RationaleCode, Money]:
    """Relay a per-payment figure the way the prompt instructs, and read back
    the rationale plus the total the engine actually ruled on."""
    result = execute(
        ToolCall(
            name="validate_consumer_offer",
            arguments={
                "amount_each": amount_each,
                "payment_count": payment_count,
                "cadence": cadence,
            },
        ),
        ToolContext.opening(policy),
    )
    assert result.ok, result.payload
    assert result.verdict is not None
    assert result.proposal is not None
    return result.verdict.rationale_code, result.proposal.total


# --------------------------------------------------------------------------
# "I can only do three hundred dollars" — a figure that clears the floor
# --------------------------------------------------------------------------


class TestFigureAboveTheMinimumIsNeverCalledTooSmall:
    """$300 is above the $250 minimum payment. Nothing the model invents around
    it may turn it into a figure the consumer is told is too small."""

    @pytest.mark.parametrize("payment_count", INVENTED_COUNTS)
    def test_three_hundred_each_is_not_refused_as_below_a_minimum(
        self, policy: PolicyConfig, payment_count: int
    ) -> None:
        assert Money("300") >= policy.min_payment  # premise of the test
        rationale, _ = _relay(policy, amount_each="300", payment_count=payment_count)

        assert rationale not in TOO_SMALL_CODES, (
            f"relaying $300 each with an invented payment_count={payment_count} told the "
            f"consumer {rationale.value}; $300 clears the ${policy.min_payment.amount} "
            "minimum payment, so no 'your amount is too small' rationale is true of it"
        )

    def test_three_hundred_each_gives_the_same_answer_whatever_count_is_invented(
        self, policy: PolicyConfig
    ) -> None:
        """The consumer said one thing. The explanation they hear must not be a
        function of a number they never said."""
        by_count = {
            n: _relay(policy, amount_each="300", payment_count=n)[0] for n in INVENTED_COUNTS
        }

        assert len(set(by_count.values())) == 1, (
            "the same utterance produced different rationales depending on the count the "
            f"model invented: { ({n: r.value for n, r in by_count.items()}) }"
        )

    def test_a_bare_three_hundred_is_not_read_as_settling_the_whole_debt(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """Engine level. A per-payment figure arriving as ``total`` with
        ``payment_count=1`` is classified as a settlement offer of the entire
        balance and refused on the settlement floor — the live-call sentence."""
        proposal = ConsumerProposal(total=Money("300"), payment_count=1, cadence=Cadence.IMMEDIATE)
        verdict = validate_offer(proposal, fresh, policy)

        assert verdict.rationale_code not in TOO_SMALL_CODES, (
            "a $300 capacity statement read as a $300 lump-sum settlement is refused as "
            f"{verdict.rationale_code.value}, which is not true of a $300 payment"
        )


# --------------------------------------------------------------------------
# "A hundred and fifty dollars" — a figure that genuinely is too small
# --------------------------------------------------------------------------


class TestFigureBelowTheMinimumIsExplainedAsSuch:
    """$150 is below the $250 minimum payment, so BELOW_MIN_PAYMENT is the whole
    truth about it — for every count, since no multiple of a sub-floor payment
    ever clears the floor."""

    @pytest.mark.parametrize("payment_count", (*INVENTED_COUNTS, 5, 6, 7))
    def test_one_fifty_each_is_explained_as_below_the_minimum_payment(
        self, policy: PolicyConfig, payment_count: int
    ) -> None:
        assert Money("150") < policy.min_payment  # premise of the test
        rationale, total = _relay(policy, amount_each="150", payment_count=payment_count)

        assert rationale is RationaleCode.BELOW_MIN_PAYMENT, (
            f"relaying $150 each with an invented payment_count={payment_count} was ruled "
            f"on a fabricated total of {total} and answered {rationale.value}; the "
            f"instalment is below the ${policy.min_payment.amount} floor and that is the "
            "only thing true of it"
        )

    def test_one_fifty_each_is_never_answered_with_over_collection(
        self, policy: PolicyConfig
    ) -> None:
        """The live call's first wrong sentence. ABOVE_BALANCE_OWED fires only
        because ``amount_each * payment_count`` fabricated $1,050 out of a
        figure the consumer never multiplied."""
        offending = {
            n: total
            for n in (*INVENTED_COUNTS, 5, 6, 7)
            for rationale, total in [_relay(policy, amount_each="150", payment_count=n)]
            if rationale is RationaleCode.ABOVE_BALANCE_OWED
        }

        fabricated = sorted(str(t) for t in offending.values())
        assert not offending, (
            "telling a consumer their $150 would collect more than the balance is an "
            "internal-consistency guard leaking into speech; it fired for "
            f"payment_count={sorted(offending)} on fabricated totals {fabricated}"
        )


# --------------------------------------------------------------------------
# The guard itself
# --------------------------------------------------------------------------


class TestOverCollectionIsNotAConsumerFacingRationale:
    """NO_OVER_COLLECTION exists to stop the engine authorizing more than is
    owed. A consumer proposal can only trip it via arithmetic the consumer did
    not do, so it must never be the code the model is handed to phrase."""

    @pytest.mark.parametrize("amount_each", ("150", "300", "400"))
    def test_no_relayed_per_payment_figure_ever_yields_over_collection(
        self, policy: PolicyConfig, amount_each: str
    ) -> None:
        for payment_count in (1, 2, 3, 4, 5, 6, 7):
            rationale, total = _relay(policy, amount_each=amount_each, payment_count=payment_count)
            assert rationale is not RationaleCode.ABOVE_BALANCE_OWED, (
                f"${amount_each} each x {payment_count} was multiplied out to {total}, "
                f"above the {policy.original_balance} balance, and the consumer is told so"
            )

    def test_a_schedule_that_overshoots_is_capped_not_refused(self, policy: PolicyConfig) -> None:
        """Four payments of $300 is $1,200 only if the last one is not trimmed.
        A consumer proposing to overpay is proposing a shorter plan, not an
        illegal one: the engine must never rule on more than is owed."""
        _, total = _relay(policy, amount_each="300", payment_count=4)

        assert total <= policy.original_balance, (
            f"the engine ruled on {total}, more than the {policy.original_balance} owed; "
            "amount_each * payment_count is the model's arithmetic, not the consumer's"
        )


# --------------------------------------------------------------------------
# The capacity path, which the schema does not actually leave open
# --------------------------------------------------------------------------


class TestCapacityRelayIsNotAnOffer:
    def test_signaled_capacity_alone_is_not_an_agreement_to_pay_in_full(
        self, policy: PolicyConfig
    ) -> None:
        """``payment_count`` is required, so a model relaying "I can only do
        three hundred" as pure capacity must still pass a count. With no total
        the balance is assumed, and count=1 makes the consumer's refusal into an
        accepted promise to pay $1,000 today."""
        result = execute(
            ToolCall(
                name="validate_consumer_offer",
                arguments={
                    "signaled_capacity": "300",
                    "payment_count": 1,
                    "cadence": "immediate",
                },
            ),
            ToolContext.opening(policy),
        )
        assert result.ok, result.payload
        assert result.verdict is not None

        assert result.verdict.outcome != "accept", (
            "a consumer who said they can only manage $300 had a full-balance "
            f"{result.proposal.total if result.proposal else '?'} payment accepted on "
            "their behalf and placed on the table to confirm"
        )

    def test_the_min_payment_rule_is_evaluated_before_the_total_floor(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """Ordering is what decides which sentence is spoken. A sub-floor
        instalment is a fact about the figure the consumer named; the total
        floor is a fact about a total the model assembled, so the per-payment
        rule is the more fundamental of the two and must be answered first."""
        proposal = ConsumerProposal(total=Money("300"), payment_count=2, cadence=Cadence.MONTHLY)
        order = [c.rule_id for c in validate_offer(proposal, fresh, policy).conditions]

        assert order.index(RuleId.MIN_PAYMENT) < order.index(RuleId.TOTAL_FLOOR), (
            "TOTAL_FLOOR is evaluated first, so _first_failure explains an assembled "
            f"total before it explains the consumer's own instalment: {[r.value for r in order]}"
        )
