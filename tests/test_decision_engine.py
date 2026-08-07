"""Decision engine tests — SPEC §4, plan tasks T1-T6.

The engine is a pure function of (proposal, state, policy). Every test here is a
table of inputs and expected verdicts: no mocks, no I/O, no clock.
"""

from decimal import Decimal

import pytest

from collector.money import Money
from collector.negotiation import NegotiationState
from collector.offers import Cadence, ConsumerProposal, Tier
from collector.policy import PolicyConfig

# --------------------------------------------------------------------------
# T1 - thinnest end-to-end path: a full payment is accepted
# --------------------------------------------------------------------------


@pytest.fixture
def policy() -> PolicyConfig:
    return PolicyConfig.default()


@pytest.fixture
def fresh(policy: PolicyConfig) -> NegotiationState:
    return NegotiationState.opening(policy)


class TestMoney:
    def test_rejects_float(self) -> None:
        """Floats are a bug in money code. SPEC §9."""
        with pytest.raises(TypeError):
            Money(1000.0)  # type: ignore[arg-type]

    def test_accepts_int_str_decimal(self) -> None:
        assert Money(1000) == Money("1000.00") == Money(Decimal("1000"))

    def test_is_frozen(self) -> None:
        m = Money(100)
        with pytest.raises(AttributeError):
            m.amount = Decimal("5")  # type: ignore[misc]

    def test_formats_as_currency(self) -> None:
        assert str(Money("1000")) == "$1,000.00"
        assert str(Money("250.5")) == "$250.50"

    def test_arithmetic_stays_exact(self) -> None:
        assert Money("0.1") + Money("0.2") == Money("0.3")
        total = sum((Money("333.33"), Money("333.33"), Money("333.34")), Money(0))
        assert total == Money("1000.00")


class TestFullPayment:
    def test_accepts_full_balance_in_one_payment(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import validate_offer

        proposal = ConsumerProposal(total=Money("1000"), payment_count=1, cadence=Cadence.IMMEDIATE)
        verdict = validate_offer(proposal, fresh, policy)

        assert verdict.outcome == "accept"
        assert verdict.tier is Tier.PAY_IN_FULL

    def test_records_every_condition_evaluated_not_just_failures(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """SPEC §4.1: the decision record must show evaluated conditions and a
        policy path, not just an outcome. This is the vendor test from the
        research report - it is what proves the engine decided, not the model."""
        from collector.decision_engine import validate_offer

        proposal = ConsumerProposal(total=Money("1000"), payment_count=1, cadence=Cadence.IMMEDIATE)
        verdict = validate_offer(proposal, fresh, policy)

        assert len(verdict.conditions) > 0
        assert all(c.passed for c in verdict.conditions)
        # Every condition is self-describing: rule, actual, limit.
        for c in verdict.conditions:
            assert c.rule_id and c.actual and c.limit

    def test_accepted_full_payment_needs_no_counter(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import validate_offer

        proposal = ConsumerProposal(total=Money("1000"), payment_count=1, cadence=Cadence.IMMEDIATE)
        assert validate_offer(proposal, fresh, policy).counter is None

    def test_verdict_is_frozen(self, policy: PolicyConfig, fresh: NegotiationState) -> None:
        from collector.decision_engine import validate_offer

        proposal = ConsumerProposal(total=Money("1000"), payment_count=1, cadence=Cadence.IMMEDIATE)
        verdict = validate_offer(proposal, fresh, policy)
        with pytest.raises(AttributeError):
            verdict.outcome = "reject"  # type: ignore[misc]


class TestPolicyConstants:
    def test_derives_floors_from_the_brief(self, policy: PolicyConfig) -> None:
        """SPEC §2.1. A1 resolved: 25% is of the ORIGINAL BALANCE, fixed at $250."""
        assert policy.original_balance == Money("1000.00")
        assert policy.min_payment == Money("250.00")
        assert policy.settlement_floor == Money("800.00")

    def test_min_payment_is_derived_not_hardcoded(self) -> None:
        doubled = PolicyConfig.default().replace(original_balance=Money("2000"))
        assert doubled.min_payment == Money("500.00")
        assert doubled.settlement_floor == Money("1600.00")

    def test_max_installments_falls_out_of_the_floor(self, policy: PolicyConfig) -> None:
        """§2.3 rule 1: $1000 / $250 = 4. No plan can ever exceed 4 payments."""
        assert policy.max_installments == 4


# --------------------------------------------------------------------------
# T2 - rejection plus the counter machinery.
# The brief's core requirement: "validated AND COUNTERED by logic outside the
# agent, mid-call."
# --------------------------------------------------------------------------


def _propose(
    total: str,
    count: int,
    cadence: Cadence = Cadence.MONTHLY,
    capacity: str | None = None,
) -> ConsumerProposal:
    return ConsumerProposal(
        total=Money(total),
        payment_count=count,
        cadence=cadence,
        signaled_capacity=Money(capacity) if capacity else None,
    )


def _assert_offer_is_legal(offer: object, policy: PolicyConfig) -> None:
    """Every engine-authored counter must itself satisfy SPEC §4.3."""
    from collector.offers import Offer

    assert isinstance(offer, Offer)
    assert offer.smallest_payment >= policy.min_payment
    assert offer.total >= policy.settlement_floor
    assert offer.payment_count <= policy.max_payments_for(offer.tier)
    assert offer.duration_days <= policy.max_plan_days
    assert offer.cadence in policy.allowed_cadences
    assert sum((i.amount for i in offer.installments), Money.zero()) == offer.total
    if offer.tier is not Tier.SETTLEMENT:
        assert offer.total == policy.original_balance


class TestRejectionAndCounter:
    @pytest.mark.parametrize("lowball", ["50", "200", "500", "799.99"])
    def test_lowballs_are_rejected_with_a_counter(
        self, lowball: str, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose(lowball, 1), fresh, policy)
        assert verdict.outcome == "reject"
        assert verdict.counter is not None, "a rejection must always carry a counter"
        _assert_offer_is_legal(verdict.counter, policy)

    def test_failing_condition_names_the_rule_with_actual_and_limit(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import RuleId, validate_offer

        verdict = validate_offer(_propose("50", 1), fresh, policy)
        failed = [c for c in verdict.conditions if not c.passed]
        assert any(c.rule_id is RuleId.TOTAL_FLOOR for c in failed)
        floor = next(c for c in failed if c.rule_id is RuleId.TOTAL_FLOOR)
        assert floor.actual == "$50.00"
        assert floor.limit == ">= $800.00"

    def test_rationale_code_is_a_closed_enum_never_free_text(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import RationaleCode, validate_offer

        verdict = validate_offer(_propose("50", 1), fresh, policy)
        assert isinstance(verdict.rationale_code, RationaleCode)

    def test_settlement_boundary(self, policy: PolicyConfig, fresh: NegotiationState) -> None:
        from collector.decision_engine import validate_offer

        assert validate_offer(_propose("799.99", 1), fresh, policy).outcome == "reject"
        assert validate_offer(_propose("800.00", 1), fresh, policy).outcome == "accept"


# --------------------------------------------------------------------------
# T3 - Tier 2: downpayment + one more, downpayment maximized
# --------------------------------------------------------------------------


class TestDownpaymentPlusOne:
    def test_counter_maximizes_the_downpayment_when_no_capacity_signalled(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """'Highest-amount downpayment' - cap is balance minus the floor."""
        from collector.decision_engine import build_counter

        offer = build_counter(fresh, policy, tier=Tier.DOWNPAYMENT_PLUS_ONE, capacity=None)
        assert offer.installments[0].amount == Money("750.00")
        assert offer.installments[1].amount == Money("250.00")
        assert offer.total == Money("1000.00")

    @pytest.mark.parametrize(
        ("capacity", "expected_down"),
        [("900", "750"), ("750", "750"), ("400", "400"), ("250", "250")],
    )
    def test_downpayment_is_clamped_to_capacity(
        self, capacity: str, expected_down: str, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """downpayment == clamp(capacity, $250, $750)."""
        from collector.decision_engine import build_counter

        offer = build_counter(
            fresh, policy, tier=Tier.DOWNPAYMENT_PLUS_ONE, capacity=Money(capacity)
        )
        assert offer.installments[0].amount == Money(expected_down)
        _assert_offer_is_legal(offer, policy)

    def test_capacity_below_floor_falls_through_never_emits_subfloor(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """Capacity $100 cannot support T2. The engine must fall through to the
        most affordable legal structure, never invent a $100 installment."""
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("100", 1, capacity="100"), fresh, policy)
        assert verdict.counter is not None
        assert verdict.counter.smallest_payment >= policy.min_payment
        _assert_offer_is_legal(verdict.counter, policy)


# --------------------------------------------------------------------------
# T4 - Tier 3: settlement, up to 20% off, max 3 payments
# --------------------------------------------------------------------------


class TestSettlement:
    def test_accepts_within_discount_window(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("900", 2), fresh, policy)
        assert verdict.outcome == "accept"
        assert verdict.tier is Tier.SETTLEMENT

    def test_settlement_of_800_over_3_respects_the_250_floor(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """§2.3 rule 3: the $250 floor binds harder than the discount.

        A1 makes the floor 25% of the ORIGINAL balance, so 200/300/300 is
        illegal even though $200 is 25% of the $800 settlement. Any split
        clearing $250 is fine - this asserts the rule, not one example of it.
        """
        from collector.decision_engine import build_counter

        offer = build_counter(
            fresh, policy, tier=Tier.SETTLEMENT, capacity=Money("300"), payments=3
        )
        amounts = [i.amount for i in offer.installments]
        assert len(amounts) == 3
        assert all(a >= Money("250") for a in amounts), "200/300/300 is illegal"
        assert sum(amounts, Money.zero()) == Money("800.00")

    def test_more_than_three_payments_is_not_a_settlement(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("850", 4), fresh, policy)
        assert verdict.outcome != "accept"
        assert verdict.counter is not None
        assert verdict.counter.payment_count <= 3 or verdict.counter.tier is not Tier.SETTLEMENT

    def test_installments_sum_exactly_no_decimal_drift(self, policy: PolicyConfig) -> None:
        from collector.decision_engine import build_counter
        from collector.negotiation import NegotiationState as NS

        for cap in ("250", "267", "300", "334", "500"):
            offer = build_counter(
                NS.opening(policy), policy, tier=Tier.SETTLEMENT, capacity=Money(cap)
            )
            assert sum((i.amount for i in offer.installments), Money.zero()) == offer.total


# --------------------------------------------------------------------------
# T5 - Tier 4: payment plan, no discount, and the impossible-schedule counter
# --------------------------------------------------------------------------


class TestPaymentPlan:
    def test_weekly_over_three_months_is_impossible_and_gets_countered(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """§2.3 rule 1. 13 weekly payments of ~$77 is below the $250 floor.
        The engine must counter with a legal structure, not accommodate it."""
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("1000", 13, Cadence.WEEKLY), fresh, policy)
        assert verdict.outcome == "reject"
        assert verdict.counter is not None
        assert verdict.counter.payment_count == 4
        assert verdict.counter.cadence is Cadence.WEEKLY, "honour their cadence preference"
        assert all(i.amount == Money("250") for i in verdict.counter.installments)
        assert verdict.counter.duration_days == 21
        _assert_offer_is_legal(verdict.counter, policy)

    def test_no_accepted_plan_ever_exceeds_four_installments(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import validate_offer

        for count in range(5, 15):
            verdict = validate_offer(_propose("1000", count), fresh, policy)
            assert verdict.outcome != "accept", f"{count} payments must never be accepted"

    def test_three_monthly_payments_is_accepted(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("1000", 3, Cadence.MONTHLY), fresh, policy)
        assert verdict.outcome == "accept"
        assert verdict.tier is Tier.PAYMENT_PLAN

    def test_plan_carries_no_discount(self, policy: PolicyConfig, fresh: NegotiationState) -> None:
        """A3: only settlements discount. A 3-payment plan at $900 is a
        settlement, not a plan."""
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("900", 3), fresh, policy)
        assert verdict.tier is Tier.SETTLEMENT
