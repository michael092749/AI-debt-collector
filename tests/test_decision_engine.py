"""Decision engine tests.

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
        """Floats are a bug in money code."""
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

    @pytest.mark.parametrize("value", ["NaN", "nan", "Infinity", "-Infinity", "inf"])
    def test_rejects_non_finite_values(self, value: str) -> None:
        """A quiet NaN previously quantized without trapping and crashed the
        engine two calls downstream (ADVERSARIAL_TESTING.md C8)."""
        with pytest.raises(ValueError, match="non-finite"):
            Money(value)


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
        """The decision record must show evaluated conditions and a
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


class TestOverCollection:
    """The total is `== ORIGINAL_BALANCE` unless the tier is
    settlement. Only the lower half of that equality was ever enforced.

    This is the worst shape a defect can take in this system. The model did not
    originate the figure — the engine did — so `you_may_say` published it as
    authorized, the numeric guard passed it correctly, and the agreement record
    was written. No guardrail above the engine can catch an illegal number the
    engine itself authorized.
    """

    def test_a_total_above_the_balance_is_refused(self, policy: PolicyConfig) -> None:
        """Reproduces a consumer who misheard the balance: "a thousand a month
        for three months" on a $1,000 debt was accepted at $3,000."""
        from collector.decision_engine import RationaleCode, validate_offer

        verdict = validate_offer(
            ConsumerProposal(total=Money("3000.00"), payment_count=3, cadence=Cadence.MONTHLY),
            NegotiationState.opening(policy),
            policy,
        )
        assert verdict.outcome != "accept"
        assert verdict.rationale_code == RationaleCode.ABOVE_BALANCE_OWED
        assert verdict.counter is not None
        assert verdict.counter.total <= policy.original_balance

    @pytest.mark.parametrize("total", ["1000.01", "1500.00", "3000.00", "1000000.00"])
    def test_no_amount_above_the_balance_is_ever_accepted(
        self, policy: PolicyConfig, total: str
    ) -> None:
        """There was no ceiling at all, so $1,000,000 was accepted too."""
        from collector.decision_engine import validate_offer

        verdict = validate_offer(
            ConsumerProposal(total=Money(total), payment_count=1, cadence=Cadence.IMMEDIATE),
            NegotiationState.opening(policy),
            policy,
        )
        assert verdict.outcome != "accept"

    def test_the_condition_trail_names_the_ceiling_either_way(self, policy: PolicyConfig) -> None:
        """A passing condition is part of the record too — that is the whole
        point of carrying every evaluated rule."""
        from collector.decision_engine import RuleId, validate_offer

        verdict = validate_offer(
            ConsumerProposal(
                total=policy.original_balance, payment_count=1, cadence=Cadence.IMMEDIATE
            ),
            NegotiationState.opening(policy),
            policy,
        )
        ceiling = next(c for c in verdict.conditions if c.rule_id == RuleId.NO_OVER_COLLECTION)
        assert ceiling.passed
        assert verdict.outcome == "accept", "paying exactly what is owed must still be accepted"


class TestPolicyConstants:
    def test_derives_floors_from_the_brief(self, policy: PolicyConfig) -> None:
        """A1: 25% is of the ORIGINAL BALANCE, not the agreed total — fixed at $250."""
        assert policy.original_balance == Money("1000.00")
        assert policy.min_payment == Money("250.00")
        assert policy.settlement_floor == Money("800.00")

    def test_min_payment_is_derived_not_hardcoded(self) -> None:
        doubled = PolicyConfig.default().replace(original_balance=Money("2000"))
        assert doubled.min_payment == Money("500.00")
        assert doubled.settlement_floor == Money("1600.00")

    def test_max_installments_falls_out_of_the_floor(self, policy: PolicyConfig) -> None:
        """$1000 / $250 = 4. No plan can ever exceed 4 payments."""
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
    """Every engine-authored counter must itself satisfy the total rule."""
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
        """$799.99 is below the floor at every tier; $800.00 is a legal
        settlement — legal, but only agreeable once the ladder has reached T3."""
        from collector.decision_engine import validate_offer

        assert validate_offer(_propose("799.99", 1), fresh, policy).outcome == "reject"
        settled = fresh.conceded_to(Tier.SETTLEMENT)
        assert validate_offer(_propose("800.00", 1), settled, policy).outcome == "accept"


# --------------------------------------------------------------------------
# T2b - the ladder anchors the negotiation
#
# The tiers are a preference order, not a menu. A consumer naming a low figure
# must not thereby collect every concession the ladder had left to give.
# --------------------------------------------------------------------------


def _at(tier: Tier, policy: PolicyConfig) -> NegotiationState:
    """A call that has already conceded its way down to ``tier``."""
    return NegotiationState.opening(policy).conceded_to(tier)


class TestLadderAnchoring:
    def test_a_legal_bottom_tier_proposal_is_countered_not_taken_on_turn_one(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """'$250 a month for four months' is policy-legal and the worst outcome
        on the list. Nothing has been refused, so it is countered at the top."""
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("1000", 4, capacity="250"), fresh, policy)

        assert verdict.outcome == "counter"
        assert verdict.counter is not None
        assert verdict.counter.tier is Tier.PAY_IN_FULL
        _assert_offer_is_legal(verdict.counter, policy)

    def test_the_ladder_failure_is_named_as_a_condition_and_a_rationale_code(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """A counter with six passing conditions and no failure would leave the
        audit trail unable to say why. LADDER is that condition."""
        from collector.decision_engine import RationaleCode, RuleId, validate_offer

        verdict = validate_offer(_propose("1000", 4, capacity="250"), fresh, policy)

        failed = [c for c in verdict.conditions if not c.passed]
        assert [c.rule_id for c in failed] == [RuleId.LADDER]
        assert verdict.rationale_code is RationaleCode.PREFERRED_TIER_AVAILABLE

    def test_a_substantive_failure_outranks_the_ladder_in_the_rationale(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """'$77 a week' fails the floor. Say that, not 'we would prefer more'."""
        from collector.decision_engine import RationaleCode, validate_offer

        verdict = validate_offer(_propose("1000", 13, Cadence.WEEKLY), fresh, policy)
        assert verdict.rationale_code is RationaleCode.BELOW_MIN_PAYMENT

    def test_a_low_capacity_signal_does_not_skip_the_ladder(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """Naming '$77 at a time' used to hand back a four-payment plan — every
        concession on the ladder, spent in one turn, for nothing."""
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("1000", 13, Cadence.WEEKLY, capacity="77"), fresh, policy)

        assert verdict.counter is not None
        assert verdict.counter.tier is Tier.PAY_IN_FULL

    def test_each_tier_costs_exactly_one_refusal(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import build_counter

        state = fresh
        walked = [build_counter(state, policy).tier]
        for _ in range(3):
            state = state.record_refusal().advance_ladder()
            walked.append(build_counter(state, policy).tier)

        assert walked == [
            Tier.PAY_IN_FULL,
            Tier.DOWNPAYMENT_PLUS_ONE,
            Tier.SETTLEMENT,
            Tier.PAYMENT_PLAN,
        ]

    @pytest.mark.parametrize(
        ("tier", "total", "count"),
        [
            (Tier.PAY_IN_FULL, "1000", 1),
            (Tier.DOWNPAYMENT_PLUS_ONE, "1000", 2),
            (Tier.SETTLEMENT, "850", 3),
            (Tier.PAYMENT_PLAN, "1000", 4),
        ],
    )
    def test_a_proposal_at_the_floor_is_accepted(
        self, tier: Tier, total: str, count: int, policy: PolicyConfig
    ) -> None:
        """The ladder gates what we *offer*; once it has arrived at a tier, a
        proposal there is agreeable."""
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose(total, count), _at(tier, policy), policy)
        assert verdict.outcome == "accept"
        assert verdict.tier is tier

    def test_a_proposal_better_than_the_floor_is_always_accepted(
        self, policy: PolicyConfig
    ) -> None:
        """A4 runs one way. Someone at the bottom of the ladder who offers to
        clear it outright is taken at their word, not countered."""
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("1000", 1), _at(Tier.PAYMENT_PLAN, policy), policy)
        assert verdict.outcome == "accept"
        assert verdict.tier is Tier.PAY_IN_FULL

    def test_terms_repeated_through_a_counter_are_taken(self, policy: PolicyConfig) -> None:
        """The ladder asks once for better. It does not hold out forever.

        A consumer walked down to settlement who still cannot manage $266.67 a
        month has nowhere left to go: the agent may not raise its own ask from
        $800 back to $1,000 (``_concede``), so without this their own legal
        four-payment plan — worth $200 more to us than the settlement — is
        unreachable by any path and the account is simply lost.

        This case now closes on the *first* ask rather than the second. The
        "better" the ladder was holding out for was $800 against the $1,000
        already offered, so the round it used to spend was spent volunteering a
        $200 discount to someone who had just offered the whole balance. The
        plan is still reachable — reached sooner, and for more money — so the
        loss this test was written to prevent is prevented harder. The
        ask-once-for-better behaviour it guards is unchanged wherever the
        counter is not for strictly less money; ``TestLadderAnchoring``'s
        turn-one cases below still assert it.
        """
        from collector.decision_engine import validate_offer

        state = _at(Tier.SETTLEMENT, policy)
        theirs = _propose("1000", 4, capacity="250")

        taken = validate_offer(theirs, state, policy)

        assert taken.outcome == "accept", "a full-balance offer is not countered down"
        assert taken.tier is Tier.PAYMENT_PLAN
        assert taken.counter is None

    def test_an_illegal_proposal_does_not_unlock_its_tier(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """ "Can I pay $77 a week?" is a plan-shaped proposal, refused on the
        payment floor. The "$250 a month" that follows is a different offer, not
        terms held to — it gets the ladder's counter like any other first ask."""
        from collector.decision_engine import validate_offer

        weekly = _propose("1000", 13, Cadence.WEEKLY)
        first = validate_offer(weekly, fresh, policy)
        state = fresh.record_round(weekly, first.counter, first.rationale_code)

        assert validate_offer(_propose("1000", 4), state, policy).outcome == "counter"

    def test_terms_sharpened_after_a_counter_do_not_count_as_holding_to_them(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """ "Settle at $900" answered with a counter must not become "$800,
        then" — that is the ladder talking us down $100, which is the opposite
        of what it is for."""
        from collector.decision_engine import validate_offer

        nine = _propose("900", 2)
        first = validate_offer(nine, fresh, policy)
        state = fresh.record_round(nine, first.counter, first.rationale_code)

        assert validate_offer(_propose("800", 2), state, policy).outcome == "counter"
        assert validate_offer(nine, state, policy).outcome == "accept", "$900 was held to"

    def test_a_repeat_does_not_excuse_an_illegal_proposal(self, policy: PolicyConfig) -> None:
        """Holding out is not a way through the policy floors. Only the ladder
        condition is satisfied by repetition; every other rule still binds."""
        from collector.decision_engine import validate_offer

        state = _at(Tier.SETTLEMENT, policy)
        theirs = _propose("500", 1)
        state = state.record_round(theirs, None, None)

        assert validate_offer(theirs, state, policy).outcome == "reject"

    def test_the_discount_cannot_be_had_by_asking_for_it(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """A3: the 20% is conceded to, never fallen into."""
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("800", 1), fresh, policy)
        assert verdict.outcome == "counter"
        assert verdict.counter is not None
        assert verdict.counter.total == policy.original_balance


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

    def test_a_low_first_offer_is_answered_at_the_top_of_the_ladder(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """ "$400" on turn one buys no concession at all.

        This used to counter $400 down and then $600 — a figure they had just
        said they could not manage — because capacity picked the tier. It now
        picks only the schedule, and on turn one the tier is still T1.
        """
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("400", 1), fresh, policy)

        assert verdict.counter is not None
        assert verdict.counter.tier is Tier.PAY_IN_FULL
        assert verdict.counter.total == Money("1000.00"), (
            "no refusal on record yet, so the discount is unearned: total stays at the balance"
        )
        _assert_offer_is_legal(verdict.counter, policy)

    def test_a_tier_that_cannot_hold_their_capacity_is_still_offered_at_its_floor(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """No two-payment split of $1,000 clears $400 twice, so T2 asks for more
        than they signalled. That is the tier being what it is, and it is theirs
        to refuse — which is what earns the step to a structure that fits."""
        from collector.decision_engine import build_counter

        offer = build_counter(
            fresh.conceded_to(Tier.DOWNPAYMENT_PLUS_ONE), policy, capacity=Money("400")
        )

        assert [i.amount for i in offer.installments] == [Money("400"), Money("600")]
        _assert_offer_is_legal(offer, policy)


# --------------------------------------------------------------------------
# T4 - Tier 3: settlement, up to 20% off, max 3 payments
# --------------------------------------------------------------------------


class TestSettlement:
    def test_accepts_within_discount_window(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import validate_offer

        verdict = validate_offer(_propose("900", 2), fresh.conceded_to(Tier.SETTLEMENT), policy)
        assert verdict.outcome == "accept"
        assert verdict.tier is Tier.SETTLEMENT

    def test_settlement_of_800_over_3_respects_the_250_floor(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """The $250 floor binds harder than the discount.

        A1 makes the floor 25% of the ORIGINAL balance, so 200/300/300 is
        illegal even though $200 is 25% of the $800 settlement. Any split
        clearing $250 is fine - this asserts the rule, not one example of it.
        """
        from collector.decision_engine import build_counter

        # At the floor specifically, which is where the $250 rule bites: an
        # even split of the $900 opening settlement clears it without trying.
        # A settlement already put up and not taken is what lowers the tier to
        # its floor (``_tier_total``).
        opening = build_counter(fresh, policy, tier=Tier.SETTLEMENT)
        at_the_floor = fresh.record_round(None, opening, "PREFERRED_TIER_AVAILABLE")

        offer = build_counter(
            at_the_floor, policy, tier=Tier.SETTLEMENT, capacity=Money("300"), payments=3
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

    def test_a_capacity_read_off_a_proposal_never_raises_the_one_they_stated(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """They said $300. Then they asked to settle at $700 in one payment.

        A lump sum they are asking us to *accept* is not a statement about what
        they can produce, and reading $700 back as capacity built the counter
        against a figure they never offered: $400/$400 to someone who had twice
        said $300. Inference is evidence of a ceiling, never of a floor.
        """
        from collector.decision_engine import validate_offer

        state = fresh.conceded_to(Tier.SETTLEMENT).with_capacity(Money("300"))
        verdict = validate_offer(_propose("700", 1, Cadence.IMMEDIATE), state, policy)

        assert verdict.outcome == "reject"
        assert verdict.counter is not None
        assert verdict.counter.installments[0].amount == Money("300.00")
        _assert_offer_is_legal(verdict.counter, policy)

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
        """13 weekly payments of ~$77 is below the $250 floor.
        The engine must counter with a legal structure, not accommodate it.

        Read at the bottom of the ladder, where a plan is what is on the table:
        earlier than that the counter is whichever tier the ladder stands on,
        which is the subject of ``TestLadderAnchoring``.
        """
        from collector.decision_engine import validate_offer

        state = fresh.conceded_to(Tier.PAYMENT_PLAN)
        verdict = validate_offer(_propose("1000", 13, Cadence.WEEKLY), state, policy)
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

        state = fresh.conceded_to(Tier.PAYMENT_PLAN)
        verdict = validate_offer(_propose("1000", 3, Cadence.MONTHLY), state, policy)
        assert verdict.outcome == "accept"
        assert verdict.tier is Tier.PAYMENT_PLAN

    def test_stated_downpayment_leads_the_plan_and_the_rest_splits_evenly(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """$400 on the table is $400 taken: the plan opens with what they
        offered and spreads the remaining $600 over two more payments. Money
        sooner beats a tidier $333.34/$333.33/$333.33, and it still totals the
        balance."""
        from collector.decision_engine import validate_offer

        state = fresh.conceded_to(Tier.PAYMENT_PLAN)
        verdict = validate_offer(_propose("400", 1), state, policy)

        assert verdict.counter is not None
        amounts = [i.amount for i in verdict.counter.installments]
        assert amounts == [Money("400"), Money("300"), Money("300")]
        assert verdict.counter.tier is Tier.PAYMENT_PLAN

    def test_plan_carries_no_discount(self, policy: PolicyConfig, fresh: NegotiationState) -> None:
        """A3: only settlements discount. A 3-payment plan at $900 is a
        settlement, not a plan — however far down the ladder it is proposed."""
        from collector.decision_engine import classify, validate_offer

        assert classify(_propose("900", 3), policy) is Tier.SETTLEMENT
        verdict = validate_offer(_propose("900", 3), fresh.conceded_to(Tier.PAYMENT_PLAN), policy)
        assert verdict.tier is Tier.SETTLEMENT


class TestALegalProposalIsNeverCounteredBelowItself:
    """A counter may ask for better terms. It may never ask for less money.

    The ``LADDER`` condition ranks tiers, not totals, and a settlement outranks
    a payment plan — so a consumer offering the *whole balance* in four legal
    instalments, once the ladder has reached a settlement, was answered with the
    $800 floor. The agent talked the consumer $200 down from their own offer,
    unprompted, on a proposal that broke no rule at all.

    Countering for a *better* tier is the point of the ladder (A7) and is kept.
    What cannot stand is a counter whose total is below the total already on
    offer: there is nothing left to negotiate for.
    """

    def test_full_balance_in_four_legal_payments_is_not_countered_with_a_discount(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import validate_offer

        state = fresh.conceded_to(Tier.SETTLEMENT)
        proposal = _propose("1000", 4, capacity="250")

        verdict = validate_offer(proposal, state, policy)

        assert verdict.counter is None or verdict.counter.total >= proposal.total, (
            f"countered a legal ${proposal.total} offer with "
            f"${verdict.counter.total if verdict.counter else None}"
        )

    def test_that_proposal_is_simply_accepted(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """It breaks no rule and pays more than anything we would have asked."""
        from collector.decision_engine import validate_offer

        state = fresh.conceded_to(Tier.SETTLEMENT)
        verdict = validate_offer(_propose("1000", 4, capacity="250"), state, policy)

        assert verdict.outcome == "accept"

    def test_a_sharpened_re_proposal_still_does_not_walk_us_down(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """The ``_held_to`` protection this must not undo: a consumer who is
        countered and comes back *lower* has sharpened their offer, not held to
        it, and the ladder may not be talked down to meet them."""
        from collector.decision_engine import validate_offer

        state = fresh.conceded_to(Tier.SETTLEMENT)
        verdict = validate_offer(_propose("700", 1), state, policy)

        assert verdict.outcome != "accept"


class TestSettlementOpensAboveItsFloor:
    """The most we may ever give up must cost more than one refusal.

    ``_tier_total`` returned ``settlement_floor`` for the settlement tier, so
    the first settlement a consumer ever heard *was* the maximum authorized
    discount. There was nothing left to concede inside the tier, and the number
    we least wanted to disclose was the first one we said.

    The floor is unchanged as a floor: ``max_settlement_discount`` still caps
    every offer. What changes is where the tier opens.
    """

    def test_the_first_settlement_is_above_the_floor(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import build_counter

        offer = build_counter(fresh.conceded_to(Tier.SETTLEMENT), policy)

        assert offer.tier is Tier.SETTLEMENT
        assert offer.total > policy.settlement_floor
        assert offer.total <= policy.original_balance

    def test_the_floor_is_still_reachable_as_a_later_concession(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """A consumer who refuses the opening settlement can still be met at
        the floor — otherwise this trades one dead rung for another."""
        from collector.decision_engine import build_counter

        state = fresh.conceded_to(Tier.SETTLEMENT)
        opening = build_counter(state, policy)
        refused = state.record_round(None, opening, "PREFERRED_TIER_AVAILABLE")

        assert build_counter(refused, policy).total == policy.settlement_floor

    def test_no_settlement_ever_discounts_past_the_policy_ceiling(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        from collector.decision_engine import build_counter

        state = fresh.conceded_to(Tier.SETTLEMENT)
        for _ in range(6):
            offer = build_counter(state, policy)
            assert offer.total >= policy.settlement_floor, (
                f"discounted past the ceiling: {offer.total}"
            )
            state = state.record_round(None, offer, "PREFERRED_TIER_AVAILABLE")

    def test_the_in_tier_step_reads_as_a_concession(
        self, policy: PolicyConfig, fresh: NegotiationState
    ) -> None:
        """Otherwise the tool layer reports no movement and the step is wasted."""
        from collector.decision_engine import build_counter
        from collector.tools import _is_concession

        state = fresh.conceded_to(Tier.SETTLEMENT)
        opening = build_counter(state, policy)
        refused = state.record_round(None, opening, "PREFERRED_TIER_AVAILABLE")

        assert _is_concession(build_counter(refused, policy), opening)
