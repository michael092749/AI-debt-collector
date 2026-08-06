"""Guardrail tests — SPEC §5, build step 4.

Three things are being proven here:

1. Prohibited persuasion is caught (the brief's automatic-fail condition).
2. No figure the engine did not authorize can reach TTS — spelled out or not.
3. Legitimate phrasing is *not* caught. False positives are the failure mode
   that gets a guardrail switched off in production, so they are tested as
   hard as the true positives.

Everything is offline and deterministic: no API keys, no clock, no network.
"""

from decimal import Decimal

import pytest

from collector.decision_engine import validate_offer
from collector.guardrails import (
    AI_DISCLOSURE_TEXT,
    MAX_REGENERATION_STRIKES,
    MINI_MIRANDA_TEXT,
    SAFE_FALLBACK_TEXT,
    AuthorizedFigures,
    DisclosureId,
    DisclosureRuleId,
    DisclosureState,
    DurationUnit,
    EscalationTrigger,
    FigureKind,
    GuardrailRing,
    GuardrailState,
    NumericRuleId,
    PreCallContext,
    ProhibitedRuleId,
    RingRuleId,
    Severity,
    authorized_for,
    check_inbound,
    check_numeric,
    check_outbound,
    check_pre_call,
    detect_escalation,
    escalation_closing,
    extract_figures,
    finalize_call,
    fires_mini_miranda,
    scan_prohibited,
)
from collector.money import Money
from collector.negotiation import NegotiationState
from collector.offers import Cadence, ConsumerProposal, Installment, Offer, Tier
from collector.policy import PolicyConfig


@pytest.fixture
def policy() -> PolicyConfig:
    return PolicyConfig.default()


@pytest.fixture
def settlement_offer() -> Offer:
    """$800 over three monthly payments, structured 250/250/300 per §2.3 rule 3."""
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
def ready(policy: PolicyConfig) -> GuardrailState:
    """Mid-call state: identity confirmed, both disclosures already fired."""
    return GuardrailState(
        authorized=authorized_for(policy),
        disclosures=DisclosureState(fired=frozenset(DisclosureId), agent_turns=1),
        identity_confirmed=True,
    )


def rule_ids(violations: tuple[object, ...]) -> set[str]:
    return {v.rule_id for v in violations}  # type: ignore[attr-defined]


# ==========================================================================
# Prohibited persuasion — SPEC §5.2. "Fails regardless of how well it converts."
# ==========================================================================


class TestProhibitedCategories:
    @pytest.mark.parametrize(
        ("utterance", "expected"),
        [
            ("If you don't pay, we will sue you.", ProhibitedRuleId.THREAT_LEGAL_ACTION),
            ("We'll take you to court over this.", ProhibitedRuleId.THREAT_LEGAL_ACTION),
            (
                "I'm turning this over to our legal department tomorrow.",
                ProhibitedRuleId.THREAT_LEGAL_ACTION,
            ),
            ("You could be arrested for this.", ProhibitedRuleId.THREAT_ARREST),
            ("There's a warrant coming if you ignore me.", ProhibitedRuleId.THREAT_ARREST),
            ("We will garnish your wages.", ProhibitedRuleId.THREAT_GARNISHMENT),
            ("We'll put a lien on your house.", ProhibitedRuleId.THREAT_GARNISHMENT),
            ("This will destroy your credit score.", ProhibitedRuleId.THREAT_CREDIT_REPORT),
            (
                "We report this to the credit bureaus every month.",
                ProhibitedRuleId.THREAT_CREDIT_REPORT,
            ),
            ("This deal is today only.", ProhibitedRuleId.FALSE_URGENCY),
            ("This is your final notice.", ProhibitedRuleId.FALSE_URGENCY),
            ("Last chance before things get worse.", ProhibitedRuleId.FALSE_URGENCY),
            ("If you hang up, that's it.", ProhibitedRuleId.FALSE_URGENCY),
            ("This offer expires at midnight.", ProhibitedRuleId.FALSE_URGENCY),
            (
                "This goes on your permanent record.",
                ProhibitedRuleId.INVENTED_CONSEQUENCE,
            ),
            ("We'll contact your employer next.", ProhibitedRuleId.INVENTED_CONSEQUENCE),
            (
                "You'll never be able to rent an apartment again.",
                ProhibitedRuleId.INVENTED_CONSEQUENCE,
            ),
            ("Someone will show up at your house.", ProhibitedRuleId.INVENTED_CONSEQUENCE),
            ("Don't be a deadbeat about it.", ProhibitedRuleId.ABUSIVE_LANGUAGE),
            (
                "Honestly, you should take out a loan to cover this.",
                ProhibitedRuleId.UNAUTHORIZED_ADVICE,
            ),
        ],
    )
    def test_every_prohibited_category_is_caught(
        self, utterance: str, expected: ProhibitedRuleId
    ) -> None:
        assert expected in rule_ids(scan_prohibited(utterance))

    def test_violation_is_structured_not_a_bare_boolean(self) -> None:
        """The audit log needs the rule, the span and the severity."""
        (violation,) = scan_prohibited("We will garnish your wages.")
        assert violation.rule_id == ProhibitedRuleId.THREAT_GARNISHMENT
        assert violation.severity is Severity.BLOCK
        assert violation.blocking
        assert violation.span == "garnish"
        assert "We will garnish your wages."[violation.start : violation.end] == "garnish"
        assert violation.detail

    def test_one_utterance_can_trip_several_rules(self) -> None:
        found = rule_ids(scan_prohibited("Final notice: we will sue you and garnish your wages."))
        assert {
            ProhibitedRuleId.FALSE_URGENCY,
            ProhibitedRuleId.THREAT_LEGAL_ACTION,
            ProhibitedRuleId.THREAT_GARNISHMENT,
        } <= found


class TestProhibitedFalsePositives:
    """False positives are a real failure mode: they train the agent out of
    saying the true, reassuring things a consumer most needs to hear."""

    @pytest.mark.parametrize(
        "utterance",
        [
            "No, we are not going to sue you.",
            "Nobody is taking you to court, and I can't give you legal advice.",
            "I'm not able to discuss credit reporting on this call.",
            "I can't tell you whether this affects your credit score.",
            "This account is 180 days past due.",
            "Your final payment would be $300.",
            "That would be your last payment on the plan.",
            "I'm not threatening you and there is no warrant involved here.",
            MINI_MIRANDA_TEXT,
            AI_DISCLOSURE_TEXT,
            SAFE_FALLBACK_TEXT,
            "I understand this is frustrating. What would work for you?",
        ],
    )
    def test_legitimate_phrasing_does_not_trip(self, utterance: str) -> None:
        assert scan_prohibited(utterance) == ()

    def test_negation_must_be_in_the_same_clause(self) -> None:
        """'not' in a previous clause must not launder a threat in the next one."""
        assert scan_prohibited("I'm not unreasonable. We will sue you.") != ()

    def test_consumer_speech_is_never_scanned_for_prohibited_persuasion(
        self, policy: PolicyConfig
    ) -> None:
        """'Are you going to sue me?' is the consumer's question, not a threat."""
        state = GuardrailState.opening(policy)
        result = check_inbound(state, "Are you going to sue me? Will you garnish my wages?")
        assert all(not e.violations for e in result.state.events)


# ==========================================================================
# Numeric authorization — SPEC §5.2. The model may never originate a number.
# ==========================================================================


class TestFigureExtraction:
    @pytest.mark.parametrize(
        ("utterance", "kind", "value"),
        [
            ("You'd pay $800 today.", FigureKind.MONEY, Decimal(800)),
            ("The balance is $1,000.00.", FigureKind.MONEY, Decimal(1000)),
            ("That's 800 dollars.", FigureKind.MONEY, Decimal(800)),
            ("eight hundred dollars", FigureKind.MONEY, Decimal(800)),
            ("one thousand dollars", FigureKind.MONEY, Decimal(1000)),
            ("two hundred fifty dollars", FigureKind.MONEY, Decimal(250)),
            ("eight hundred and fifty dollars", FigureKind.MONEY, Decimal(850)),
            ("a thousand dollars", FigureKind.MONEY, Decimal(1000)),
            ("I can do two fifty.", FigureKind.MONEY, Decimal(250)),
            ("Let's say seven fifty up front.", FigureKind.MONEY, Decimal(750)),
            ("We can take 20% off.", FigureKind.PERCENT, Decimal(20)),
            ("That's twenty percent off.", FigureKind.PERCENT, Decimal(20)),
            ("over 3 months", FigureKind.DURATION, Decimal(3)),
            ("180 days past due", FigureKind.DURATION, Decimal(180)),
            ("in three payments", FigureKind.PAYMENT_COUNT, Decimal(3)),
        ],
    )
    def test_extracts_value_and_kind(
        self, utterance: str, kind: FigureKind, value: Decimal
    ) -> None:
        figures = [f for f in extract_figures(utterance) if f.kind is kind]
        assert [f.value for f in figures] == [value]

    def test_span_covers_the_whole_figure(self) -> None:
        text = "You'd pay $800 today."
        (money,) = [f for f in extract_figures(text) if f.kind is FigureKind.MONEY]
        assert money.text == "$800"
        assert text[money.start : money.end] == "$800"

    def test_dates_are_extracted_and_normalized(self) -> None:
        figures = extract_figures("How about next Tuesday?")
        assert [f.token for f in figures if f.kind is FigureKind.DATE] == ["next tuesday"]

    def test_small_bare_numbers_are_not_treated_as_money(self) -> None:
        (figure,) = extract_figures("Let me ask you two things.")
        assert figure.kind is FigureKind.BARE

    def test_account_identifiers_are_not_treated_as_money(self) -> None:
        (figure,) = extract_figures("the account ending in 4417")
        assert figure.kind is FigureKind.BARE


class TestNumericAuthorization:
    def test_spelled_out_number_is_blocked_exactly_like_digits(self) -> None:
        """A model improvising says 'eight hundred dollars', not '$800'. Both must
        hit the same wall — this is the whole point of the check."""
        authorized = AuthorizedFigures(money=frozenset({Decimal(1000)}))
        digits = check_numeric("You'd pay $800.", authorized)
        words = check_numeric("You'd pay eight hundred dollars.", authorized)

        assert rule_ids(digits) == rule_ids(words) == {NumericRuleId.UNAUTHORIZED_AMOUNT}
        assert digits[0].severity is Severity.BLOCK
        assert words[0].span == "eight hundred dollars"

    def test_colloquial_hundreds_are_blocked(self) -> None:
        authorized = AuthorizedFigures(money=frozenset({Decimal(1000)}))
        (violation,) = check_numeric("I could do two fifty a month.", authorized)
        assert violation.rule_id == NumericRuleId.UNAUTHORIZED_AMOUNT
        assert violation.span == "two fifty"

    def test_engine_authored_offer_figures_pass(self, settlement_offer: Offer) -> None:
        authorized = AuthorizedFigures.empty().with_offer(settlement_offer)
        utterance = "That's $250 now, $250 in 30 days, and $300 in 60 days — $800 in total."
        assert check_numeric(utterance, authorized) == ()

    def test_one_wrong_digit_in_an_otherwise_authorized_line_still_blocks(
        self, settlement_offer: Offer
    ) -> None:
        authorized = AuthorizedFigures.empty().with_offer(settlement_offer)
        (violation,) = check_numeric("That's $250 now and $350 in 30 days.", authorized)
        assert violation.span == "$350"

    def test_policy_figures_are_authorized(self, policy: PolicyConfig) -> None:
        authorized = authorized_for(policy)
        assert check_numeric("The balance is $1,000.00.", authorized) == ()
        assert check_numeric("Payments can't be under $250.", authorized) == ()
        assert check_numeric("The most I can discount is 20%.", authorized) == ()

    def test_unauthorized_percent_blocks(self, policy: PolicyConfig) -> None:
        (violation,) = check_numeric("I can knock 50% off.", authorized_for(policy))
        assert violation.rule_id == NumericRuleId.UNAUTHORIZED_PERCENT

    def test_duration_units_are_compared_in_days(self, policy: PolicyConfig) -> None:
        """'three months' and '90 days' are the same authorization; rephrasing an
        engine figure in friendlier units is not the failure mode we're chasing."""
        authorized = authorized_for(policy)
        assert check_numeric("We can spread it over 3 months.", authorized) == ()
        assert check_numeric("We can spread it over 90 days.", authorized) == ()
        assert check_numeric("We can spread it over 12 months.", authorized) != ()

    def test_account_facts_must_be_authorized_explicitly(self, policy: PolicyConfig) -> None:
        bare = authorized_for(policy)
        (violation,) = check_numeric("This account is 180 days past due.", bare)
        assert violation.rule_id == NumericRuleId.UNAUTHORIZED_DURATION

        with_facts = authorized_for(policy, extra_durations=((180, DurationUnit.DAY),))
        assert check_numeric("This account is 180 days past due.", with_facts) == ()

    def test_today_is_allowed_but_invented_dates_are_not(self, policy: PolicyConfig) -> None:
        authorized = authorized_for(policy)
        assert check_numeric("Could you take care of it today?", authorized) == ()
        (violation,) = check_numeric("Let's say March 14th then.", authorized)
        assert violation.rule_id == NumericRuleId.UNAUTHORIZED_DATE

    def test_unattached_small_numbers_warn_rather_than_block(
        self, policy: PolicyConfig
    ) -> None:
        chatter = "There are 17 things I could say here."
        (violation,) = check_numeric(chatter, authorized_for(policy))
        assert violation.rule_id == NumericRuleId.UNVERIFIED_NUMBER
        assert violation.severity is Severity.WARN
        assert not violation.blocking

    def test_verdict_conditions_authorize_the_figures_they_state(
        self, policy: PolicyConfig
    ) -> None:
        """The engine's own condition trail is, by definition, engine-authored."""
        proposal = ConsumerProposal(total=Money("500"), payment_count=2, cadence=Cadence.MONTHLY)
        verdict = validate_offer(proposal, NegotiationState.opening(policy), policy)
        authorized = AuthorizedFigures.empty().with_verdict(verdict)

        assert Decimal(800) in authorized.money  # the settlement floor it failed
        assert check_numeric("The lowest total I can accept is $800.00.", authorized) == ()

    def test_money_comparison_is_exact_decimal(self) -> None:
        authorized = AuthorizedFigures(money=frozenset({Decimal("800.00")}))
        assert check_numeric("$800", authorized) == ()
        assert check_numeric("$800.01", authorized) != ()


# ==========================================================================
# Disclosures — SPEC §5.2 / §10
# ==========================================================================


class TestDisclosureDetection:
    def test_half_a_mini_miranda_is_not_a_mini_miranda(self) -> None:
        assert fires_mini_miranda("This is an attempt to collect a debt.") is None
        assert fires_mini_miranda(MINI_MIRANDA_TEXT) is not None

    def test_opening_state_owes_both_disclosures(self) -> None:
        state = DisclosureState.opening()
        assert not state.may_discuss_debt
        assert set(state.pending) == {DisclosureId.AI_DISCLOSURE, DisclosureId.MINI_MIRANDA}

    def test_firing_updates_the_state_the_negotiator_reads(self) -> None:
        state = DisclosureState.opening().observe_agent(AI_DISCLOSURE_TEXT)
        assert DisclosureId.AI_DISCLOSURE in state.fired
        assert not state.may_discuss_debt

        state = state.observe_agent(MINI_MIRANDA_TEXT)
        assert state.may_discuss_debt
        assert state.pending == ()


class TestDisclosureGating:
    def test_first_turn_must_disclose_the_ai(self) -> None:
        state = DisclosureState.opening()
        violations = state.check_agent_turn("Hi, is this Jordan?")
        assert rule_ids(violations) == {DisclosureRuleId.AI_DISCLOSURE_MISSING_AT_OPEN}
        assert state.check_agent_turn(f"{AI_DISCLOSURE_TEXT} Is this Jordan?") == ()

    def test_substantive_discussion_before_mini_miranda_is_blocked(self) -> None:
        state = DisclosureState(fired=frozenset({DisclosureId.AI_DISCLOSURE}), agent_turns=1)
        violations = state.check_agent_turn("You owe a balance we need to resolve.")
        assert rule_ids(violations) == {DisclosureRuleId.MINI_MIRANDA_NOT_FIRED}

    def test_mini_miranda_must_come_first_within_the_turn(self) -> None:
        """Saying the words after the pitch is a log entry, not a disclosure."""
        state = DisclosureState(fired=frozenset({DisclosureId.AI_DISCLOSURE}), agent_turns=1)
        late = state.check_agent_turn(f"Your balance is due. {MINI_MIRANDA_TEXT}")
        assert rule_ids(late) == {DisclosureRuleId.MINI_MIRANDA_OUT_OF_ORDER}

        early = state.check_agent_turn(f"{MINI_MIRANDA_TEXT} Your balance is due.")
        assert early == ()

    def test_identity_and_greeting_are_allowed_before_the_mini_miranda(self) -> None:
        state = DisclosureState(fired=frozenset({DisclosureId.AI_DISCLOSURE}), agent_turns=1)
        assert state.check_agent_turn("Am I speaking with Jordan Reyes?") == ()

    def test_ai_disclosure_on_request(self) -> None:
        state = DisclosureState(
            fired=frozenset({DisclosureId.AI_DISCLOSURE, DisclosureId.MINI_MIRANDA}),
            agent_turns=2,
        ).observe_consumer("Wait — am I talking to a real person?")

        assert state.ai_disclosure_requested
        assert DisclosureId.AI_DISCLOSURE in state.pending
        assert rule_ids(state.check_agent_turn("Let's get back to the account.")) == {
            DisclosureRuleId.AI_DISCLOSURE_REQUEST_IGNORED
        }
        assert state.check_agent_turn(AI_DISCLOSURE_TEXT) == ()

    def test_answering_the_request_clears_it(self) -> None:
        state = (
            DisclosureState(fired=frozenset(DisclosureId), agent_turns=2)
            .observe_consumer("Are you a robot?")
            .observe_agent(AI_DISCLOSURE_TEXT)
        )
        assert not state.ai_disclosure_requested


# ==========================================================================
# Escalation — SPEC §5.2, assumption A6
# ==========================================================================


class TestEscalationTriggers:
    @pytest.mark.parametrize(
        ("utterance", "expected"),
        [
            ("I dispute this debt, it isn't mine.", EscalationTrigger.DISPUTE),
            ("I never opened that account.", EscalationTrigger.DISPUTE),
            ("Send me proof before I pay anything.", EscalationTrigger.DISPUTE),
            ("This is identity theft.", EscalationTrigger.DISPUTE),
            ("I lost my job in March.", EscalationTrigger.HARDSHIP),
            ("I'm on disability and there's no income here.", EscalationTrigger.HARDSHIP),
            ("I'm filing for bankruptcy.", EscalationTrigger.HARDSHIP),
            ("I honestly want to kill myself over this.", EscalationTrigger.DISTRESS),
            ("I can't go on like this.", EscalationTrigger.DISTRESS),
            ("You'll have to talk to my attorney.", EscalationTrigger.ATTORNEY_REPRESENTATION),
            ("I'm represented by a lawyer.", EscalationTrigger.ATTORNEY_REPRESENTATION),
            ("Stop calling me.", EscalationTrigger.CEASE_AND_DESIST),
            ("Cease and desist, take me off your list.", EscalationTrigger.CEASE_AND_DESIST),
            ("Do not contact me again.", EscalationTrigger.CEASE_AND_DESIST),
        ],
    )
    def test_every_trigger_is_detected(
        self, utterance: str, expected: EscalationTrigger
    ) -> None:
        assert expected in {s.trigger for s in detect_escalation(utterance)}

    @pytest.mark.parametrize(
        "utterance",
        [
            "I don't owe that much — be reasonable.",
            "I can't afford $500 a month right now.",
            "Money's tight this month.",
            "Do I need a lawyer for something like this?",
            "Are you going to sue me?",
            "Can you call me back tomorrow?",
            "I can pay $400 now and the rest later.",
        ],
    )
    def test_ordinary_negotiation_is_not_an_escalation(self, utterance: str) -> None:
        """Haggling is capacity signal for the decision engine, not a trigger.
        If every hard bargain escalated, the agent would never close anything."""
        assert detect_escalation(utterance) == ()

    def test_most_serious_trigger_wins_when_several_match(
        self, policy: PolicyConfig
    ) -> None:
        result = check_inbound(
            GuardrailState.opening(policy),
            "I don't owe this, I lost my job, and honestly I want to end my life.",
        )
        assert result.escalation is not None
        assert result.escalation.trigger is EscalationTrigger.DISTRESS
        assert len(result.escalation.signals) >= 3


class TestEscalationBehavior:
    def test_record_captures_full_context(self, policy: PolicyConfig) -> None:
        utterance = "You need to talk to my attorney from here."
        result = check_inbound(GuardrailState.opening(policy), utterance)

        record = result.escalation
        assert record is not None
        assert record.trigger is EscalationTrigger.ATTORNEY_REPRESENTATION
        assert record.consumer_utterance == utterance
        assert record.turn_index == 0
        assert record.closing_line == result.closing_line
        assert result.state.escalated

    @pytest.mark.parametrize("trigger", list(EscalationTrigger))
    def test_closing_line_promises_a_human_and_passes_its_own_guard(
        self, trigger: EscalationTrigger, ready: GuardrailState
    ) -> None:
        """A6: stop negotiating, say a human will follow up, close politely."""
        closing = escalation_closing(trigger)
        assert "team" in closing
        result = check_outbound(ready, closing, authorized=AuthorizedFigures.empty())
        assert result.allowed, result.violations

    def test_negotiation_after_escalation_is_blocked(
        self, ready: GuardrailState, settlement_offer: Offer
    ) -> None:
        state = check_inbound(ready, "Stop calling me.").state
        result = check_outbound(
            state,
            "Before you go — could you settle for $800 today?",
            authorized=AuthorizedFigures.empty().with_offer(settlement_offer),
        )
        assert not result.allowed
        assert RingRuleId.NEGOTIATION_AFTER_ESCALATION in rule_ids(result.violations)

    def test_escalated_call_falls_back_to_the_closing_line(
        self, ready: GuardrailState
    ) -> None:
        state = check_inbound(ready, "I'm filing for bankruptcy.").state
        for _ in range(MAX_REGENERATION_STRIKES):
            result = check_outbound(state, "Let's settle this today.")
            state = result.state
        assert result.fallback_text == escalation_closing(EscalationTrigger.HARDSHIP)
        assert result.speak == result.fallback_text


# ==========================================================================
# Rings — SPEC §5.1 / §5.2 / §5.3
# ==========================================================================


class TestPreCallRing:
    def test_a_loaded_eligible_account_may_be_called(self) -> None:
        result = check_pre_call(PreCallContext(account_loaded=True, within_calling_window=True))
        assert result.allowed
        assert result.ring is GuardrailRing.PRE_CALL
        assert result.violations == ()

    @pytest.mark.parametrize(
        ("context", "expected"),
        [
            (
                PreCallContext(account_loaded=False, within_calling_window=True),
                RingRuleId.ACCOUNT_NOT_LOADED,
            ),
            (
                PreCallContext(account_loaded=True, within_calling_window=False),
                RingRuleId.CALL_WINDOW_CLOSED,
            ),
            (
                PreCallContext(account_loaded=True, within_calling_window=True, do_not_call=True),
                RingRuleId.DO_NOT_CALL_FLAG,
            ),
            (
                PreCallContext(
                    account_loaded=True, within_calling_window=True, attorney_on_file=True
                ),
                RingRuleId.ATTORNEY_ON_FILE,
            ),
            (
                PreCallContext(account_loaded=True, within_calling_window=True, cease_on_file=True),
                RingRuleId.CEASE_ON_FILE,
            ),
        ],
    )
    def test_ineligible_calls_are_blocked(
        self, context: PreCallContext, expected: RingRuleId
    ) -> None:
        result = check_pre_call(context)
        assert not result.allowed
        assert expected in rule_ids(result.violations)


class TestDuringCallRing:
    def test_no_balance_before_identity_is_confirmed(self, policy: PolicyConfig) -> None:
        state = GuardrailState(
            authorized=authorized_for(policy),
            disclosures=DisclosureState(fired=frozenset(DisclosureId), agent_turns=1),
        )
        result = check_outbound(state, "Your balance is $1,000.00.")
        assert RingRuleId.IDENTITY_NOT_CONFIRMED in rule_ids(result.violations)

        confirmed = check_outbound(state.with_identity_confirmed(), "Your balance is $1,000.00.")
        assert confirmed.allowed

    def test_a_clean_turn_is_allowed_and_advances_the_state(
        self, ready: GuardrailState, settlement_offer: Offer
    ) -> None:
        result = check_outbound(
            ready,
            "I can do $250 today, $250 in 30 days, and $300 in 60 days.",
            authorized=AuthorizedFigures.empty().with_offer(settlement_offer),
        )
        assert result.allowed
        assert result.speak == result.candidate
        assert result.strikes == 0
        assert result.state.agent_turns == ready.agent_turns + 1
        assert result.state.substantive_discussed

    def test_a_blocked_turn_names_the_violation_for_regeneration(
        self, ready: GuardrailState
    ) -> None:
        result = check_outbound(ready, "This is your final notice — pay $537 today.")
        assert not result.allowed
        assert result.speak is None
        assert {ProhibitedRuleId.FALSE_URGENCY, NumericRuleId.UNAUTHORIZED_AMOUNT} <= rule_ids(
            result.violations
        )
        assert "FALSE_URGENCY" in result.regeneration_note()
        # A blocked turn never advances the disclosure state.
        assert result.state.agent_turns == ready.agent_turns

    def test_two_strikes_fall_back_to_the_scripted_line(self, ready: GuardrailState) -> None:
        state = ready
        result = check_outbound(state, "Pay $537 now.")
        assert result.strikes == 1
        assert result.fallback_text is None

        result = check_outbound(result.state, "Fine — $499 then.")
        assert result.strikes == MAX_REGENERATION_STRIKES
        assert result.fallback_text == SAFE_FALLBACK_TEXT

    def test_the_safe_fallback_passes_the_guard_with_nothing_authorized(
        self, ready: GuardrailState
    ) -> None:
        """The escape hatch must itself be unable to trip anything."""
        result = check_outbound(ready, SAFE_FALLBACK_TEXT, authorized=AuthorizedFigures.empty())
        assert result.allowed, result.violations

    def test_a_clean_turn_resets_the_strike_count(self, ready: GuardrailState) -> None:
        blocked = check_outbound(ready, "Pay $537 now.")
        cleared = check_outbound(blocked.state, "What would work for you?")
        assert cleared.allowed
        assert cleared.strikes == 0

    def test_figures_are_reported_for_the_audit_log_even_when_allowed(
        self, ready: GuardrailState, settlement_offer: Offer
    ) -> None:
        result = check_outbound(
            ready,
            "That's $250 today.",
            authorized=AuthorizedFigures.empty().with_offer(settlement_offer),
        )
        assert [f.text for f in result.figures] == ["$250", "today"]


class TestPostCallRing:
    def test_a_compliant_call_summarizes_clean(
        self, policy: PolicyConfig, settlement_offer: Offer
    ) -> None:
        state = GuardrailState.opening(policy)
        opening = f"{AI_DISCLOSURE_TEXT} Am I speaking with Jordan Reyes?"
        state = check_outbound(state, opening).state
        state = check_inbound(state, "Yes, this is Jordan.").state
        state = state.with_identity_confirmed()
        state = check_outbound(state, MINI_MIRANDA_TEXT).state
        state = check_outbound(
            state,
            "The balance is $1,000.00. Could you take care of it today?",
            authorized=authorized_for(policy, offers=(settlement_offer,)),
        ).state

        summary = finalize_call(state)
        assert summary.compliant
        assert summary.ring is GuardrailRing.POST_CALL
        assert summary.blocked_turns == 0
        assert set(summary.disclosures_fired) == set(DisclosureId)
        assert summary.missing_disclosures == ()
        assert len(summary.events) == 4

    def test_blocked_turns_are_retained_but_are_not_a_call_failure(
        self, ready: GuardrailState
    ) -> None:
        """A blocked turn is the guard working; nothing reached the consumer."""
        state = check_outbound(ready, "This is your final notice.").state
        summary = finalize_call(state)
        assert summary.blocked_turns == 1
        assert ProhibitedRuleId.FALSE_URGENCY in rule_ids(summary.violations)
        assert summary.compliant

    def test_a_call_that_discussed_the_debt_without_disclosing_fails(
        self, policy: PolicyConfig
    ) -> None:
        state = GuardrailState(
            authorized=authorized_for(policy),
            disclosures=DisclosureState(
                fired=frozenset({DisclosureId.AI_DISCLOSURE}), agent_turns=1
            ),
            identity_confirmed=True,
            substantive_discussed=True,
        )
        summary = finalize_call(state)
        assert not summary.compliant
        assert RingRuleId.DISCLOSURE_NEVER_FIRED in rule_ids(summary.violations)
        assert DisclosureId.MINI_MIRANDA in summary.missing_disclosures

    def test_an_unpersisted_transcript_fails_the_call(self, ready: GuardrailState) -> None:
        summary = finalize_call(ready, transcript_persisted=False)
        assert not summary.compliant
        assert RingRuleId.TRANSCRIPT_NOT_PERSISTED in rule_ids(summary.violations)

    def test_escalation_is_carried_into_the_summary(self, ready: GuardrailState) -> None:
        state = check_inbound(ready, "I dispute this, it was identity theft.").state
        summary = finalize_call(state)
        assert summary.escalation is not None
        assert summary.escalation.trigger is EscalationTrigger.DISPUTE
