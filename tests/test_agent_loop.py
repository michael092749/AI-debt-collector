"""The full loop, offline — SPEC §7.1.

Everything here runs with an empty ``.env``: the scripted client stands in for
Claude, and the Claude mapping is tested as a pure function so it needs no key
and no network.

The assertions are about the architecture, not the phrasing. A different model
would produce different sentences; it would still have to route every figure
through the engine, fire the disclosures in order, and stop dead on an
escalation trigger.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from collector.agent import NegotiationAgent
from collector.audit.store import AuditStore
from collector.decision_engine import RationaleCode
from collector.guardrails.disclosures import DisclosureId, confirms_identity
from collector.guardrails.numeric import extract_figures
from collector.guardrails.rings import EscalationTrigger, PreCallContext
from collector.llm.anthropic_client import _to_anthropic
from collector.llm.base import LLMResponse, Message, ToolCall, system_prompt
from collector.llm.mock_client import MockLLMClient, parse_proposal
from collector.money import Money
from collector.negotiation import CallOutcome
from collector.offers import Cadence, Tier
from collector.policy import PolicyConfig
from collector.tools import TOOL_NAMES, ToolContext, execute

POLICY = PolicyConfig.default()


def _agent(store: AuditStore | None = None) -> NegotiationAgent:
    return NegotiationAgent(llm=MockLLMClient(), policy=POLICY, store=store)


def _run(script: list[str], store: AuditStore | None = None) -> NegotiationAgent:
    agent = _agent(store)
    agent.open_call(PreCallContext(account_loaded=True, within_calling_window=True))
    for said in script:
        if agent.ended:
            break
        agent.turn(said)
    return agent


# --------------------------------------------------------------------------
# The tool whitelist
# --------------------------------------------------------------------------


class TestToolWhitelist:
    def test_unknown_tool_is_refused_not_executed(self) -> None:
        """SPEC §5.2: the agent may only take the actions in tools.py."""
        result = execute(ToolCall(name="wire_me_the_money"), ToolContext.opening(POLICY))
        assert not result.ok
        assert "not an available action" in result.payload["error"]

    def test_every_advertised_tool_is_dispatchable(self) -> None:
        context = ToolContext.opening(POLICY)
        for name in TOOL_NAMES:
            assert execute(ToolCall(name=name), context).payload.get("error", "") != (
                f"{name!r} is not an available action"
            )

    def test_malformed_arguments_return_a_payload_not_an_exception(self) -> None:
        """A typo must not end a call."""
        result = execute(
            ToolCall(name="validate_consumer_offer", arguments={"payment_count": 2,
                                                               "cadence": "fortnightly"}),
            ToolContext.opening(POLICY),
        )
        assert not result.ok
        assert "cadence" in result.payload["error"]

    def test_float_amounts_survive_the_boundary_exactly(self) -> None:
        """JSON has one number type and it decodes to float; Money rejects those."""
        result = execute(
            ToolCall(
                name="validate_consumer_offer",
                arguments={"total": 850.5, "payment_count": 3, "cadence": "monthly"},
            ),
            ToolContext.opening(POLICY),
        )
        assert result.ok
        assert result.proposal is not None
        assert result.proposal.total == Money(Decimal("850.50"))


class TestConcessionsAreEarned:
    def test_concede_refuses_without_a_refusal_on_record(self) -> None:
        context = ToolContext.opening(POLICY)
        context = execute(ToolCall(name="propose_offer"), context).context
        result = execute(ToolCall(name="concede"), context)
        assert not result.ok
        assert result.payload["may_concede"] is False

    def test_concede_never_hands_back_a_harder_offer(self) -> None:
        """The tier order puts settlement above payment plan, so a naive step
        down can raise the ask. Walking the ladder must never do that."""
        context = ToolContext.opening(POLICY)
        context = execute(ToolCall(name="propose_offer"), context).context
        standing = context.standing_offer
        assert standing is not None

        for _ in range(6):
            context = execute(ToolCall(name="record_refusal"), context).context
            result = execute(ToolCall(name="concede"), context)
            context = result.context
            offer = context.standing_offer
            assert offer is not None
            assert offer.total <= standing.total, "a concession raised the ask"
            standing = offer

    def test_confirm_agreement_revalidates_the_final_terms(self) -> None:
        context = ToolContext.opening(POLICY)
        context = execute(ToolCall(name="propose_offer"), context).context
        result = execute(ToolCall(name="confirm_agreement"), context)
        assert result.ok
        assert result.verdict is not None
        assert result.verdict.outcome == "accept"
        assert result.verdict.rationale_code is RationaleCode.ACCEPTED
        assert result.payload["conditions"], "the agreement carries its condition trail"

    def test_nothing_can_be_agreed_before_an_offer_exists(self) -> None:
        result = execute(ToolCall(name="confirm_agreement"), ToolContext.opening(POLICY))
        assert not result.ok


# --------------------------------------------------------------------------
# The turn loop
# --------------------------------------------------------------------------


class TestTurnLoop:
    def test_a_cooperative_call_closes_a_legal_agreement(self) -> None:
        agent = _run(["Yes, speaking.", "I can put down $400 today.", "Okay, that works."])
        report = agent.close(transcript_persisted=True)

        assert report.outcome is CallOutcome.AGREED
        offer = report.agreed_offer
        assert offer is not None
        assert offer.total >= POLICY.settlement_floor
        assert offer.smallest_payment >= POLICY.min_payment
        assert offer.payment_count <= POLICY.max_installments
        assert offer.duration_days <= POLICY.max_plan_days
        assert offer.cadence in POLICY.allowed_cadences

    def test_disclosures_fire_in_order_and_before_any_substance(self) -> None:
        agent = _run(["Yes, this is her.", "What do you want?"])
        spoken = [m.content for m in agent.messages if m.role == "agent"]

        assert "AI assistant" in spoken[0], "AI disclosure opens the call"
        mini_miranda_turn = next(
            i for i, text in enumerate(spoken) if "attempt to collect a debt" in text
        )
        substantive_turns = [
            i for i, text in enumerate(spoken) if "$" in text
        ]
        assert all(i > mini_miranda_turn for i in substantive_turns), (
            "no figure may be spoken before the Mini-Miranda"
        )

    def test_identity_gates_the_whole_conversation(self) -> None:
        agent = _run(["Who is this?", "I'm not telling you that."])
        assert not agent.guard.identity_confirmed
        assert not any("$" in m.content for m in agent.messages if m.role == "agent")

    def test_every_spoken_figure_was_authorized_by_the_engine(self) -> None:
        """The load-bearing invariant: the model never originates a number."""
        agent = _run(
            [
                "Yes, that's me.",
                "I can only do $60 a month.",
                "That's way too much.",
                "No.",
                "Fine, I'll do it.",
            ]
        )
        for message in agent.messages:
            if message.role != "agent":
                continue
            for figure in extract_figures(message.content):
                assert agent.authorized.permits(figure), (
                    f"unauthorized figure {figure.text!r} in {message.content!r}"
                )

    def test_escalation_halts_negotiation_immediately(self) -> None:
        agent = _run(["Yes.", "I lost my job, I can't pay anything."])
        report = agent.close(transcript_persisted=True)

        assert report.outcome is CallOutcome.ESCALATED
        assert report.agreed_offer is None
        assert agent.guard.escalation is not None
        assert agent.guard.escalation.trigger is EscalationTrigger.HARDSHIP
        assert agent.ended

    def test_cease_and_desist_ends_the_call_without_an_agreement(self) -> None:
        agent = _run(["Speaking.", "Stop calling me and don't contact me again."])
        assert agent.tools.state.outcome is CallOutcome.ESCALATED
        assert agent.close(transcript_persisted=True).agreed_offer is None

    def test_pre_call_block_means_no_call_is_placed(self) -> None:
        agent = _agent()
        check, opening = agent.open_call(
            PreCallContext(account_loaded=True, within_calling_window=True, cease_on_file=True)
        )
        assert not check.allowed
        assert opening is None
        assert agent.ended

    def test_a_blocked_turn_is_regenerated_not_spoken(self) -> None:
        """The guard holds the turn back; the loop names the violation and retries."""

        class ThreateningClient:
            def __init__(self) -> None:
                self.calls = 0

            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        text="Pay today or we will garnish your wages and sue you."
                    )
                return LLMResponse(text="Am I speaking with the account holder?")

        llm = ThreateningClient()
        agent = NegotiationAgent(llm=llm, policy=POLICY)
        _, opening = agent.open_call()

        assert opening is not None
        assert "garnish" not in opening
        assert not any("garnish" in m.content for m in agent.messages if m.role == "agent")
        assert any(not e.allowed for e in agent.guard.events)

    def test_the_round_cap_stops_the_agent_badgering(self) -> None:
        agent = _agent()
        agent.open_call()
        agent.turn("Yes, speaking.")
        for _ in range(12):
            if agent.ended:
                break
            agent.turn("No, I won't.")
        assert agent.tools.state.round_count <= POLICY.max_negotiation_rounds


class TestCallRecord:
    def test_the_agreement_record_carries_its_decision_trail(self, tmp_path: Path) -> None:
        with AuditStore(tmp_path / "collector.db", json_dir=tmp_path) as store:
            agent = _run(["Yes, speaking.", "I could do $500 down.", "Yes, let's do that."], store)
            report = agent.close()

            assert report.compliant, [v.detail for v in report.summary.violations]
            record = store.agreement(report.call_id)
            assert record is not None
            assert record.conditions, "the record names the rules that authorized it"
            assert all(c.passed for c in record.conditions)
            assert record.confirmation.confirmed
            assert record.total >= POLICY.settlement_floor
            assert (tmp_path / f"{record.agreement_id}.json").exists()

    def test_an_escalated_call_writes_no_agreement(self, tmp_path: Path) -> None:
        with AuditStore(tmp_path / "collector.db", json_dir=tmp_path) as store:
            agent = _run(["Yes.", "I dispute this, it isn't my account."], store)
            report = agent.close()
            assert store.agreement(report.call_id) is None
            assert store.escalations(report.call_id)

    def test_the_trace_holds_both_speakers_and_every_decision(self, tmp_path: Path) -> None:
        with AuditStore(tmp_path / "collector.db", json_dir=tmp_path) as store:
            agent = _run(["Yes, speaking.", "How about $200?", "No."], store)
            report = agent.close()

            turns = store.turns(report.call_id)
            assert {t.speaker for t in turns} == {"consumer", "agent"}
            decisions = store.decisions(report.call_id)
            assert decisions, "the consumer's proposal reached the engine"
            assert all(d.verdict.conditions for d in decisions)


# --------------------------------------------------------------------------
# The scripted client
# --------------------------------------------------------------------------


class TestMockClient:
    def test_it_is_deterministic(self) -> None:
        messages = (
            system_prompt(consumer_name="Dana", account_ref="A-1"),
            Message(role="agent", content="Am I speaking with the account holder?"),
            Message(role="consumer", content="Yes, this is Dana."),
        )
        first = MockLLMClient().respond(messages)
        assert first == MockLLMClient().respond(messages)

    @pytest.mark.parametrize(
        ("said", "total", "count", "cadence"),
        [
            ("I can give you $50.", "50", 1, "immediate"),
            ("I could do $300 a month for three months.", "900", 3, "monthly"),
            ("weekly for a year", None, 52, "weekly"),
            ("Let me split it into 4 payments.", None, 4, "monthly"),
        ],
    )
    def test_it_hears_proposals_in_ordinary_speech(
        self, said: str, total: str | None, count: int, cadence: str
    ) -> None:
        parsed = parse_proposal(said)
        assert parsed is not None
        assert parsed.payment_count == count
        assert parsed.cadence == cadence
        assert (str(parsed.total) if parsed.total is not None else None) == total

    def test_plain_conversation_is_not_a_proposal(self) -> None:
        assert parse_proposal("I'm having a hard time right now.") is None

    def test_a_blocked_turn_produces_figure_free_phrasing(self) -> None:
        response = MockLLMClient().respond(
            (
                system_prompt(consumer_name="Dana", account_ref="A-1"),
                Message(role="agent", content="hello"),
                Message(role="system", content="UNAUTHORIZED_AMOUNT: $9,999.00"),
            )
        )
        assert not extract_figures(response.text)


class TestIdentityGate:
    @pytest.mark.parametrize(
        "said", ["Yes.", "Speaking.", "That's me.", "yeah, go ahead"]
    )
    def test_affirmations_confirm(self, said: str) -> None:
        assert confirms_identity(said)

    @pytest.mark.parametrize(
        "said",
        [
            "Who's asking?",
            "No, wrong number.",
            "She doesn't live here.",
            "Yes, but no, that's not me.",
        ],
    )
    def test_anything_short_of_a_clear_yes_does_not(self, said: str) -> None:
        assert not confirms_identity(said)


# --------------------------------------------------------------------------
# The Claude mapping — pure, so no key and no network
# --------------------------------------------------------------------------


class TestAnthropicMapping:
    def test_the_opening_system_prompt_becomes_the_system_parameter(self) -> None:
        system, conversation = _to_anthropic(
            (
                system_prompt(consumer_name="Dana", account_ref="A-1"),
                Message(role="consumer", content="Hello?"),
            )
        )
        assert "collections representative" in system
        assert conversation == [{"role": "user", "content": "Hello?"}]

    def test_a_guardrail_note_rides_in_tagged_as_an_operator_note(self) -> None:
        """This model has no mid-conversation system role, and the note must not
        be mistakable for something the consumer said."""
        _, conversation = _to_anthropic(
            (
                system_prompt(consumer_name="Dana", account_ref="A-1"),
                Message(role="agent", content="..."),
                Message(role="system", content="PROHIBITED_THREAT"),
            )
        )
        last = conversation[-1]
        assert last["role"] == "user"
        assert "<compliance_note>" in last["content"][0]["text"]

    def test_a_tool_result_is_preceded_by_the_call_that_asked_for_it(self) -> None:
        call = ToolCall(name="propose_offer", arguments={}, call_id="toolu_abc")
        _, conversation = _to_anthropic(
            (
                system_prompt(consumer_name="Dana", account_ref="A-1"),
                Message(role="consumer", content="What can you do?"),
                Message(role="tool", content='{"ok": true}', tool_call=call),
            )
        )
        request, result = conversation[-2], conversation[-1]
        assert request["role"] == "assistant"
        assert request["content"][0]["type"] == "tool_use"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == request["content"][0]["id"]

    def test_a_missing_call_id_maps_the_same_way_every_run(self) -> None:
        message = Message(
            role="tool", content='{"ok": true}', tool_call=ToolCall(name="propose_offer")
        )
        preamble = (system_prompt(consumer_name="D", account_ref="A"),)
        first = _to_anthropic((*preamble, message))[1]
        second = _to_anthropic((*preamble, message))[1]
        assert first == second

    def test_the_tool_surface_the_model_sees_is_the_whitelist(self) -> None:
        from collector.llm.anthropic_client import tool_schemas_json

        assert {name for name in TOOL_NAMES} == {
            entry["name"] for entry in __import__("json").loads(tool_schemas_json())
        }


class TestOfferedTerms:
    """Spot-check that the loop's agreements land on real tiers, not near ones."""

    def test_a_capable_consumer_is_not_talked_down_the_ladder(self) -> None:
        agent = _run(["Yes, speaking.", "I can pay the whole thing today.", "Yes."])
        offer = agent.close(transcript_persisted=True).agreed_offer
        assert offer is not None
        assert offer.tier is Tier.PAY_IN_FULL
        assert offer.total == POLICY.original_balance
        assert offer.cadence is Cadence.IMMEDIATE

    def test_no_agreement_ever_dips_below_the_floors(self) -> None:
        scripts = [
            ["Yes.", "I'll give you $100.", "No.", "No.", "Okay fine."],
            ["Speaking.", "$25 a week is all I have.", "No way.", "Alright."],
            ["That's me.", "Can I pay $700 total?", "No.", "Yes, okay."],
        ]
        for script in scripts:
            offer = _run(script).close(transcript_persisted=True).agreed_offer
            if offer is None:
                continue
            assert offer.total >= POLICY.settlement_floor
            assert offer.smallest_payment >= POLICY.min_payment


def test_the_disclosure_ids_are_the_ones_the_summary_reports() -> None:
    agent = _run(["Yes, speaking.", "What's this about?"])
    summary = agent.close(transcript_persisted=True).summary
    assert DisclosureId.AI_DISCLOSURE in summary.disclosures_fired
    assert DisclosureId.MINI_MIRANDA in summary.disclosures_fired
