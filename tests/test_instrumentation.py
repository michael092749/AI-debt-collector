"""Instrumentation, argument validation and the streaming path.

Three things this proves, all offline:

1. **Every model call and every tool call reaches the log.** A voice turn has a
   hang-up budget of well under two seconds and can spend four tool rounds and
   two regeneration strikes inside it; a latency figure you cannot attribute is
   a budget you cannot defend.
2. **The tool boundary is a type boundary, not a policy one.** Malformed
   arguments come back as a readable ``ok: false``; an *unreasonable but
   well-formed* proposal goes straight through to the engine, because ruling on
   those is the engine's whole job.
3. **The streaming path enforces the same guardrails as the text path**, with
   one deliberate difference: a blocked sentence aborts to the scripted
   fallback instead of regenerating, because earlier sentences are already
   audio in the consumer's ear.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest

from collector.agent import NegotiationAgent, _split_sentences
from collector.audit.events import EventType, ModelCalled, ToolInvoked, event_from_json, event_json
from collector.audit.store import DB_PATH_ENV_VAR, DEFAULT_DB_PATH, AuditStore, default_db_path
from collector.guardrails.disclosures import AI_DISCLOSURE_TEXT
from collector.guardrails.rings import SAFE_FALLBACK_TEXT, PreCallContext
from collector.llm.base import (
    LLMClient,
    LLMResponse,
    LLMUsage,
    Message,
    StreamCompleted,
    StreamEvent,
    StreamingLLMClient,
    TextDelta,
    ToolCall,
    stream_response,
)
from collector.llm.mock_client import MockLLMClient
from collector.money import Money
from collector.offers import Cadence
from collector.policy import PolicyConfig
from collector.tools import (
    MAX_PROPOSED_PAYMENTS,
    TOOL_SCHEMAS,
    ArgumentError,
    ToolContext,
    execute,
)

POLICY = PolicyConfig.default()
SCRIPT = ["Yes, this is Dana.", "I could do $500 down.", "Yes, let's do that."]

# An opening line that clears every ring: AI disclosure fired, nothing
# substantive said before identity is confirmed, no unauthorized figure.
_GREETING = f"{AI_DISCLOSURE_TEXT} Am I speaking with the account holder?"


def _agent(store: AuditStore | None = None, llm: LLMClient | None = None) -> NegotiationAgent:
    return NegotiationAgent(llm=llm or MockLLMClient(), policy=POLICY, store=store)


def _run(script: list[str], store: AuditStore) -> NegotiationAgent:
    agent = _agent(store)
    agent.open_call(PreCallContext(account_loaded=True, within_calling_window=True))
    for said in script:
        if agent.ended:
            break
        agent.turn(said)
    return agent


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(name=name, arguments=dict(arguments))


# ==========================================================================
# Argument validation at the tool boundary
# ==========================================================================


class TestArgumentSchema:
    def test_the_schema_the_model_sees_is_generated_from_the_declaration(self) -> None:
        """One declaration, not three. A ``minimum`` in the schema and a bounds
        check in the coercion cannot disagree if only one of them is written."""
        schema = next(s for s in TOOL_SCHEMAS if s.name == "validate_consumer_offer")
        properties = schema.input_schema["properties"]

        assert properties["payment_count"] == {
            "type": "integer",
            "description": "How many payments to split it into. 1 for a lump sum.",
            "minimum": 1,
            "maximum": MAX_PROPOSED_PAYMENTS,
        }
        assert properties["cadence"]["enum"] == [c.value for c in Cadence]
        assert set(schema.input_schema["required"]) == {"payment_count", "cadence"}

    def test_arguments_reach_the_handler_already_typed(self) -> None:
        parsed = next(s for s in TOOL_SCHEMAS if s.name == "validate_consumer_offer").parse(
            {"total": "500.00", "payment_count": "2", "cadence": " Monthly "}
        )
        assert parsed == {
            "total": Money("500.00"),
            "payment_count": 2,
            "cadence": Cadence.MONTHLY,
        }

    def test_a_float_never_gets_past_the_boundary(self) -> None:
        """JSON has one number type and it decodes to float; Money refuses one
        (SPEC §9). The exact digits the model emitted survive the trip."""
        parsed = next(s for s in TOOL_SCHEMAS if s.name == "validate_consumer_offer").parse(
            {"total": 500.5, "payment_count": 1, "cadence": "immediate"}
        )
        assert parsed["total"] == Money("500.50")
        assert isinstance(parsed["total"].amount, Decimal)

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ({"payment_count": 2}, "cadence is required"),
            ({"cadence": "monthly"}, "payment_count is required"),
            ({"payment_count": 0, "cadence": "monthly"}, "at least 1"),
            (
                {"payment_count": MAX_PROPOSED_PAYMENTS + 1, "cadence": "monthly"},
                f"at most {MAX_PROPOSED_PAYMENTS}",
            ),
            ({"payment_count": 2, "cadence": "fortnightly"}, "must be one of"),
            ({"payment_count": 2, "cadence": "monthly", "total": "lots"}, "total"),
        ],
    )
    def test_a_malformed_argument_names_itself(self, arguments: dict, expected: str) -> None:
        schema = next(s for s in TOOL_SCHEMAS if s.name == "validate_consumer_offer")
        with pytest.raises(ArgumentError, match=expected):
            schema.parse(arguments)

    def test_a_bad_argument_is_a_payload_not_an_exception(self) -> None:
        """A typo must not end a call. The model reads the error and retries."""
        result = execute(
            _call("validate_consumer_offer", payment_count=2, cadence="fortnightly"),
            ToolContext.opening(POLICY),
        )
        assert not result.ok
        assert "cadence" in result.payload["error"]
        assert result.context.state == ToolContext.opening(POLICY).state

    def test_an_impossible_but_well_formed_proposal_reaches_the_engine(self) -> None:
        """'Weekly for a year' is 52 payments. The schema is a *type* boundary;
        ruling that unreasonable is the decision engine's job, not this layer's."""
        result = execute(
            _call("validate_consumer_offer", payment_count=52, cadence="weekly"),
            ToolContext.opening(POLICY),
        )
        assert result.ok
        assert result.verdict is not None
        assert result.verdict.outcome != "accept"
        assert result.payload["offer_on_the_table"] is not None


# ==========================================================================
# Audit instrumentation
# ==========================================================================


class TestToolInstrumentation:
    def test_every_tool_call_is_logged_not_just_the_ruling_ones(self, tmp_path: Path) -> None:
        """``DecisionRecorded`` is the compliance record and only two tools
        produce one. The other four move the negotiation and used to leave no
        trace of having been called at all."""
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _run(SCRIPT, store)
            invocations = store.tool_calls(agent.call_id)

        assert invocations
        assert {i.tool for i in invocations} <= {s.name for s in TOOL_SCHEMAS}
        # confirm_agreement produces no verdict of its own on the decisions
        # table, so this is the only place it appears.
        assert "confirm_agreement" in {i.tool for i in invocations}
        assert all(i.latency_ms >= 0 for i in invocations)

    def test_a_failed_tool_call_records_its_reason(self, tmp_path: Path) -> None:
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store)
            agent.open_call()
            # concede with nothing refused: the tool's own guard rail.
            agent._run_tool(_call("concede"))
            (invocation,) = [i for i in store.tool_calls(agent.call_id) if i.tool == "concede"]

        assert not invocation.ok
        assert "record_refusal" in (invocation.error or "")

    def test_the_arguments_are_recorded_as_the_model_sent_them(self, tmp_path: Path) -> None:
        """Verbatim, not coerced: the audit question is what the model asked
        for, and a normalized copy cannot answer it."""
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store)
            agent.open_call()
            agent._run_tool(_call("validate_consumer_offer", payment_count=2, cadence="MONTHLY"))
            (invocation,) = [
                i for i in store.tool_calls(agent.call_id) if i.tool == "validate_consumer_offer"
            ]

        assert invocation.arguments == {"payment_count": 2, "cadence": "MONTHLY"}


class TestModelInstrumentation:
    def test_usage_reaches_the_log_when_the_client_reports_it(self, tmp_path: Path) -> None:
        class MeteredClient:
            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                return LLMResponse(
                    text="Am I speaking with the account holder?",
                    usage=LLMUsage(
                        model="claude-sonnet-5",
                        latency_ms=412,
                        input_tokens=1200,
                        output_tokens=30,
                        cost_usd=Decimal("0.0000000"),
                    ),
                )

        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=MeteredClient())
            agent.open_call()
            call = store.model_calls(agent.call_id)[0]

        assert call.model == "claude-sonnet-5"
        assert call.latency_ms == 412
        assert (call.input_tokens, call.output_tokens) == (1200, 30)

    def test_a_failed_model_call_speaks_the_fallback_and_logs_the_reason(
        self, tmp_path: Path
    ) -> None:
        """A transient error used to kill the turn with no fallback and no
        logged reason. Silence is not the fix either: on a phone line silence
        *is* the dropped call, so the scripted line goes out and the reason is
        recorded against the turn."""

        class FailingClient:
            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                return LLMResponse(
                    error="APITimeoutError: timed out",
                    usage=LLMUsage(model="claude-sonnet-5", latency_ms=6000),
                )

        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=FailingClient())
            check, opening = agent.open_call()
            assert check.allowed
            assert opening == SAFE_FALLBACK_TEXT
            assert not agent.ended, "a failed call is recoverable from the next utterance"

            call = store.model_calls(agent.call_id)[0]

        assert call.error is not None
        assert "APITimeoutError" in call.error
        assert call.latency_ms == 6000

    def test_a_failed_model_call_mid_stream_still_says_something(self) -> None:
        class FailingStreamer:
            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                return LLMResponse(text=_GREETING)

            def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
                yield StreamCompleted(LLMResponse(error="APIConnectionError: reset"))

        agent = _agent(llm=FailingStreamer())
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))

        assert spoken == [SAFE_FALLBACK_TEXT]
        assert not agent.ended

    def test_every_round_trip_in_a_turn_is_accounted_for(self, tmp_path: Path) -> None:
        """A turn spends several calls — tool rounds plus the closing words.
        Counting only the last one under-reports the latency budget."""
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=_UsageMock())
            agent.open_call()
            agent.turn("Yes, this is Dana.")
            agent.turn("I could do $500 down.")
            calls = store.model_calls(agent.call_id)

        assert len(calls) > len(agent.turns), "tool rounds cost round trips too"


class _UsageMock(MockLLMClient):
    """The scripted client, with a usage record attached to every turn."""

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        response = super().respond(messages)
        usage = LLMUsage(model="mock", latency_ms=1, input_tokens=10, output_tokens=5)
        return LLMResponse(
            text=response.text, tool_calls=response.tool_calls, usage=usage, error=response.error
        )


class TestPricing:
    def test_a_priced_model_reports_a_decimal_cost(self) -> None:
        from collector.llm.anthropic_client import estimate_cost

        cost = estimate_cost("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0)
        assert cost == Decimal("3.00")
        assert isinstance(cost, Decimal), "SPEC §9: no floats, cost reports included"

    def test_an_unpriced_model_reports_no_cost_rather_than_a_guess(self) -> None:
        from collector.llm.anthropic_client import estimate_cost

        assert estimate_cost("some-future-model", input_tokens=1000, output_tokens=100) is None

    def test_cached_input_is_priced_below_fresh_input(self) -> None:
        from collector.llm.anthropic_client import estimate_cost

        fresh = estimate_cost("claude-sonnet-5", input_tokens=1000, output_tokens=0)
        cached = estimate_cost(
            "claude-sonnet-5", input_tokens=0, output_tokens=0, cache_read_tokens=1000
        )
        assert cached is not None and fresh is not None
        assert cached < fresh


# ==========================================================================
# The compliance score — SPEC §5.3
# ==========================================================================


class TestCompliancePersistence:
    def test_the_score_is_written_to_the_calls_row(self, tmp_path: Path) -> None:
        """It used to be computed by finalize_call and then dropped, which left
        the log unable to answer the one question it exists to answer."""
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _run(SCRIPT, store)
            report = agent.close()
            persisted = store.compliance(report.call_id)

        assert persisted is not None
        assert persisted.compliant is report.compliant
        assert persisted.blocked_turns == report.summary.blocked_turns
        assert persisted.turn_count == report.turns

    def test_a_call_that_was_never_closed_out_has_no_score(self, tmp_path: Path) -> None:
        """``None`` is not ``False``: a dropped process is not a failed call."""
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _run(["Yes, this is Dana."], store)
            assert store.compliance(agent.call_id) is None

    def test_a_log_written_before_the_columns_existed_still_opens(self, tmp_path: Path) -> None:
        """CREATE TABLE IF NOT EXISTS silently skips an existing table, columns
        and all, so an older log would fail every insert naming a new column."""
        import sqlite3

        path = tmp_path / "old.db"
        legacy = sqlite3.connect(path)
        legacy.execute(
            "CREATE TABLE calls (call_id TEXT PRIMARY KEY, account_ref TEXT, consumer_ref TEXT,"
            " channel TEXT, original_balance TEXT, started_at TEXT, ended_at TEXT,"
            " outcome TEXT, turn_count INTEGER)"
        )
        legacy.commit()
        legacy.close()

        with AuditStore(path, json_dir=tmp_path) as store:
            agent = _run(SCRIPT, store)
            report = agent.close()
            assert store.compliance(report.call_id) is not None


class TestModelSelection:
    def test_the_model_can_be_pinned_by_the_environment(self, monkeypatch) -> None:
        """Staged rollback without a redeploy: pin the previous model when a
        new one regresses, rather than editing a constant and shipping."""
        from collector.llm.anthropic_client import MODEL, MODEL_ENV_VAR, resolve_model

        assert resolve_model() == MODEL
        monkeypatch.setenv(MODEL_ENV_VAR, "claude-opus-5")
        assert resolve_model() == "claude-opus-5"
        assert resolve_model("claude-haiku-4-5") == "claude-haiku-4-5", "an explicit arg wins"


class TestStoreConfiguration:
    def test_the_db_path_can_be_set_by_the_environment(self, monkeypatch, tmp_path: Path) -> None:
        """A CWD-relative default breaks on the first deploy whose entrypoint
        does not run from the repo root."""
        assert default_db_path() == DEFAULT_DB_PATH
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "elsewhere.db"))
        assert default_db_path() == tmp_path / "elsewhere.db"

    def test_wal_is_on_so_a_commit_does_not_block_the_turn(self, tmp_path: Path) -> None:
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            journal = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
            synchronous = store._conn.execute("PRAGMA synchronous").fetchone()[0]

        assert journal.lower() == "wal"
        assert synchronous == 1  # NORMAL


class TestEventRoundTrip:
    def test_the_new_events_survive_the_json_round_trip(self) -> None:
        """The log keeps the full event verbatim; a type it cannot decode is a
        row that silently stops being readable."""
        events = (
            ToolInvoked(
                call_id="c1",
                turn_index=2,
                tool="concede",
                arguments={"preferred_cadence": "weekly"},
                ok=False,
                latency_ms=3,
                error="nothing to concede against",
            ),
            ModelCalled(
                call_id="c1",
                turn_index=2,
                model="claude-sonnet-5",
                latency_ms=380,
                input_tokens=900,
                output_tokens=42,
                cost_usd=Decimal("0.0033"),
                stop_reason="end_turn",
            ),
        )
        for event in events:
            assert event_from_json(event_json(event)) == event

    def test_the_new_events_appear_in_the_timeline(self, tmp_path: Path) -> None:
        """A type missing from the trace query vanishes from the timeline
        without failing anything."""
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _run(SCRIPT, store)
            kinds = {e.EVENT_TYPE for e in store.trace(agent.call_id)}

        assert EventType.TOOL_CALL in kinds


# ==========================================================================
# The streaming path — the voice shape
# ==========================================================================


class TestSentenceSplitting:
    @pytest.mark.parametrize(
        ("buffer", "sentences", "rest"),
        [
            ("Hello there. How are", ["Hello there."], "How are"),
            ("One. Two! Three? ", ["One.", "Two!", "Three?"], ""),
            ("", [], ""),
            ("no terminal punctuation yet", [], "no terminal punctuation yet"),
            # The figures this system speaks must not be split mid-number.
            ("That's $1,000.00 total. Okay? ", ["That's $1,000.00 total.", "Okay?"], ""),
            ('She said "yes." Then paid.', ['She said "yes."'], "Then paid."),
        ],
    )
    def test_sentences_flush_at_the_boundary(
        self, buffer: str, sentences: list[str], rest: str
    ) -> None:
        assert _split_sentences(buffer) == (sentences, rest)


class TestStreamAdapter:
    def test_a_non_streaming_client_degrades_to_one_delta(self) -> None:
        """The sentence-guard machinery stays testable with no key and no
        network: the scripted client emits its whole turn at once."""
        client = MockLLMClient()
        assert not isinstance(client, StreamingLLMClient)

        events = list(stream_response(client, (Message(role="consumer", content="Hello?"),)))
        assert isinstance(events[-1], StreamCompleted)
        assert all(isinstance(e, TextDelta) for e in events[:-1])

    def test_a_streaming_client_is_used_directly(self) -> None:
        class Streamer:
            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                raise AssertionError("respond must not be called when stream() exists")

            def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
                yield TextDelta("Hi. ")
                yield TextDelta("Speaking?")
                yield StreamCompleted(LLMResponse(text="Hi. Speaking?"))

        assert isinstance(Streamer(), StreamingLLMClient)
        events = list(stream_response(Streamer(), ()))
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hi. ", "Speaking?"]


class TestOpeningIsNotStreamed:
    def test_the_greeting_is_guarded_whole_because_the_rule_is_turn_scoped(self) -> None:
        """The AI-disclosure rule governs the first agent *turn*. A first
        sentence that is not the disclosure cannot be judged against it without
        knowing what the second sentence will say, so the opening is guarded as
        one unit — sentence-by-sentence guarding would block every greeting
        whose disclosure is not its opening clause.
        """
        agent = _agent()
        greeting = agent.messages  # capture identity before the call
        _, spoken = agent.open_call()

        assert spoken is not None
        assert spoken != SAFE_FALLBACK_TEXT, "guarded whole, this clears; per sentence it would not"
        assert AI_DISCLOSURE_TEXT.split(":")[0] in spoken
        assert greeting is agent.messages

    def test_the_mini_miranda_rule_does_compose_sentence_by_sentence(self) -> None:
        """It constrains *order* within a turn, and a per-sentence guard
        enforces order by construction — so mid-call turns stream fine."""
        agent = _agent()
        agent.open_call()
        sentences = list(agent.stream_turn("Yes, this is Dana."))

        assert len(sentences) > 1
        spoken = " ".join(sentences)
        assert "attempt to collect a debt" in spoken
        assert spoken != SAFE_FALLBACK_TEXT


class TestStreamingTurn:
    def test_it_yields_guarded_sentences_one_at_a_time(self) -> None:
        agent = _agent()
        agent.open_call()
        sentences = list(agent.stream_turn("Yes, this is Dana."))

        assert len(sentences) > 1, "a paragraph reaches TTS in pieces, not all at once"
        assert all(s == s.strip() and s for s in sentences)

    def test_it_reaches_the_same_outcome_as_the_text_path(self) -> None:
        """Same rings, same engine, same whitelist — only the guard's
        granularity moves. A divergence here is a compliance divergence."""
        spoken_text, spoken_stream = [], []

        text_agent = _agent()
        text_agent.open_call()
        stream_agent = _agent()
        stream_agent.open_call()

        for said in SCRIPT:
            if not text_agent.ended:
                spoken_text.append(text_agent.turn(said).spoken or "")
            if not stream_agent.ended:
                spoken_stream.append(" ".join(stream_agent.stream_turn(said)))

        assert spoken_stream == spoken_text
        assert stream_agent.close().outcome is text_agent.close().outcome

    def test_a_closing_tool_still_gets_its_closing_words(self) -> None:
        """confirm_agreement ends the call inside a tool. The consumer still has
        to hear the arrangement read back."""
        agent = _agent()
        agent.open_call()
        for said in SCRIPT[:-1]:
            list(agent.stream_turn(said))
        closing = list(agent.stream_turn(SCRIPT[-1]))

        assert agent.ended
        assert closing, "a turn that ends the call in silence is a dead line"

    def test_a_blocked_sentence_aborts_to_the_fallback_rather_than_regenerating(self) -> None:
        """The text path can retry because nothing was spoken. Here the earlier
        sentences are already audio, so a retry would contradict what the
        consumer just heard."""

        class ThreateningStreamer:
            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                # Used only for the opening line, which stream_turn does not drive.
                return LLMResponse(text=_GREETING)

            def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
                yield TextDelta("Am I speaking with the account holder? ")
                yield TextDelta("Pay today or we will garnish your wages. ")
                yield TextDelta("So what will it be? ")
                yield StreamCompleted(LLMResponse(text="..."))

        agent = _agent(llm=ThreateningStreamer())
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))

        assert "Am I speaking with the account holder?" in spoken
        assert not any("garnish" in s for s in spoken)
        assert spoken[-1] == SAFE_FALLBACK_TEXT
        assert not any("So what will it be?" in s for s in spoken), "the stream aborts, not skips"
        assert agent.turns[-1].blocked

    def test_an_unauthorized_figure_never_reaches_tts(self) -> None:
        """The numeric guard is the point of the whole architecture, and it has
        to hold sentence by sentence too."""

        class InventingStreamer:
            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                return LLMResponse(text=_GREETING)

            def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
                yield TextDelta("Am I speaking with the account holder? ")
                yield TextDelta("I can settle this for four hundred dollars. ")
                yield StreamCompleted(LLMResponse(text="..."))

        agent = _agent(llm=InventingStreamer())
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))

        assert not any("four hundred" in s for s in spoken)
        assert spoken[-1] == SAFE_FALLBACK_TEXT

    def test_an_escalation_stops_the_stream_before_any_generation(self) -> None:
        agent = _agent()
        agent.open_call()
        list(agent.stream_turn("Yes, this is Dana."))
        spoken = list(agent.stream_turn("I've lost my job and I have no income."))

        assert agent.ended
        assert agent.turns[-1].escalated
        assert spoken and "member of our team" in " ".join(spoken)

    def test_the_transcript_records_one_agent_turn_not_one_per_sentence(self) -> None:
        """The model wrote a paragraph; the next round trip should see it that
        way, whatever granularity TTS consumed it at."""
        agent = _agent()
        agent.open_call()
        before = sum(1 for m in agent.messages if m.role == "agent")
        list(agent.stream_turn("Yes, this is Dana."))
        after = sum(1 for m in agent.messages if m.role == "agent")

        assert after - before == 1

    def test_an_abandoned_stream_records_no_turn(self) -> None:
        """A caller that walks away mid-stream dropped the call; recording a
        completed turn for it would be a false record."""
        agent = _agent()
        agent.open_call()
        turns_before = len(agent.turns)
        stream = agent.stream_turn("Yes, this is Dana.")
        next(stream)
        stream.close()

        assert len(agent.turns) == turns_before
