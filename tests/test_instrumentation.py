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

from collector.agent import MAX_TOOL_ROUNDS, NegotiationAgent, _split_sentences
from collector.audit.events import (
    EventType,
    GuardrailAction,
    GuardrailTripped,
    ModelCalled,
    Speaker,
    ToolInvoked,
    TurnRecorded,
    event_from_json,
    event_json,
)
from collector.audit.store import DB_PATH_ENV_VAR, DEFAULT_DB_PATH, AuditStore, default_db_path
from collector.guardrails.disclosures import (
    AI_DISCLOSURE_TEXT,
    MINI_MIRANDA_TEXT,
    DisclosureId,
    fires_ai_disclosure,
)
from collector.guardrails.rings import (
    CONNECTIVE_TEXT,
    MAX_REGENERATION_STRIKES,
    SAFE_FALLBACK_TEXT,
    PreCallContext,
    check_outbound,
)
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
        The exact digits the model emitted survive the trip."""
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
        trace of having been called at all.

        Every one of the six is driven explicitly. An earlier version of this
        test ran a script that only reached ``validate_consumer_offer`` and
        ``confirm_agreement``, so reverting to verdict-only logging left all of
        its assertions passing — it could not catch the regression it exists
        for.
        """
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store)
            agent.open_call()
            agent._perceive("Yes, this is Dana.")
            for call in (
                _call("propose_offer"),
                _call("validate_consumer_offer", payment_count=1, cadence="immediate", total="50"),
                _call("record_refusal"),
                _call("concede"),
                _call("confirm_agreement"),
                _call("end_call", reason="wrapped up"),
            ):
                agent._run_tool(call)
            invocations = store.tool_calls(agent.call_id)

        assert {i.tool for i in invocations} == {s.name for s in TOOL_SCHEMAS}, (
            "all six tools must leave a trace, not just the ruling ones"
        )
        assert all(i.latency_ms >= 0 for i in invocations)

    def test_a_failed_tool_call_records_its_reason(self, tmp_path: Path) -> None:
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store)
            agent.open_call()
            # Past the identity gate first, or that refusal is the reason
            # recorded and concede's own guard never runs.
            agent.guard = agent.guard.with_identity_confirmed()
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


class TestInstrumentationDoesNotBreakTheCall:
    """Regressions found by review. Each of these killed a live call.

    The shared shape: a tool or a model call worked, and then *recording* it
    raised or lost the record. Instrumentation added to observe a system must
    not be able to take it down.
    """

    def test_a_float_argument_survives_the_audit_write(self, tmp_path: Path) -> None:
        """JSON has one number type and it decodes to float. ``_parse_money``
        exists to absorb that; logging the raw arguments must not undo it.

        Before the fix: ``execute()`` returned ``ok`` and then ``to_jsonable``
        raised ``TypeError: float 500.5 in an audit record``, which unwound
        through ``_run_tool`` and ended the call.
        """
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store)
            agent.open_call()
            agent._perceive("Yes, this is Dana.")
            result = agent._run_tool(
                _call("validate_consumer_offer", total=500.5, payment_count=2, cadence="monthly")
            )
            (invocation,) = [
                i for i in store.tool_calls(agent.call_id) if i.tool == "validate_consumer_offer"
            ]

        assert result.ok
        assert not agent.ended
        # The digits the model actually emitted, kept as a string — never a
        # float in the record.
        assert invocation.arguments["total"] == "500.5"

    def test_a_failed_model_call_with_no_usage_still_leaves_a_reason(self, tmp_path: Path) -> None:
        """Gating the event on ``usage`` meant a client that reports an error
        without one logged nothing — "never fail into silence" held for the
        consumer's ear and not for the audit trail."""

        class BareFailure:
            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                return LLMResponse(error="APIConnectionError: reset by peer")

        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=BareFailure())
            _, spoken = agent.open_call()
            calls = store.model_calls(agent.call_id)

        assert spoken == SAFE_FALLBACK_TEXT
        assert calls, "a failure with no usage record must still be on the trail"
        assert "APIConnectionError" in (calls[0].error or "")


class TestArgumentsCannotFalsifyTheRecord:
    def test_a_non_finite_amount_is_refused_at_the_boundary(self) -> None:
        """``Decimal("NaN")`` is a valid Decimal and quantizes to NaN, so it
        cleared Money and reached ``validate_offer`` — which raised
        ``InvalidOperation`` out of a function documented as pure and total,
        escaping ``execute``'s ``ArgumentError`` catch and dropping the call.
        """
        result = execute(
            _call("validate_consumer_offer", total="NaN", payment_count=1, cadence="immediate"),
            ToolContext.opening(POLICY),
        )
        assert not result.ok
        assert "finite" in result.payload["error"]

    @pytest.mark.parametrize("bad", ["NaN", "-NaN", "sNaN", "Infinity", "-Infinity"])
    def test_no_non_finite_spelling_gets_through(self, bad: str) -> None:
        result = execute(
            _call("validate_consumer_offer", total=bad, payment_count=1, cadence="immediate"),
            ToolContext.opening(POLICY),
        )
        assert not result.ok

    def test_an_unknown_argument_is_named_not_ignored(self) -> None:
        """Silently dropping it turns a key confusion into a falsified decision
        record: ``amount`` instead of ``total`` leaves ``total`` absent, the
        full balance is assumed, the engine *accepts* $1,000 in one payment,
        and the agreement describes a proposal the consumer never made.
        """
        result = execute(
            _call("validate_consumer_offer", amount="500", payment_count=1, cadence="immediate"),
            ToolContext.opening(POLICY),
        )
        assert not result.ok
        assert "amount" in result.payload["error"]
        assert "total" in result.payload["error"], "name what it should have said instead"
        assert result.verdict is None, "nothing was ruled on"

    def test_a_deliberate_full_balance_offer_still_works(self) -> None:
        """The guard above must not block the legitimate case it resembles:
        omitting ``total`` on purpose means the balance stands."""
        result = execute(
            _call("validate_consumer_offer", payment_count=1, cadence="immediate"),
            ToolContext.opening(POLICY),
        )
        assert result.ok
        assert result.verdict is not None and result.verdict.outcome == "accept"


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
        assert isinstance(cost, Decimal), "no floats, cost reports included"

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
# The compliance score
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

        monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
        assert resolve_model() == MODEL
        monkeypatch.setenv(MODEL_ENV_VAR, "claude-opus-5")
        assert resolve_model() == "claude-opus-5"
        assert resolve_model("claude-haiku-4-5") == "claude-haiku-4-5", "an explicit arg wins"


class TestStoreConfiguration:
    def test_the_db_path_can_be_set_by_the_environment(self, monkeypatch, tmp_path: Path) -> None:
        """A CWD-relative default breaks on the first deploy whose entrypoint
        does not run from the repo root."""
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
        assert default_db_path() == DEFAULT_DB_PATH
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "elsewhere.db"))
        assert default_db_path() == tmp_path / "elsewhere.db"

    def test_wal_is_on_so_a_commit_does_not_block_the_turn(self, tmp_path: Path) -> None:
        # Through _rows, not _conn: the connection is confined to the store's
        # worker thread, so touching it from the test thread raises
        # ProgrammingError rather than reporting the pragma.
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            journal = store._rows("PRAGMA journal_mode", ())[0][0]
            synchronous = store._rows("PRAGMA synchronous", ())[0][0]

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
            agent = _agent(store, llm=_UsageMock())
            agent.open_call()
            for said in SCRIPT:
                if agent.ended:
                    break
                agent.turn(said)
            kinds = {e.EVENT_TYPE for e in store.trace(agent.call_id)}

        assert EventType.TOOL_CALL in kinds
        assert EventType.MODEL_CALL in kinds, "dropping it from _TRACE_EVENT_TYPES must fail here"


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
            # Nor mid-date. Split at the period in "Jan.", neither fragment
            # matches a date pattern and the orphaned "15" is too small to read
            # as money, so both clear a guard that blocks the whole sentence.
            (
                "Your first payment is due Jan. 15. Okay? ",
                ["Your first payment is due Jan. 15.", "Okay?"],
                "",
            ),
            ("We can start Sept. 3. Fine? ", ["We can start Sept. 3.", "Fine?"], ""),
            # The cost of that: a real sentence ending in an abbreviation
            # reaches TTS a beat late, still whole and still guarded.
            ("Back in Aug. We agreed then. ", ["Back in Aug. We agreed then."], ""),
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


# -- scripted streamers ----------------------------------------------------
#
# Every one of these speaks in more than one sentence, because that is the gap
# the suite had: ``MINI_MIRANDA_TEXT`` and ``AI_DISCLOSURE_TEXT`` are each a
# single sentence, so nothing here ever guarded a disclosure that spans two.


class _Streamer:
    """Base for a scripted stream. ``respond`` serves the un-streamed opening."""

    lines: tuple[str, ...] = ()

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        return LLMResponse(text=_GREETING)

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        for line in self.lines:
            yield TextDelta(f"{line} ")
        yield StreamCompleted(LLMResponse(text=" ".join(self.lines)))


class _AcknowledgesThenDiscloses(_Streamer):
    """Answers "are you a robot?" — in its second sentence, the way a model
    that acknowledges before it answers actually talks."""

    lines = (
        "That's a fair question, and I'm glad you asked.",
        "Yes, I am an AI assistant, and I can put you through to a person.",
        "Would you like me to keep going?",
    )


class _NeverDiscloses(_Streamer):
    """Ducks the question entirely, in every turn it is asked."""

    lines = (
        "I understand the concern.",
        "Let's stay focused on what I can help with.",
    )


class _SplitMiniMiranda(_Streamer):
    """The canonical FDCPA wording, spoken as the two sentences it is."""

    lines = (
        "Thank you for confirming.",
        "This is an attempt to collect a debt.",
        "Any information obtained will be used for that purpose.",
        "I'd like to find something that works for you.",
    )


class _ThreatensAfterDisclosing(_Streamer):
    """Clears the disclosure, then threatens. The block lands mid-stream, so
    the round aborts before ``StreamCompleted`` ever arrives."""

    lines = (
        MINI_MIRANDA_TEXT,
        "Pay today or we will garnish your wages.",
        "So what will it be?",
    )


class _InventsADueDate(_Streamer):
    """No engine result authorized a date, and nothing in ``agent.py`` ever
    passes one, so any date the model originates has to be blocked."""

    lines = (
        MINI_MIRANDA_TEXT,
        "Your first payment is due Jan. 15.",
        "Does that work?",
    )


class _SpeaksWhileItLooksSomethingUp:
    """Narrates the lookup and calls the tool in the same round — the shape
    that makes round two's prompt matter."""

    def __init__(self) -> None:
        self.prompts: list[tuple[Message, ...]] = []

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        return LLMResponse(text=_GREETING)

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        self.prompts.append(messages)
        if len(self.prompts) == 1:
            yield TextDelta("Let me see what I can put together for you. ")
            yield StreamCompleted(
                LLMResponse(
                    text="Let me see what I can put together for you.",
                    tool_calls=(ToolCall(name="propose_offer"),),
                )
            )
            return
        yield TextDelta("Here is where that leaves us. ")
        yield StreamCompleted(LLMResponse(text="Here is where that leaves us."))


class _AlwaysAnotherTool:
    """Answers every round with one more tool call and never says a word."""

    def __init__(self) -> None:
        self.opened = False
        self.rounds = 0

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        if not self.opened:
            self.opened = True
            return LLMResponse(text=_GREETING)
        return LLMResponse(tool_calls=(ToolCall(name="record_refusal"),))

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        self.rounds += 1
        yield StreamCompleted(LLMResponse(tool_calls=(ToolCall(name="record_refusal"),)))


class TestOpeningIsNotStreamed:
    def test_the_greeting_is_guarded_whole_because_the_rule_is_turn_scoped(self) -> None:
        """The AI-disclosure rule governs the first agent *turn*. A first
        sentence that is not the disclosure cannot be judged against it without
        knowing what the second sentence will say, so the opening is guarded as
        one unit — sentence-by-sentence guarding would block every greeting
        whose disclosure is not its opening clause.
        """
        agent = _agent()
        _, spoken = agent.open_call()

        assert spoken is not None
        assert spoken != SAFE_FALLBACK_TEXT, "guarded whole, this clears; per sentence it would not"
        # The disclosure itself, not a prefix of it: asserting the leading
        # clause ("Before we go further") would pass on a greeting that never
        # mentions an AI at all.
        assert fires_ai_disclosure(spoken), spoken
        assert DisclosureId.AI_DISCLOSURE in agent.guard.disclosures.fired

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
        granularity moves, plus the one rule ``stream_turn``'s docstring names
        (abort, do not regenerate). Anything else is a compliance divergence.

        The script is deliberately not the happy path. ``SCRIPT`` alone drives
        one turn shape through ``MockLLMClient``, where the paths *cannot* part
        company, which is why this test stayed green through three defects. It
        now asks whether it is talking to a machine — a rule scoped to the turn,
        and so the one a per-sentence guard silently narrows — and it refuses,
        which is the turn that spends two engine round trips.

        What it still cannot reach is a round that speaks *and* calls a tool:
        ``_act`` discards the text on a tool round while the streaming path
        speaks it, so the two paths differ there by construction. That shape has
        its own test, ``test_a_later_round_sees_what_the_turn_already_spoke``.
        """
        script = [
            "Yes, this is Dana.",
            "Hold on — am I talking to a machine?",
            "I could do $500 down.",
            "No, that's too much.",
            "Okay, yes, let's do that.",
        ]
        spoken_text, spoken_stream = [], []

        text_agent = _agent()
        text_agent.open_call()
        stream_agent = _agent()
        stream_agent.open_call()

        for said in script:
            if not text_agent.ended:
                spoken_text.append(text_agent.turn(said).spoken or "")
            if not stream_agent.ended:
                spoken_stream.append(" ".join(stream_agent.stream_turn(said)))

        assert spoken_stream == spoken_text

        # The transcript, not just the audio: it is what the next turn is asked
        # with, and a line written into it twice is a line the model reads twice.
        def transcript(agent: NegotiationAgent) -> list[str]:
            return [m.content for m in agent.messages if m.role == "agent"]

        assert transcript(stream_agent) == transcript(text_agent)

        for agent in (text_agent, stream_agent):
            assert any(fires_ai_disclosure(s) for s in transcript(agent)[1:]), (
                "the consumer asked mid-call; it must be answered, not deferred"
            )
            assert not agent.guard.disclosures.ai_disclosure_requested, (
                "a request left pending blocks every turn after it, with no way out"
            )

        stream_report, text_report = stream_agent.close(), text_agent.close()
        assert stream_report.outcome is text_report.outcome
        assert stream_report.compliant is text_report.compliant
        assert stream_report.turns == text_report.turns

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

    def test_a_blocked_sentence_aborts_rather_than_regenerating_once_audio_exists(
        self,
    ) -> None:
        """The text path can retry because nothing was spoken. Here the earlier
        sentences are already audio, so a retry would contradict what the
        consumer just heard.

        The turn closes on the connective rather than the scripted fallback —
        see ``TestABlockAfterRealSpeechClosesTheThought``. What this test is
        about is that the stream *aborts*: the blocked sentence never goes out
        and neither does anything the model wrote after it.
        """

        class ThreateningStreamer:
            attempts = 0

            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                # Used only for the opening line, which stream_turn does not drive.
                return LLMResponse(text=_GREETING)

            def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
                type(self).attempts += 1
                yield TextDelta("Am I speaking with the account holder? ")
                yield TextDelta("Pay today or we will garnish your wages. ")
                yield TextDelta("So what will it be? ")
                yield StreamCompleted(LLMResponse(text="..."))

        llm = ThreateningStreamer()
        agent = _agent(llm=llm)
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))

        assert llm.attempts == 1, "the model was asked again after speech was already out"
        assert "Am I speaking with the account holder?" in spoken
        assert not any("garnish" in s for s in spoken)
        # Nothing substantive was said, so the closer is the fallback rather
        # than the connective — see ``TestABlockAfterRealSpeechClosesTheThought``.
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

    def test_an_abandoned_stream_records_what_it_already_spoke(self, tmp_path: Path) -> None:
        """Barge-in is the ordinary case on a voice line, not a dropped call.

        The sentence reached the audit log the moment it cleared the guard, so
        a turn that goes unrecorded leaves the log holding speech the
        transcript denies: the next prompt omits a line the consumer heard, and
        ``CallReport.turns`` undercounts. The two records have to agree.
        """
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store)
            agent.open_call()
            turns_before = len(agent.turns)
            stream = agent.stream_turn("Yes, this is Dana.")
            heard = next(stream)
            stream.close()
            logged = [
                event.text
                for event in store.trace(agent.call_id)
                if isinstance(event, TurnRecorded) and event.speaker is Speaker.AGENT
            ]

        assert heard in logged, "the guard put it on the trail before it reached TTS"
        assert len(agent.turns) == turns_before + 1
        assert agent.turns[-1].spoken == heard
        transcript = [m.content for m in agent.messages if m.role == "agent"]
        assert heard in transcript, "the audit log and the transcript must not disagree"

    def test_a_runaway_tool_loop_still_says_something(self) -> None:
        """A client that asks for one more tool every round exhausts the rounds
        and used to hang up in silence. Three docstrings in ``agent.py`` promise
        otherwise — a turn that produces nothing to say is a dead phone line —
        and both paths have to keep that promise."""
        streamed = _agent(llm=_AlwaysAnotherTool())
        streamed.open_call()
        spoken = list(streamed.stream_turn("Yes, this is Dana."))

        assert spoken, "the streaming path went silent"
        assert streamed.turns[-1].spoken

        texted = _agent(llm=_AlwaysAnotherTool())
        texted.open_call()
        assert texted.turn("Yes, this is Dana.").spoken, "and so did the text path"

    def test_a_later_round_sees_what_the_turn_already_spoke(self) -> None:
        """The turn's speech reached ``self.messages`` only once the round loop
        had exited, so round two was asked with the tool result and *not* with
        the sentence already in the consumer's ear. A model reading that prompt
        says the sentence again — over the top of itself, on live audio, which
        is the exact failure abort-instead-of-regenerate exists to avoid."""
        client = _SpeaksWhileItLooksSomethingUp()
        agent = _agent(llm=client)
        agent.open_call()
        list(agent.stream_turn("Yes, this is Dana."))

        assert len(client.prompts) == 2, "the tool call has to buy a second round"
        second = client.prompts[1]
        spoke_at = [
            i for i, m in enumerate(second) if m.role == "agent" and "put together" in m.content
        ]
        assert spoke_at, "round two was asked without the sentence round one already spoke"
        tool_at = [i for i, m in enumerate(second) if m.role == "tool"]
        assert tool_at and tool_at[-1] > spoke_at[-1], "spoken first, then the engine's answer"

    def test_the_scripted_closing_line_is_written_to_the_transcript_once(self) -> None:
        """``_speak_verbatim`` put the scripted line into ``self.messages`` and
        the end-of-turn join put it there again — two consecutive assistant
        messages carrying the same line.

        This turn speaks before it is blocked, so the line that closes it is
        the connective; the duplication it guards against is a property of
        every code-authored line, whichever one applies.
        """
        agent = _agent(llm=_ThreatensAfterDisclosing())
        agent.open_call()
        before = sum(1 for m in agent.messages if m.role == "agent")
        list(agent.stream_turn("Yes, this is Dana."))
        written = [m.content for m in agent.messages if m.role == "agent"][before:]

        assert sum(m.count(CONNECTIVE_TEXT) for m in written) == 1
        assert len(written) == 1, "one assistant message for the turn, not one per line"

    def test_an_aborted_round_still_records_its_model_call(self, tmp_path: Path) -> None:
        """The round returned on the blocked sentence, before ``StreamCompleted``
        arrived, so ``_record_model_call`` never ran — and the blocked rounds
        are the ones whose latency and cost you most want to account for."""
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=_ThreatensAfterDisclosing())
            agent.open_call()
            spoken = list(agent.stream_turn("Yes, this is Dana."))
            calls = store.model_calls(agent.call_id)

        assert spoken[-1] == CONNECTIVE_TEXT, "the round did abort"
        assert calls, "an aborted round still spent a model call"
        assert any(c.stop_reason == "aborted" for c in calls)
        assert all(c.latency_ms >= 0 for c in calls)

    def test_an_abbreviated_date_cannot_slip_through_split_in_two(self) -> None:
        """Nothing in ``agent.py`` passes ``extra_dates``, so an engine has
        never authorized a date and every one the model originates must be
        blocked. Splitting the sentence at the period in "Jan." handed the
        guard two fragments that each classify as harmless."""
        agent = _agent(llm=_InventsADueDate())
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))

        assert not any("Jan" in s for s in spoken), "an unauthorized date reached TTS"
        assert not any("15" in s for s in spoken)
        assert spoken[-1] == CONNECTIVE_TEXT


def _notes(agent: NegotiationAgent) -> list[str]:
    """The guard's notes back to the model — every system message except the
    system prompt itself, which is always ``messages[0]``."""
    return [m.content for m in agent.messages[1:] if m.role == "system"]


class _ThreateningStreamer:
    """Speaks one clean sentence, then one the prohibited-language ring blocks."""

    def __init__(self) -> None:
        self.attempts = 0

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        return LLMResponse(text=_GREETING)

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        self.attempts += 1
        yield TextDelta("Am I speaking with the account holder? ")
        yield TextDelta("Pay today or we will garnish your wages. ")
        yield StreamCompleted(LLMResponse(text="..."))


class _ToolsForeverThenThreatens:
    """Asks for one more tool every round until the budget is nearly gone,
    then blocks — putting the block on the loop's last available iteration."""

    def __init__(self, tool_rounds: int) -> None:
        self.tool_rounds = tool_rounds
        self.attempts = 0

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        return LLMResponse(text=_GREETING)

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        self.attempts += 1
        if self.attempts <= self.tool_rounds:
            yield StreamCompleted(LLMResponse(tool_calls=(ToolCall(name="record_refusal"),)))
            return
        yield TextDelta("Pay today or we will garnish your wages. ")
        yield StreamCompleted(LLMResponse(text="..."))


class TestABlockedStreamTellsTheModelWhy:
    """The text path names the violation back to the model and the next
    generation is informed by it (``_guard_and_speak``). The streaming path
    threw that away: it aborted, spoke a scripted line, and left the message
    history looking exactly as it did before — so the model's next turn had
    no reason not to reach for the same blocked phrasing again.
    """

    def test_the_violation_reason_reaches_the_message_history(self) -> None:
        agent = _agent(llm=_ThreateningStreamer())
        agent.open_call()
        list(agent.stream_turn("Yes, this is Dana."))

        notes = _notes(agent)
        assert notes, "the model was told nothing about why it was cut off"
        assert "THREAT" in notes[-1]
        assert "garnish" in notes[-1], "name the phrasing that was blocked"

    def test_the_note_follows_what_was_actually_spoken(self) -> None:
        """History order is what the next round reads. A note about a blocked
        sentence filed *before* the sentences that were spoken reads as though
        the spoken ones were the problem."""
        agent = _agent(llm=_ThreateningStreamer())
        agent.open_call()
        list(agent.stream_turn("Yes, this is Dana."))

        spoke_at = max(
            i for i, m in enumerate(agent.messages) if "account holder" in m.content
        )
        noted_at = max(i for i, m in enumerate(agent.messages) if m.role == "system")
        assert spoke_at < noted_at, [m.role for m in agent.messages]

    def test_a_clean_turn_leaves_no_note(self) -> None:
        agent = _agent()
        agent.open_call()
        list(agent.stream_turn("Yes, this is Dana."))

        assert not _notes(agent)


class _BlocksThenComplies:
    """Blocks on its first streamed attempt, then says something clean.

    The block lands on the *first* sentence, so nothing has been spoken and
    the streaming contract's reason for aborting — "a retry would contradict
    what the consumer just heard" — does not apply.
    """

    def __init__(self, attempts_before_complying: int = 1) -> None:
        self.attempts_before_complying = attempts_before_complying
        self.attempts = 0
        self.prompts: list[tuple[Message, ...]] = []

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        return LLMResponse(text=_GREETING)

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        self.attempts += 1
        self.prompts.append(messages)
        if self.attempts <= self.attempts_before_complying:
            yield TextDelta("Pay today or we will garnish your wages. ")
        else:
            yield TextDelta("Thanks for confirming. ")
            yield TextDelta("What would be manageable for you? ")
        yield StreamCompleted(LLMResponse(text="..."))


class TestTheRoundBudgetIsNotInflatedByTheRewriteAllowance:
    def test_a_turn_that_never_blocks_gets_the_tool_budget_it_always_had(self) -> None:
        """The rewrite allowance was added to the loop bound unconditionally,
        so every turn — including one with no guard trip at all — bought two
        extra model round-trips and two extra tool batches before giving up.
        On a voice line that is latency and spend the budget exists to cap."""
        llm = _AlwaysAnotherTool()
        agent = _agent(llm=llm)
        agent.open_call()
        list(agent.stream_turn("Yes, this is Dana."))

        assert llm.rounds == MAX_TOOL_ROUNDS + 1, llm.rounds


class TestAZeroSpokenStreamBlockRegenerates:
    """The streaming path aborted to the scripted fallback on *any* block.

    The reason it gives is sound only when something has already been
    spoken: a retry would then contradict live audio. When the block lands
    before a single sentence has gone to TTS, nothing has been contradicted,
    and the turn can be rewritten exactly the way the text path rewrites it.
    Aborting there spent a scripted non-sequitur on a turn that had every
    chance of clearing on the second attempt.
    """

    def test_a_block_before_any_speech_is_rewritten_not_abandoned(self) -> None:
        llm = _BlocksThenComplies()
        agent = _agent(llm=llm)
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))
        joined = " ".join(spoken)

        assert llm.attempts == 2, "the turn was never retried"
        assert "Thanks for confirming." in joined
        assert SAFE_FALLBACK_TEXT not in joined
        assert not any("garnish" in s for s in spoken)
        assert agent.turns[-1].blocked, "the block still belongs on the record"

    def test_the_retry_is_told_why_before_it_runs(self) -> None:
        """A rewrite asked with unchanged context reproduces the blocked
        phrasing and burns the strike budget for nothing."""
        llm = _BlocksThenComplies()
        agent = _agent(llm=llm)
        agent.open_call()
        list(agent.stream_turn("Yes, this is Dana."))

        assert len(llm.prompts) == 2, "no retry to inspect"
        retry_prompt = " ".join(m.content for m in llm.prompts[1])
        assert "THREAT" in retry_prompt, "the retry was asked blind"
        assert "garnish" in retry_prompt, "and was not told which phrasing to drop"

    def test_rewrites_are_capped_by_the_existing_strike_budget(self) -> None:
        """A model that will not comply must still stop costing round-trips."""
        llm = _BlocksThenComplies(attempts_before_complying=99)
        agent = _agent(llm=llm)
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))

        assert llm.attempts == MAX_REGENERATION_STRIKES
        assert spoken[-1] == SAFE_FALLBACK_TEXT

    def test_a_block_after_real_speech_still_does_not_retry(self) -> None:
        """The contract that survives: those sentences are already audio."""
        llm = _ThreateningStreamer()
        agent = _agent(llm=llm)
        agent.open_call()
        list(agent.stream_turn("Yes, this is Dana."))

        assert llm.attempts == 1, "the model was asked again after speech was already out"
        assert agent.turns[-1].spoken is not None
        assert "Am I speaking with the account holder?" in agent.turns[-1].spoken

    @pytest.mark.parametrize("tool_rounds", [MAX_TOOL_ROUNDS, MAX_TOOL_ROUNDS + 2])
    def test_a_rewrite_is_never_recorded_when_no_round_remains_to_run_it(
        self, tmp_path: Path, tool_rounds: int
    ) -> None:
        """The action was decided from "nothing spoken, strikes left" alone,
        which ignores whether the loop has an iteration left to *do* the
        rewrite in. A block on the last one recorded REGENERATED and then
        fell straight out of the loop to the scripted line — an audit trail
        claiming a rewrite that never happened.

        The invariant, whatever the budget: a recorded rewrite means the
        model really was asked to speak again. So a turn that reached the
        guard only once cannot have rewritten anything.
        """
        llm = _ToolsForeverThenThreatens(tool_rounds=tool_rounds)
        with AuditStore(tmp_path / f"budget{tool_rounds}.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=llm)
            agent.open_call()
            list(agent.stream_turn("Yes, this is Dana."))
            actions = [
                event.action
                for event in store.trace(agent.call_id)
                if isinstance(event, GuardrailTripped)
            ]

        speaking_attempts = llm.attempts - tool_rounds
        if GuardrailAction.REGENERATED in actions:
            assert speaking_attempts >= 2, (
                f"claimed a rewrite after {speaking_attempts} attempt(s) to speak"
            )
        assert agent.turns[-1].was_regenerated == bool(actions)

    def test_the_audit_log_says_regenerated_only_when_it_regenerated(
        self, tmp_path: Path
    ) -> None:
        """An audit trail that says "blocked" about a turn that was rewritten,
        or "regenerated" about one that was abandoned, is a false record of
        what the guard did."""
        with AuditStore(tmp_path / "retried.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=_BlocksThenComplies())
            agent.open_call()
            list(agent.stream_turn("Yes, this is Dana."))
            retried = [
                event.action
                for event in store.trace(agent.call_id)
                if isinstance(event, GuardrailTripped)
            ]

        assert retried and all(a is GuardrailAction.REGENERATED for a in retried), retried

        with AuditStore(tmp_path / "abandoned.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=_ThreateningStreamer())
            agent.open_call()
            list(agent.stream_turn("Yes, this is Dana."))
            abandoned = [
                event.action
                for event in store.trace(agent.call_id)
                if isinstance(event, GuardrailTripped)
            ]

        # Nothing substantive was spoken, so the turn is handed to the
        # scripted fallback and the trail says so — the text path's own word
        # for the same situation. See TestTheTrailSaysHowAStreamedTurnClosed.
        assert abandoned and all(
            a is GuardrailAction.SAFE_FALLBACK for a in abandoned
        ), abandoned


def _actions(store: AuditStore, agent: NegotiationAgent) -> list[GuardrailAction]:
    return [
        event.action
        for event in store.trace(agent.call_id)
        if isinstance(event, GuardrailTripped)
    ]


class TestTheTrailSaysHowAStreamedTurnClosed:
    """A streamed turn now ends one of three ways — rewritten, closed on the
    connective, or handed to the scripted fallback. The trail knew two
    verbs, so it could not say which line the consumer actually heard, and
    it disagreed with the text path about the one situation they share.
    """

    def test_a_streamed_fallback_is_recorded_the_way_the_text_path_records_it(
        self, tmp_path: Path
    ) -> None:
        """Same situation, same word for it. Counting scripted lines off the
        trail otherwise undercounts the voice path to zero."""
        with AuditStore(tmp_path / "stream.db", json_dir=tmp_path) as store:
            streamed = _agent(store, llm=_BlocksThenComplies(attempts_before_complying=99))
            streamed.open_call()
            spoken = list(streamed.stream_turn("Yes, this is Dana."))
            streamed_actions = _actions(store, streamed)

        with AuditStore(tmp_path / "text.db", json_dir=tmp_path) as store:
            texted = _agent(store, llm=_AlwaysBlocked())
            texted.open_call()
            texted._perceive("Yes, this is Dana.")
            texted.turn("What are my options?")
            text_actions = _actions(store, texted)

        assert spoken[-1] == SAFE_FALLBACK_TEXT
        assert GuardrailAction.SAFE_FALLBACK in text_actions, "the text path's own word"
        assert GuardrailAction.SAFE_FALLBACK in streamed_actions, streamed_actions

    def test_a_connective_close_is_not_recorded_as_a_fallback(self, tmp_path: Path) -> None:
        """The consumer heard the offer plus "Does that work for you?" in one
        case and a scripted restart in the other. A reader of the trail has
        to be able to tell those apart."""
        with AuditStore(tmp_path / "connective.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=_ThreatensAfterDisclosing())
            agent.open_call()
            spoken = list(agent.stream_turn("Yes, this is Dana."))
            actions = _actions(store, agent)

        assert spoken[-1] == CONNECTIVE_TEXT
        assert actions and GuardrailAction.SAFE_FALLBACK not in actions, actions
        assert GuardrailAction.CONNECTIVE in actions, actions


class TestABlockAfterRealSpeechClosesTheThought:
    """The scripted fallback restarts the conversation — "let me keep this
    simple, what would work for you?" That is a recovery when the consumer
    heard nothing. After the agent has just finished laying out an offer it
    is a non-sequitur that talks over its own proposal, and the offer
    already stands on its own. A short connective closes the thought and
    leaves it standing.
    """

    def test_the_connective_replaces_the_fallback_after_substantive_speech(self) -> None:
        agent = _agent(llm=_ThreatensAfterDisclosing())
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))

        assert spoken[-1] == CONNECTIVE_TEXT
        assert SAFE_FALLBACK_TEXT not in spoken
        assert MINI_MIRANDA_TEXT in spoken, "the turn did say something that stands"
        assert not any("garnish" in s for s in spoken), "the block still holds"

    def test_a_turn_that_only_said_pleasantries_gets_the_fallback(self) -> None:
        """"Does that work for you?" asks about something. After "Thanks for
        confirming." it is the same non-sequitur it was brought in to
        replace, just shorter — there is no offer standing for it to refer
        to. The gate is substantive speech, not any speech at all.
        """

        class _ThanksThenThreatens:
            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                return LLMResponse(text=_GREETING)

            def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
                yield TextDelta("Thanks for confirming. ")
                yield TextDelta("Pay today or we will garnish your wages. ")
                yield StreamCompleted(LLMResponse(text="..."))

        agent = _agent(llm=_ThanksThenThreatens())
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))

        assert "Thanks for confirming." in spoken
        assert spoken[-1] == SAFE_FALLBACK_TEXT
        assert CONNECTIVE_TEXT not in spoken

    def test_a_turn_that_never_spoke_still_gets_the_fallback(self) -> None:
        """Nothing to connect to, so there is nothing to close."""
        agent = _agent(llm=_BlocksThenComplies(attempts_before_complying=99))
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))

        assert spoken[-1] == SAFE_FALLBACK_TEXT

    def test_a_pending_ai_disclosure_still_wins(self) -> None:
        """The fallback carries the AI disclosure when the consumer has just
        asked whether they are talking to a machine — that question is the
        obligation, and a connective that closes the thought instead would
        silently drop the answer.

        Checked at the seam, like the escalation case and for a similar
        reason: ``stream_turn`` cannot reach this branch either.
        ``_owes_a_disclosure`` withholds every sentence while the request is
        outstanding, so a turn with the flag set has no speech to close on
        and takes the rewrite path instead. Driving it through
        ``stream_turn`` produces a test that passes on the strength of that
        detour while asserting nothing about this branch at all.
        """
        agent = _agent()
        agent.open_call()
        agent._perceive("Wait, am I talking to a machine?")
        assert agent.guard.disclosures.ai_disclosure_requested

        line = agent._stream_connective()
        assert line != CONNECTIVE_TEXT
        assert fires_ai_disclosure(line), line

    def test_an_escalation_closing_line_is_never_replaced(self) -> None:
        """Once the consumer has escalated, ``fallback_for`` returns the
        closing line and that line is the compliance obligation — a
        connective inviting more negotiation must not displace it.

        Checked at the seam rather than through ``stream_turn``, which cannot
        reach it: ``_perceive`` hands an escalated turn straight to
        ``_escalate`` and ends the call before any generation runs.
        """
        agent = _agent()
        agent.open_call()
        agent._perceive("I've retained a lawyer for this.")

        assert agent.guard.escalated
        assert agent._stream_connective() != CONNECTIVE_TEXT


class _AlwaysBlocked:
    """Never produces a sentence the guard will pass, on any turn."""

    def __init__(self, opened: bool = False) -> None:
        self.opened = opened

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        if not self.opened:
            self.opened = True
            return LLMResponse(text=_GREETING)
        return LLMResponse(text="Pay today or we will garnish your wages.")

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        yield TextDelta("Pay today or we will garnish your wages. ")
        yield StreamCompleted(LLMResponse(text="..."))


class TestRepeatedFallbacksEscalateToTheStandingOffer:
    """Two fallbacks in a row is the agent asking "what would work for you?"
    twice while the consumer waits for terms it has already been given. The
    second trip says something instead: the offer on the table, rendered by
    code from the engine's own numbers.
    """

    def _agent_with_an_offer(self, llm: object) -> NegotiationAgent:
        """Mid-call: identity confirmed, both disclosures made, an offer on
        the table. The Mini-Miranda matters — nothing substantive clears the
        guard before it, so a restatement of terms cannot precede it either."""
        agent = _agent(llm=llm)
        agent.open_call()
        agent._perceive("Yes, this is Dana.")
        agent._observe_scripted(MINI_MIRANDA_TEXT)
        agent._run_tool(_call("propose_offer"))
        return agent

    def test_the_first_fallback_still_asks_the_open_question(self) -> None:
        agent = self._agent_with_an_offer(_AlwaysBlocked())
        first = agent.turn("What are my options?")

        assert first.spoken == SAFE_FALLBACK_TEXT

    def test_the_second_consecutive_fallback_restates_the_offer(self) -> None:
        agent = self._agent_with_an_offer(_AlwaysBlocked())
        agent.turn("What are my options?")
        second = agent.turn("You still haven't told me anything.")

        offer = agent.tools.standing_offer
        assert offer is not None
        assert second.spoken is not None
        assert second.spoken != SAFE_FALLBACK_TEXT
        for installment in offer.installments:
            assert str(installment.amount) in second.spoken

    def test_the_restatement_clears_the_guard_on_its_own_figures(self) -> None:
        """It is spoken through ``_speak_verbatim``, which bypasses
        ``check_outbound`` entirely — so a line that would *not* have cleared
        is a line that reaches TTS unchecked. Every figure in it comes from
        the engine's own offer, and that has to be true by test, not by
        inspection.

        The default offer is pay-in-full, whose only figure is the account
        balance — and ``authorized_for`` puts the balance in the base set
        before any offer exists. A line built from that clears against a
        guard that has never seen an offer, so it cannot detect a broken
        offer-to-authorized derivation. Concede first, so the figures are
        the schedule's and nothing else.
        """
        agent = self._agent_with_an_offer(_AlwaysBlocked())
        agent._run_tool(_call("record_refusal"))
        agent._run_tool(_call("concede"))
        agent.turn("What are my options?")
        line = agent._fallback_line()
        balance = str(POLICY.original_balance)

        assert line != SAFE_FALLBACK_TEXT
        assert balance not in line, f"only the balance is proven by this: {line}"
        check = check_outbound(agent.guard, line, authorized=agent.authorized)
        assert check.allowed, check.violations

    def test_a_turn_that_speaks_resets_the_count(self) -> None:
        """"Consecutive" is the whole point — one fallback early in a call and
        another ten turns later is not an agent going in circles."""
        agent = self._agent_with_an_offer(_AlwaysBlocked())
        first = agent.turn("What are my options?")
        agent.llm = MockLLMClient()
        spoke = agent.turn("I could do five hundred dollars.")
        agent.llm = _AlwaysBlocked(opened=True)
        again = agent.turn("Sorry, say that again?")

        assert first.spoken == SAFE_FALLBACK_TEXT
        assert spoke.spoken not in (None, SAFE_FALLBACK_TEXT), "the middle turn has to speak"
        assert again.spoken == SAFE_FALLBACK_TEXT, "the count restarts, it does not accumulate"

    def test_the_restatement_is_said_once_not_on_every_trip_thereafter(self) -> None:
        """Reading the consumer identical terms every turn from the second
        onward is the same circling this was meant to break, in a different
        line. It is the *second* trip that says the terms."""
        agent = self._agent_with_an_offer(_AlwaysBlocked())
        lines = [agent.turn(f"Still nothing? ({i})").spoken for i in range(4)]

        assert lines[0] == SAFE_FALLBACK_TEXT
        assert lines[1] is not None and lines[1] != SAFE_FALLBACK_TEXT
        assert lines[2:] == [SAFE_FALLBACK_TEXT, SAFE_FALLBACK_TEXT], lines

    def test_the_restatement_reads_back_a_multi_payment_plan(self) -> None:
        """Every other test offer is a single payment due today, so the join
        and the "in N days" duration figures went unexercised — and a
        duration the engine did not authorize is exactly what the guard
        exists to catch."""
        agent = self._agent_with_an_offer(_AlwaysBlocked())
        agent._run_tool(
            _call(
                "validate_consumer_offer",
                payment_count=3,
                cadence="monthly",
                total="1000.00",
            )
        )
        agent._run_tool(_call("concede"))
        agent.turn("What are my options?")
        line = agent._fallback_line()
        offer = agent.tools.standing_offer

        assert offer is not None and offer.payment_count > 1, offer
        assert line != SAFE_FALLBACK_TEXT
        assert " days" in line, line
        assert check_outbound(agent.guard, line, authorized=agent.authorized).allowed

    def test_no_offer_on_the_table_means_nothing_to_restate(self) -> None:
        agent = _agent(llm=_AlwaysBlocked())
        agent.open_call()
        agent._perceive("Yes, this is Dana.")
        agent.turn("What are my options?")
        second = agent.turn("You still haven't told me anything.")

        assert second.spoken == SAFE_FALLBACK_TEXT

    def test_an_unidentified_consumer_is_never_read_the_terms(self) -> None:
        """``_speak_verbatim`` bypasses the identity ring, and identity is
        revocable mid-call. A code-authored line full of dollar figures must
        gate on it explicitly or it becomes the way around it."""
        agent = self._agent_with_an_offer(_AlwaysBlocked())
        agent.turn("What are my options?")
        agent.guard = agent.guard.with_identity_revoked()

        assert agent._fallback_line() == SAFE_FALLBACK_TEXT


class TestTurnScopedDisclosuresOnTheStreamingPath:
    """The disclosure rules ask what the *turn* said. Applied one sentence at a
    time they quietly become "what the *first* sentence said", which is a
    different and much stricter rule than the one that applies.
    """

    def test_the_answer_may_arrive_in_the_second_sentence(self) -> None:
        """``AI_DISCLOSURE_REQUEST_IGNORED`` is turn-scoped: the turn must carry
        the disclosure. Per sentence, a model that opens with an acknowledgement
        gets sentence one blocked and the whole compliant turn thrown away."""
        agent = _agent(llm=_AcknowledgesThenDiscloses())
        agent.open_call()
        spoken = list(agent.stream_turn("Hold on — are you a real person?"))
        joined = " ".join(spoken)

        assert fires_ai_disclosure(joined), joined
        assert "That's a fair question" in joined, "an acknowledgement is not a violation"
        assert SAFE_FALLBACK_TEXT not in joined
        assert not agent.guard.disclosures.ai_disclosure_requested

    def test_a_fallback_answers_the_question_it_was_reached_for(self) -> None:
        """The abort spoke ``fallback_for`` through ``_speak_verbatim``, which
        bypasses ``check_outbound`` — so ``observe_agent`` never ran, the
        request stayed pending, and every later streamed turn was blocked at
        sentence one for ignoring it. The call could then neither proceed nor
        close, while ``finalize_call`` still scored it compliant because the
        greeting had disclosed once.
        """
        agent = _agent(llm=_NeverDiscloses())
        agent.open_call()
        first = list(agent.stream_turn("Wait, am I talking to a machine?"))

        assert fires_ai_disclosure(" ".join(first)), "required at open *and on request*"
        assert not agent.guard.disclosures.ai_disclosure_requested

        second = list(agent.stream_turn("Fine. What are my options?"))
        assert second[0].startswith("I understand the concern"), (
            "the next turn is blocked at sentence one for a question already answered"
        )
        assert SAFE_FALLBACK_TEXT not in " ".join(second)

    def test_the_mini_miranda_may_be_spoken_as_the_two_sentences_it_is(self) -> None:
        """The *ordering* rule composes sentence by sentence; the *detector*
        does not — it needs both halves in one string. Spoken the way the
        canonical FDCPA wording routinely is, sentence one is substantive, the
        purpose half is still unwritten, and the turn aborts. The disclosure
        then never fires, so the call cannot reach substantive discussion at
        all. ``turn()`` allows the identical text."""
        agent = _agent(llm=_SplitMiniMiranda())
        agent.open_call()
        spoken = list(agent.stream_turn("Yes, this is Dana."))
        joined = " ".join(spoken)

        assert "attempt to collect a debt" in joined
        assert "used for that purpose" in joined
        assert SAFE_FALLBACK_TEXT not in joined
        assert DisclosureId.MINI_MIRANDA in agent.guard.disclosures.fired
