"""OpenTelemetry export.

Four things this proves, all offline and with no collector running:

1. **The whole call is one trace.** Per-event spans that do not roll up to a
   call are not the requirement; they are the thing the requirement exists to
   rule out.
2. **Every layer reaches it** — model calls, tool calls, guardrail trips,
   disclosures, escalations, and the engine's evaluated conditions.
3. **No PII reaches it.** A span attribute leaves the process for a third party
   the way a log drain does, and this repo already keeps the debtor's name, the
   consumer's words and every figure out of that path deliberately.
4. **Instrumentation cannot take the call down** — not when the exporter
   raises, not when the mapping itself raises.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from collector import tracing
from collector.agent import CallReport, NegotiationAgent, _Round
from collector.audit.store import AuditStore
from collector.guardrails.disclosures import AI_DISCLOSURE_TEXT, DisclosureId
from collector.guardrails.rings import PreCallContext
from collector.llm.base import LLMClient, LLMResponse, LLMUsage, Message, ToolCall
from collector.llm.mock_client import MockLLMClient
from collector.policy import PolicyConfig
from collector.tracing import ENDPOINT_ENV_VAR, TRACING_ENV_VAR, TracingConfigError

POLICY = PolicyConfig.default()
SCRIPT = ["Yes, this is Dana.", "I could do $500 down.", "Yes, let's do that."]

_GREETING = f"{AI_DISCLOSURE_TEXT} Am I speaking with the account holder?"


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    """An in-memory provider installed for one test.

    ``SimpleSpanProcessor``, not ``Batch``: it exports on the calling thread, so
    an exporter that raises actually raises *into the turn* — which is the only
    way to test that a failing export cannot drop a call.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.install_tracer_provider(provider)
    try:
        yield exporter
    finally:
        tracing.install_tracer_provider(None)


def _agent(
    store: AuditStore | None = None, llm: LLMClient | None = None, **kwargs: str
) -> NegotiationAgent:
    return NegotiationAgent(llm=llm or MockLLMClient(), policy=POLICY, store=store, **kwargs)


def _run(agent: NegotiationAgent, script: Sequence[str] = SCRIPT) -> CallReport:
    agent.open_call(PreCallContext(account_loaded=True, within_calling_window=True))
    for said in script:
        if agent.ended:
            break
        agent.turn(said)
    return agent.close()


def _names(exporter: InMemorySpanExporter) -> set[str]:
    return {span.name for span in exporter.get_finished_spans()}


def _named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


# ==========================================================================
# Configuration — off by default, loud when wrong
# ==========================================================================


class TestConfiguration:
    def test_tracing_is_off_unless_it_is_asked_for(self, monkeypatch) -> None:
        monkeypatch.delenv(TRACING_ENV_VAR, raising=False)
        assert tracing.configure_tracing() is None

    def test_an_unset_variable_emits_no_spans_at_all(self, monkeypatch) -> None:
        """Not merely "no exporter" — nothing is built and nothing is recorded,
        so the default path costs the call nothing."""
        monkeypatch.delenv(TRACING_ENV_VAR, raising=False)
        tracing.install_tracer_provider(None)
        assert tracing.configure_tracing() is None
        assert tracing._tracer is None, "nothing was built, so nothing is recorded"
        # And a whole call against no provider still completes.
        assert _run(_agent()).turns

    @pytest.mark.parametrize("value", ["on", "true", "1", "langfuse", "otel", "yes"])
    def test_a_malformed_mode_raises_rather_than_reading_as_off(
        self, monkeypatch, value: str
    ) -> None:
        """``COLLECTOR_LLM=antropic`` silently routes to Anthropic and this repo
        has that on the record as a defect. A typo here would silently mean "no
        traces", which is discovered at the moment the traces are needed."""
        monkeypatch.setenv(TRACING_ENV_VAR, value)
        with pytest.raises(TracingConfigError, match="not a tracing mode"):
            tracing.configure_tracing()

    def test_otlp_without_an_endpoint_raises(self, monkeypatch) -> None:
        """The exporter's own default is localhost, so this would otherwise
        build a healthy-looking pipeline that black-holes every span.

        Spelled " OTLP " to pin the other half: case and surrounding whitespace
        are normalized, so only a genuinely different word raises above.
        """
        monkeypatch.setenv(TRACING_ENV_VAR, " OTLP ")
        monkeypatch.delenv(ENDPOINT_ENV_VAR, raising=False)
        with pytest.raises(TracingConfigError, match=ENDPOINT_ENV_VAR):
            tracing.configure_tracing()

    def test_configuring_twice_reuses_one_provider(self, monkeypatch) -> None:
        """``voice_app.entrypoint`` runs once per job. A fresh provider per job
        leaks a ``BatchSpanProcessor`` export thread per call."""
        monkeypatch.setenv(TRACING_ENV_VAR, "otlp")
        monkeypatch.setenv(ENDPOINT_ENV_VAR, "http://localhost:4318/v1/traces")
        tracing.install_tracer_provider(None)
        try:
            first = tracing.configure_tracing()
            assert first is not None
            assert tracing.configure_tracing() is first
        finally:
            if first is not None:
                first.shutdown()
            tracing.install_tracer_provider(None)


# ==========================================================================
# Session-level grouping — the headline requirement
# ==========================================================================


class TestOneTracePerCall:
    def test_every_span_in_a_call_shares_one_trace(self, spans: InMemorySpanExporter) -> None:
        _run(_agent(llm=_UsageMock()))
        finished = spans.get_finished_spans()

        assert finished
        assert len({span.context.trace_id for span in finished}) == 1, (
            "per-event spans that do not roll up to a call are not the requirement"
        )

    def test_the_call_span_is_the_one_root_and_it_parents_the_rest(
        self, spans: InMemorySpanExporter
    ) -> None:
        agent = _agent(llm=_UsageMock())
        _run(agent)
        finished = spans.get_finished_spans()

        roots = [span for span in finished if span.parent is None]
        assert [span.name for span in roots] == ["collections_call"]
        root = roots[0]
        assert root.attributes is not None
        assert root.attributes["collector.call_id"] == agent.call_id
        assert root.attributes["langfuse.session.id"] == agent.call_id
        assert root.attributes["collector.channel"] == "text"
        for span in finished:
            if span.parent is not None:
                assert span.parent.span_id == root.context.span_id

    def test_the_root_carries_the_calls_own_verdict(self, spans: InMemorySpanExporter) -> None:
        report = _run(_agent(llm=_UsageMock()))
        (root,) = [s for s in spans.get_finished_spans() if s.parent is None]

        assert root.attributes is not None
        assert root.attributes["collector.outcome"] == report.outcome.value
        assert root.attributes["collector.compliant"] is report.compliant
        assert root.attributes["collector.turn_count"] == report.turns

    def test_a_turn_run_off_the_main_thread_stays_in_one_trace(
        self, spans: InMemorySpanExporter
    ) -> None:
        """``voice_app.llm_node`` runs ``turn()`` under ``asyncio.to_thread``, and
        that is the one path none of the tests above take.

        Every child span names its parent explicitly rather than reading the
        ambient context, and this is the test that says so: drop the explicit
        ``context=`` and the voice path quietly becomes one trace per turn while
        every other test here stays green.

        The second half pins the other decision — the root is opened against an
        empty ``Context``, so a call trace is never sometimes-a-subtree of
        whatever span the SDK happened to have open.
        """
        provider = TracerProvider()
        ambient = provider.get_tracer("someone_else")
        agent = _agent(llm=_UsageMock())
        with ambient.start_as_current_span("job_entrypoint"):
            agent.open_call(PreCallContext(account_loaded=True, within_calling_window=True))
            worker = threading.Thread(target=agent.turn, args=("Yes, this is Dana.",))
            worker.start()
            worker.join()
            agent.close()

        finished = spans.get_finished_spans()
        assert len({span.context.trace_id for span in finished}) == 1
        assert [s.name for s in finished if s.parent is None] == ["collections_call"]

    def test_two_calls_do_not_share_a_trace(self, spans: InMemorySpanExporter) -> None:
        _run(_agent(llm=_UsageMock(), call_id="call-a"))
        _run(_agent(llm=_UsageMock(), call_id="call-b"))
        traces = {span.context.trace_id for span in spans.get_finished_spans()}

        assert len(traces) == 2


# ==========================================================================
# What gets traced
# ==========================================================================


class TestEveryLayerReachesTheTrace:
    def test_model_tool_decision_and_guardrail_spans_are_all_emitted(
        self, spans: InMemorySpanExporter
    ) -> None:
        agent = _agent(llm=_UsageMock())
        agent.open_call()
        agent._perceive("Yes, this is Dana.")
        agent._run_tool(ToolCall(name="propose_offer", arguments={}))
        agent._run_tool(
            ToolCall(
                name="validate_consumer_offer",
                arguments={"payment_count": 2, "cadence": "monthly", "total": "500.00"},
            )
        )
        agent._guard_sentence("Pay today or we will garnish your wages.", _Round())

        assert {"llm_call", "tool_call", "decision", "guardrail_trip"} <= _names(spans)

    def test_an_llm_span_carries_the_cost_it_already_measured(
        self, spans: InMemorySpanExporter
    ) -> None:
        """Attached from ``ModelCalled``, not recomputed: a second cost
        calculation is a second chance to disagree with the audit trail."""

        class Metered:
            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                return LLMResponse(
                    text=_GREETING,
                    usage=LLMUsage(
                        model="claude-sonnet-5",
                        latency_ms=412,
                        input_tokens=1200,
                        output_tokens=30,
                        cache_read_tokens=800,
                        cost_usd=__import__("decimal").Decimal("0.0037"),
                        stop_reason="end_turn",
                    ),
                )

        agent = _agent(llm=Metered())
        agent.open_call()
        (span,) = _named(spans, "llm_call")

        assert span.attributes is not None
        assert span.attributes["gen_ai.request.model"] == "claude-sonnet-5"
        assert span.attributes["gen_ai.usage.input_tokens"] == 1200
        assert span.attributes["gen_ai.usage.output_tokens"] == 30
        assert span.attributes["collector.cache_read_tokens"] == 800
        assert span.attributes["gen_ai.response.finish_reason"] == "end_turn"
        # A string, never a float — the no-floats rule covers the export surface too.
        assert span.attributes["collector.cost_usd"] == "0.0037"
        assert span.attributes["collector.latency_ms"] == 412
        # Back-dated, so the trace is a waterfall rather than a row of ticks.
        assert span.end_time is not None and span.start_time is not None
        assert span.end_time - span.start_time == 412 * 1_000_000

    def test_a_failed_model_call_marks_the_span_without_quoting_the_error(
        self, spans: InMemorySpanExporter
    ) -> None:
        class Failing:
            def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
                return LLMResponse(error="APITimeoutError: request Dana Whitfield timed out")

        agent = _agent(llm=Failing())
        agent.open_call()
        (span,) = _named(spans, "llm_call")

        assert span.attributes is not None
        assert span.attributes["collector.error_type"] == "APITimeoutError"
        assert span.status.is_ok is False
        assert "Dana" not in str(span.attributes), "a provider's error text can quote the request"

    def test_a_tool_span_carries_the_shape_of_the_arguments_not_their_values(
        self, spans: InMemorySpanExporter
    ) -> None:
        agent = _agent()
        agent.open_call()
        agent.guard = agent.guard.with_identity_confirmed()
        agent._run_tool(
            ToolCall(
                name="validate_consumer_offer",
                arguments={"total": "743.21", "payment_count": 3, "cadence": "monthly"},
            )
        )
        (span,) = [s for s in _named(spans, "tool_call") if s.attributes is not None]

        assert span.attributes is not None
        assert span.attributes["collector.tool"] == "validate_consumer_offer"
        assert list(span.attributes["collector.tool.argument_keys"]) == [
            "cadence",
            "payment_count",
            "total",
        ]
        assert span.attributes["collector.tool.argument_count"] == 3
        assert "743.21" not in str(span.attributes)

    def test_a_decision_span_names_every_rule_that_ran_not_just_the_outcome(
        self, spans: InMemorySpanExporter
    ) -> None:
        """The report's vendor test: "if they show you a transcript, the model
        decided; if they show you evaluated conditions and a policy path, the
        engine did." Passing rules included — a record of only the failures
        cannot show what was checked."""
        agent = _agent()
        agent.open_call()
        agent.guard = agent.guard.with_identity_confirmed()
        result = agent._run_tool(
            ToolCall(
                name="validate_consumer_offer",
                arguments={"total": "100.00", "payment_count": 1, "cadence": "immediate"},
            )
        )
        assert result.verdict is not None and result.verdict.outcome != "accept"
        (span,) = _named(spans, "decision")

        assert span.attributes is not None
        passed = set(span.attributes["collector.decision.rules_passed"])
        failed = set(span.attributes["collector.decision.rules_failed"])
        assert passed and failed, "both sides of the trail, or it is only an outcome"
        assert {c.rule_id.value for c in result.verdict.conditions} == passed | failed
        assert span.attributes["collector.decision.outcome"] == result.verdict.outcome
        assert (
            span.attributes["collector.decision.rationale_code"]
            == result.verdict.rationale_code.value
        )

    def test_a_disclosure_firing_is_an_event_on_the_call_span(
        self, spans: InMemorySpanExporter
    ) -> None:
        """A disclosure landing is guard state, not an audit event — nothing is
        written when the Mini-Miranda fires. It belongs on the trace anyway,
        so ``CallTrace`` diffs the fired set instead of missing it."""
        agent = _agent()
        _run(agent)
        (root,) = [s for s in spans.get_finished_spans() if s.parent is None]

        fired = {
            event.attributes["collector.disclosure"]
            for event in root.events
            if event.name == "guardrail.disclosure_fired" and event.attributes is not None
        }
        assert DisclosureId.AI_DISCLOSURE.value in fired
        assert DisclosureId.MINI_MIRANDA.value in fired
        assert fired == {d.value for d in agent.guard.disclosures.fired}

    def test_an_escalation_and_the_callback_it_owes_reach_the_trace(
        self, spans: InMemorySpanExporter
    ) -> None:
        agent = _agent()
        agent.open_call()
        agent.turn("Yes, this is Dana.")
        agent.turn("I've retained an attorney, so talk to them.")
        (span,) = _named(spans, "escalation")

        assert agent.turns[-1].escalated
        assert span.attributes is not None
        assert span.attributes["collector.escalation.trigger"] == "ATTORNEY_REPRESENTATION"
        assert span.attributes["collector.escalation.callback_owed"] is False

    def test_a_blocked_turn_names_the_rule_and_not_the_sentence(
        self, spans: InMemorySpanExporter
    ) -> None:
        agent = _agent()
        agent.open_call()
        round_ = _Round()
        # A sentence is already audio, so this block aborts rather than being
        # rewritten — which is what makes the recorded action "blocked".
        blocked = agent._guard_sentence(
            "Pay today or we will garnish your wages.", round_, ("Hello there.",)
        )
        # One span per blocking rule: the threat, the unauthorized figure and
        # the missing Mini-Miranda all fire on that one sentence.
        trips = _named(spans, "guardrail_trip")

        assert blocked is None
        assert trips
        for span in trips:
            assert span.attributes is not None
            assert span.attributes["collector.guardrail.ring"] == "during_call"
            assert span.attributes["collector.guardrail.action"] == "blocked"
            assert span.attributes["collector.guardrail.rule_id"]
            assert "garnish" not in str(span.attributes)

        # The blocked text and the guard's account of it come back on the
        # round, which is how the caller both aborts and tells the model why.
        assert round_.blocked is not None
        assert round_.note is not None and "THREAT" in round_.note


# ==========================================================================
# PII — a span attribute is an export surface
# ==========================================================================

# Chosen to be findable: nothing else in the repo, the fixtures or the scripted
# client produces these strings, so a substring hit is the value itself.
_NAME = "Zephaniah Quicksilver"
_ACCOUNT = "ACCT-70931"
_AMOUNT = "743.21"
_UTTERANCE = f"Yes, this is {_NAME}, and I could maybe do ${_AMOUNT} a month."
# Escalates (HARDSHIP), and carries both secrets into ``Escalated.detail`` and
# ``consumer_utterance`` — the two fields the escalation mapping does not read.
_ESCALATES = f"I've lost my job and I have no income. {_NAME} has ${_AMOUNT} left."


class _Threatens:
    """Says something the outbound guard has to block, so ``blocked_text`` — the
    one field that is by construction a sentence nobody may repeat — exists."""

    def __init__(self) -> None:
        self.opened = False

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        if not self.opened:
            self.opened = True
            return LLMResponse(text=_GREETING)
        return LLMResponse(
            text=f"Pay {_NAME} ${_AMOUNT} today or we will garnish your wages.",
            usage=LLMUsage(model="mock", latency_ms=5, input_tokens=10, output_tokens=5),
        )


class TestNoPiiInSpanAttributes:
    def test_a_full_turn_leaks_neither_the_debtor_nor_a_figure(
        self, spans: InMemorySpanExporter, tmp_path: Path
    ) -> None:
        """The distinctive name and amount go all the way through: the call
        header, an utterance, a tool argument, an engine verdict, a blocked
        sentence and an escalation. None of them may come out the other side.
        """
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=_Threatens(), consumer_name=_NAME, account_ref=_ACCOUNT)
            agent.open_call(PreCallContext(account_loaded=True, within_calling_window=True))
            agent.turn(_UTTERANCE)
            agent._run_tool(
                ToolCall(
                    name="validate_consumer_offer",
                    arguments={"total": _AMOUNT, "payment_count": 3, "cadence": "monthly"},
                )
            )
            agent.turn(_ESCALATES)
            agent.close()
            # The trail that a reader with authority does hold the values in.
            assert store.tool_calls(agent.call_id)

        rendered = _rendered(spans)
        for secret in (_NAME, "Zephaniah", "Quicksilver", _AMOUNT, "743", "no income", "garnish"):
            assert secret not in rendered, f"{secret!r} reached a span"

        # And the positive half, or a mapper that emits nothing passes this.
        assert _ACCOUNT in rendered, "account_ref is logged by design; it is the correlator"
        assert "guardrail_trip" in rendered
        assert "collector.decision.rules_failed" in rendered
        assert "TOTAL_FLOOR" in rendered
        # Hardship owes a callback, so this pins the attribute the other
        # escalation test can only ever see as False.
        (escalation,) = _named(spans, "escalation")
        assert escalation.attributes is not None
        assert escalation.attributes["collector.escalation.trigger"] == "HARDSHIP"
        assert escalation.attributes["collector.escalation.callback_owed"] is True

    def test_the_transcript_is_never_a_span(self, spans: InMemorySpanExporter) -> None:
        """``TurnRecorded`` is the transcript — both sides of it — and the
        mapping has no case for it at all."""
        agent = _agent(consumer_name=_NAME)
        _run(agent, ["Yes, this is Dana.", "I could do $500 down."])
        rendered = _rendered(spans)

        assert "turn" not in _names(spans)
        assert "Dana" not in rendered
        assert "$500" not in rendered


def _rendered(exporter: InMemorySpanExporter) -> str:
    """Everything a span carries off the process, as one searchable string.

    Names and events as well as attributes: a span event's payload is exported
    exactly like an attribute is, and ``record_exception`` would put a message
    there without anyone writing one.
    """
    parts: list[str] = []
    for span in exporter.get_finished_spans():
        parts.append(span.name)
        parts.append(str(span.attributes))
        parts.append(str(span.status.description))
        for event in span.events:
            parts.append(event.name)
            parts.append(str(event.attributes))
    return "\n".join(parts)


# ==========================================================================
# Instrumentation must not take down the call it observes (a5fc713)
# ==========================================================================


class _RaisingExporter(SpanExporter):
    """A collector that is down, or a network that is not there."""

    def __init__(self) -> None:
        self.attempts = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.attempts += 1
        raise RuntimeError("collector unreachable")


class TestTracingCannotDropACall:
    def test_a_raising_exporter_does_not_cost_the_call_a_turn(self, tmp_path: Path) -> None:
        """Through ``SimpleSpanProcessor``, so the export runs on the turn's own
        thread and really can unwind into it — behind the batching processor
        this would pass no matter what the code did."""
        exporter = _RaisingExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracing.install_tracer_provider(provider)
        try:
            with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
                agent = _agent(store, llm=_UsageMock())
                report = _run(agent)
                trail = store.trace(agent.call_id)
        finally:
            tracing.install_tracer_provider(None)

        assert exporter.attempts, "the export really was attempted, and really did raise"
        assert report.turns == len(agent.turns), "the call completed"
        assert trail, "and the audit trail is untouched"

    def test_a_raising_mapper_does_not_cost_the_call_a_turn(
        self, spans: InMemorySpanExporter, monkeypatch, tmp_path: Path
    ) -> None:
        """The a5fc713 shape exactly: the tool succeeded, and then *recording*
        it raised and unwound through ``_run_tool``. Here the failure is in the
        span mapping rather than the audit encoder."""

        def explode(self: object, tracer: object, event: object) -> None:
            raise TypeError("cannot serialize that")

        monkeypatch.setattr(tracing.CallTrace, "_emit", explode)

        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store, llm=_UsageMock())
            agent.open_call()
            agent.guard = agent.guard.with_identity_confirmed()
            result = agent._run_tool(ToolCall(name="propose_offer", arguments={}))
            survived = not agent.ended
            report = agent.close()
            invocations = store.tool_calls(agent.call_id)

        assert result.ok and survived
        assert report.turns == len(agent.turns)
        assert [i.tool for i in invocations] == ["propose_offer"], "the audit row still landed"
        assert not spans.get_finished_spans(), "and nothing was traced, which is the right cost"

    def test_a_call_with_no_provider_installed_still_runs(self, tmp_path: Path) -> None:
        tracing.install_tracer_provider(None)
        with AuditStore(tmp_path / "c.db", json_dir=tmp_path) as store:
            agent = _agent(store)
            _run(agent)
            assert store.trace(agent.call_id)

    def test_flushing_with_nothing_configured_is_a_no_op(self) -> None:
        tracing.install_tracer_provider(None)
        tracing.flush_traces()


class _UsageMock(MockLLMClient):
    """The scripted client, with a usage record attached to every turn — so the
    model spans carry something to assert on."""

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        response = super().respond(messages)
        usage = LLMUsage(model="mock", latency_ms=1, input_tokens=10, output_tokens=5)
        return LLMResponse(
            text=response.text, tool_calls=response.tool_calls, usage=usage, error=response.error
        )
