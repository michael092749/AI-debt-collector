"""OpenTelemetry export — the whole instrumented surface, minus the words.

    Every LLM call, every tool call, every guardrail event, every engine
    verdict, and the whole call as one trace.

The spans come off the audit event stream rather than off a second set of hooks
threaded through ``agent.py``. Every event the call already writes passes
through ``NegotiationAgent._record``, and those events *are* the instrumented
facts: ``ModelCalled`` carries tokens, cost, latency and stop reason;
``ToolInvoked`` carries arguments and latency; ``GuardrailTripped`` and
``Escalated`` carry the rule and the trigger; ``DecisionRecorded`` carries the
full evaluated-condition trail. Recomputing any of that here would give the
trace and the SQLite trail two chances to disagree.

**The export path is a third party and the audit trail is not.** ``agent.py``
already keeps the debtor's name, the consumer's words and every dollar figure
out of the log and puts codes and counts there instead; span attributes leave
the process the same way a log drain does, so they get the same discipline or
stricter. The mapping below is an allowlist, not a redaction pass: a field
reaches a span because it was named here, and everything else — ``detail``,
``blocked_text``, ``consumer_ref``, ``arguments`` values, ``Condition.actual``,
``Condition.limit``, the whole of ``TurnRecorded`` — is simply never read. What
goes out is enough to find the row: ``call_id`` on the trace, ``turn_index`` on
the span, and the SQLite store holds the rest under the retention policy it was
built for.

This deliberately diverges from the usual LLM-tracing advice on one point: it
asks for the prompt and the completion on each LLM span. Those are the consumer's words
and the model's, and they do not go.

Nothing here may take down the call it observes (commit a5fc713). Every public
method is wrapped: a dead collector, a full queue, an attribute the SDK refuses
— all of it degrades to a missing span, never to a dropped call.
"""

from __future__ import annotations

import logging
import os
import time

from opentelemetry import trace as otel
from opentelemetry.context import Context
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util.types import AttributeValue

from collector.audit.events import (
    CallEnded,
    CallStarted,
    DecisionRecorded,
    Escalated,
    GuardrailTripped,
    ModelCalled,
    ToolInvoked,
)
from collector.guardrails.disclosures import DisclosureId

logger = logging.getLogger("collector.tracing")

TRACING_ENV_VAR = "COLLECTOR_TRACING"
ENDPOINT_ENV_VAR = "OTEL_EXPORTER_OTLP_ENDPOINT"
SERVICE = "voice-debt-collector"

# Accepted values of COLLECTOR_TRACING. Anything else raises rather than
# falling back, because falling back is this repo's documented failure mode:
# `COLLECTOR_LLM=antropic` routes to Anthropic anyway and says nothing, and
# `COLLECTOR_VOICE_RECORDING` warns and continues. An observability switch that
# silently reads as "off" is worse than both — you find out there are no traces
# at the moment you go looking for the one call you needed.
_OFF = ("", "off")
_OTLP = "otlp"

_provider: TracerProvider | None = None
_tracer: otel.Tracer | None = None


class TracingConfigError(ValueError):
    """``COLLECTOR_TRACING`` was set to something this module cannot honour."""


def configure_tracing() -> TracerProvider | None:
    """Wire span export up, or leave it off. Called once at the top of a
    process's entrypoint — never lazily on the first span, so a typo takes down
    a worker at start-up rather than a call in progress.

    Off unless ``COLLECTOR_TRACING=otlp``, in which case spans go to
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` over OTLP/HTTP. That endpoint is required:
    the exporter's own default is localhost, so a mode set without one would
    build a working pipeline that black-holes every span — the same silent
    disablement the mode check above exists to prevent, in a different costume.

    Idempotent. ``voice_app.entrypoint`` runs once per job and a fresh
    ``BatchSpanProcessor`` per call would leak an exporter thread per call.
    """
    mode = os.environ.get(TRACING_ENV_VAR, "").strip().lower()
    if mode in _OFF:
        return None
    if mode != _OTLP:
        raise TracingConfigError(
            f"{TRACING_ENV_VAR}={mode!r} is not a tracing mode; use 'otlp' or leave it unset"
        )
    if _provider is not None:
        return _provider
    if not os.environ.get(ENDPOINT_ENV_VAR, "").strip():
        raise TracingConfigError(
            f"{TRACING_ENV_VAR}=otlp needs {ENDPOINT_ENV_VAR} set to the backend's OTLP"
            " endpoint; without it the exporter would post every span to localhost"
        )

    # Imported here, not at module scope: `collector.agent` imports this module
    # on every path including the text harness, and neither the OTLP exporter
    # nor the whole of `livekit.agents` should be dragged in to run a call that
    # exports nothing.
    from livekit.agents.telemetry import set_tracer_provider
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: SERVICE}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    install_tracer_provider(provider)
    # The SDK's own session spans go to the same backend. `set_tracer_provider`
    # is the Agents SDK's documented entry point for this (docs.livekit.io,
    # "Export traces") and is re-settable, unlike the global OTel one.
    set_tracer_provider(provider)
    return provider


def install_tracer_provider(provider: TracerProvider | None) -> None:
    """Point this module's spans at ``provider``, or at nothing.

    Kept separate from ``configure_tracing`` so a test can install an in-memory
    provider without an endpoint, an exporter thread, or ``livekit.agents``.
    Deliberately *not* ``opentelemetry.trace.set_tracer_provider``: that one is
    once-per-process and no-ops on the second call, so a second call would run
    against a dead provider.
    """
    global _provider, _tracer
    _provider = provider
    _tracer = None if provider is None else provider.get_tracer("collector")


def flush_traces() -> None:
    """Push anything still batched. The last thing a shutdown path does, since
    ``BatchSpanProcessor`` holds spans that a process exit would drop."""
    provider = _provider
    if provider is None:
        return
    try:
        provider.force_flush()
    except Exception:
        logger.exception("could not flush traces")


class CallTrace:
    """Every collector span for one call, under one root span.

    The root is opened by ``CallStarted`` and closed by ``CallEnded``, and every
    other span names it as parent *explicitly* rather than through the ambient
    context. That is what makes the headline requirement hold on this codebase:
    a turn runs under ``asyncio.to_thread`` on the voice path and inside a
    generator on the streaming path, and neither carries an implicit span
    context reliably enough for "the whole call is one trace" to survive it.
    """

    def __init__(self) -> None:
        self._root: otel.Span | None = None
        self._parent: Context | None = None
        self._disclosed: frozenset[DisclosureId] = frozenset()

    def record(self, event: object, *, disclosures_fired: frozenset[DisclosureId]) -> None:
        """Emit whatever span this audit event is worth, and never raise.

        ``disclosures_fired`` rides along because a disclosure *firing* is the
        one guardrail fact that is state rather than an event —
        nothing is written when the Mini-Miranda lands, the guard's ``fired``
        set simply grows. Diffing it here catches that without adding an audit
        event that the store, the JSON export and their round-trip tests would
        all have to learn.
        """
        tracer = _tracer
        if tracer is None:
            return
        try:
            self._emit(tracer, event)
            self._note_disclosures(disclosures_fired)
        except Exception:
            # The whole point of a5fc713: a tool ran, and then recording that it
            # ran ended the call. A span is worth strictly less than the call.
            logger.exception("could not record a span; the call continues untraced")

    # -- mapping -----------------------------------------------------------

    def _emit(self, tracer: otel.Tracer, event: object) -> None:
        attributes: dict[str, AttributeValue]
        # ``latency`` is None for the events that are instants rather than
        # durations — a verdict, a guardrail trip, an escalation.
        latency: int | None
        match event:
            case CallStarted():
                self._open(tracer, event)
                return
            case CallEnded():
                self._close(event)
                return
            case ModelCalled():
                name, latency, attributes = "llm_call", event.latency_ms, self._model(event)
            case ToolInvoked():
                name, latency, attributes = "tool_call", event.latency_ms, self._tool(event)
            case DecisionRecorded():
                name, latency, attributes = "decision", None, self._decision(event)
            case GuardrailTripped():
                name, latency, attributes = "guardrail_trip", None, self._guardrail(event)
            case Escalated():
                name, latency, attributes = "escalation", None, self._escalation(event)
            case _:
                # ``TurnRecorded`` lands here. It is the transcript — the
                # consumer's sentence, or the one the agent spoke back — and it
                # is exactly what does not leave the audit store.
                return
        self._child(tracer, name, attributes, latency)

    def _model(self, event: ModelCalled) -> dict[str, AttributeValue]:
        # `gen_ai.*` for the four fields the OTel GenAI convention names, so a
        # backend renders this as a model call rather than an anonymous span;
        # `collector.*` for the rest, which no convention covers.
        attributes: dict[str, AttributeValue] = {
            "gen_ai.request.model": event.model,
            "gen_ai.usage.input_tokens": event.input_tokens,
            "gen_ai.usage.output_tokens": event.output_tokens,
            "collector.cache_read_tokens": event.cache_read_tokens,
            "collector.cache_write_tokens": event.cache_write_tokens,
            "collector.latency_ms": event.latency_ms,
            "collector.turn_index": event.turn_index,
        }
        if event.stop_reason is not None:
            attributes["gen_ai.response.finish_reason"] = event.stop_reason
        if event.cost_usd is not None:
            # A string, not a float: the no-floats rule governs what leaves this process
            # as much as what lands in the log, and a span attribute has no
            # decimal type to put a Decimal in.
            attributes["collector.cost_usd"] = str(event.cost_usd)
        if event.error is not None:
            # The class, not the message. A provider's error text can quote the
            # request back, and the request is the transcript. The full string
            # is on the ``ModelCalled`` row this span points at.
            attributes["collector.error_type"] = _error_type(event.error)
        return attributes

    def _tool(self, event: ToolInvoked) -> dict[str, AttributeValue]:
        # Shape, not values. "which arguments did the model send" is what
        # diagnoses a malformed call; *what it sent* is a dollar figure or a
        # cadence for a real consumer's arrangement. Likewise the failure is
        # reported as ``ok=false`` and nothing else: ``tools.py`` builds its
        # error strings by interpolating the offending value.
        return {
            "collector.tool": event.tool,
            "collector.tool.argument_keys": sorted(event.arguments),
            "collector.tool.argument_count": len(event.arguments),
            "collector.ok": event.ok,
            "collector.latency_ms": event.latency_ms,
            "collector.turn_index": event.turn_index,
        }

    def _decision(self, event: DecisionRecorded) -> dict[str, AttributeValue]:
        # The evaluated conditions, which is the report's whole point: "if they
        # show you a transcript, the model decided; if they show you evaluated
        # conditions and a policy path, the engine did." Every rule that ran is
        # named, on the side it came down on — passing ones included, because a
        # record of only the failures cannot show what was checked.
        #
        # ``Condition.actual`` and ``Condition.limit`` are the amounts: the
        # proposed total against the settlement floor. They stay in SQLite. The
        # rule ids and the rationale code carry the reasoning; the figures are
        # one ``call_id`` lookup away for anyone entitled to them.
        verdict = event.verdict
        attributes: dict[str, AttributeValue] = {
            "collector.decision.outcome": verdict.outcome,
            "collector.decision.rationale_code": verdict.rationale_code.value,
            "collector.decision.rules_passed": [
                c.rule_id.value for c in verdict.conditions if c.passed
            ],
            "collector.decision.rules_failed": [
                c.rule_id.value for c in verdict.conditions if not c.passed
            ],
            "collector.decision.countered": verdict.counter is not None,
            "collector.turn_index": event.turn_index,
        }
        if verdict.tier is not None:
            attributes["collector.decision.tier"] = verdict.tier.name
        return attributes

    def _guardrail(self, event: GuardrailTripped) -> dict[str, AttributeValue]:
        # No ``detail`` and no ``blocked_text``. The blocked text is by
        # definition the sentence that must not be repeated, and ``detail`` is
        # prose that quotes what tripped the rule — at the escalation site it is
        # literally ``f"{trigger}: {consumer_utterance}"``.
        return {
            "collector.guardrail.ring": event.ring.value,
            # ``str()`` because the rings hand this over as their own StrEnum
            # and a bare enum member exports as its repr, not its id.
            "collector.guardrail.rule_id": str(event.rule_id),
            "collector.guardrail.action": event.action.value,
            "collector.turn_index": event.turn_index,
        }

    def _escalation(self, event: Escalated) -> dict[str, AttributeValue]:
        # ``callback_owed`` rides along because the obligation, not the
        # escalation, is what someone has to act on afterwards.
        return {
            "collector.escalation.trigger": event.trigger.value,
            "collector.escalation.callback_owed": event.callback_owed,
            "collector.turn_index": event.turn_index,
        }

    # -- span lifecycle ----------------------------------------------------

    def _open(self, tracer: otel.Tracer, event: CallStarted) -> None:
        # An empty ``Context`` rather than the ambient one: on the voice path
        # this runs inside a LiveKit job that may already have a span open, and
        # a call trace that is sometimes a root and sometimes a subtree is not
        # a thing you can query for. ``consumer_ref`` and ``original_balance``
        # are on the event and are not read — the name and the balance are the
        # two things ``voice_app`` already keeps out of its log context.
        self._root = tracer.start_span(
            "collections_call",
            context=Context(),
            attributes={
                "collector.call_id": event.call_id,
                "collector.account_ref": event.account_ref,
                "collector.channel": event.channel,
                # Langfuse groups a session by this attribute. It cannot come
                # from ``set_tracer_provider(metadata=...)``, which is
                # process-wide and would label every call in a worker with
                # whichever one started first.
                "langfuse.session.id": event.call_id,
            },
        )
        self._parent = otel.set_span_in_context(self._root)

    def _close(self, event: CallEnded) -> None:
        root = self._root
        if root is None:
            return
        root.set_attribute("collector.outcome", event.outcome.value)
        root.set_attribute("collector.turn_count", event.turn_count)
        root.set_attribute("collector.blocked_turns", event.blocked_turns)
        root.set_attribute("collector.violation_count", event.violation_count)
        if event.compliant is not None:
            # ``None`` means the call never reached the post-call ring. Omitted
            # rather than sent as false: a dropped process is not a failed call.
            root.set_attribute("collector.compliant", event.compliant)
        root.end()
        self._root = None
        self._parent = None

    def _child(
        self,
        tracer: otel.Tracer,
        name: str,
        attributes: dict[str, AttributeValue],
        latency_ms: int | None,
    ) -> None:
        # Back-dated so the span covers the work rather than the instant it was
        # recorded: the waterfall is the reason to have a trace at all when a
        # single voice turn can stack seven model round trips. ``latency_ms``
        # also goes on as a plain attribute, because the timestamps are derived
        # and the measured number should be readable as itself.
        end = time.time_ns()
        start = end if latency_ms is None else end - max(latency_ms, 0) * 1_000_000
        span = tracer.start_span(
            name, context=self._parent, start_time=start, attributes=attributes
        )
        if "collector.error_type" in attributes or attributes.get("collector.ok") is False:
            # No description: the failure's own words are the thing being kept
            # out. The status marks the span, the audit row explains it.
            span.set_status(otel.Status(otel.StatusCode.ERROR))
        span.end(end_time=end)

    def _note_disclosures(self, fired: frozenset[DisclosureId]) -> None:
        new = fired - self._disclosed
        if not new:
            return
        self._disclosed = fired
        root = self._root
        if root is None:
            return
        for disclosure in sorted(new):
            root.add_event("guardrail.disclosure_fired", {"collector.disclosure": disclosure.value})


def _error_type(error: str) -> str:
    """``"APITimeoutError: timed out"`` -> ``"APITimeoutError"``.

    The clients format failures as ``f"{type(exc).__name__}: {exc}"``, and the
    half before the colon is the half that cannot be quoting a request.
    """
    return error.split(":", 1)[0].strip() or "unknown"
