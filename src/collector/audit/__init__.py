"""Audit and agreement log — SPEC §6.

The agent loop only needs two things from here: ``AuditStore.record(event)``
during the call, and ``AuditStore.finalize_agreement(...)`` at the end of it.
"""

from __future__ import annotations

from collector.audit.events import (
    AgreementRecord,
    CallEnded,
    CallStarted,
    ConsumerConfirmation,
    DecisionRecorded,
    Escalated,
    EventType,
    GuardrailAction,
    GuardrailTripped,
    Speaker,
    TraceEvent,
    TurnRecorded,
    dumps,
    event_from_json,
    event_json,
    to_jsonable,
    utc_now,
)
from collector.audit.store import DEFAULT_DB_PATH, SCHEMA, AuditStore

# Re-exported from the modules that own them, so the log and the runtime cannot
# drift apart: escalation triggers are whatever the guardrails detect, and a
# call outcome is whatever the negotiation reached.
from collector.guardrails.rings import EscalationTrigger, GuardrailRing
from collector.negotiation import CallOutcome

__all__ = [
    "DEFAULT_DB_PATH",
    "SCHEMA",
    "AgreementRecord",
    "AuditStore",
    "CallEnded",
    "CallOutcome",
    "CallStarted",
    "ConsumerConfirmation",
    "DecisionRecorded",
    "Escalated",
    "EscalationTrigger",
    "EventType",
    "GuardrailAction",
    "GuardrailRing",
    "GuardrailTripped",
    "Speaker",
    "TraceEvent",
    "TurnRecorded",
    "dumps",
    "event_from_json",
    "event_json",
    "to_jsonable",
    "utc_now",
]
