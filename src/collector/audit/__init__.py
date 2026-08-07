"""Audit and agreement log.

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
    ModelCalled,
    Speaker,
    ToolInvoked,
    TraceEvent,
    TurnRecorded,
    dumps,
    event_from_json,
    event_json,
    to_jsonable,
    utc_now,
)
from collector.audit.store import (
    DB_PATH_ENV_VAR,
    DEFAULT_DB_PATH,
    SCHEMA,
    AuditStore,
    CallCompliance,
    default_db_path,
)

# Re-exported from the modules that own them, so the log and the runtime cannot
# drift apart: escalation triggers are whatever the guardrails detect, and a
# call outcome is whatever the negotiation reached.
from collector.guardrails.rings import EscalationTrigger, GuardrailRing
from collector.negotiation import CallOutcome

__all__ = [
    "DB_PATH_ENV_VAR",
    "DEFAULT_DB_PATH",
    "SCHEMA",
    "AgreementRecord",
    "AuditStore",
    "CallCompliance",
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
    "ModelCalled",
    "Speaker",
    "ToolInvoked",
    "TraceEvent",
    "TurnRecorded",
    "default_db_path",
    "dumps",
    "event_from_json",
    "event_json",
    "to_jsonable",
    "utc_now",
]
