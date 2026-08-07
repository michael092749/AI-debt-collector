"""Compliance guardrails. Three rings, in code, never in the prompt.

The brief's automatic-fail condition is persuasion by threats, false urgency or
invented consequences, so none of this lives in prompt prose where a jailbreak
can argue with it. Every rule here is a Python predicate with a stable id that
lands in the audit log.

``agent.py`` needs only the ring entry points:

    from collector.guardrails import (
        GuardrailState, check_pre_call, check_inbound, check_outbound, finalize_call,
    )
"""

from collector.guardrails.confirmation import (
    ConfirmationRuleId,
    confirmation_line,
    repeats_back,
)
from collector.guardrails.disclosures import (
    AI_DISCLOSURE_TEXT,
    AI_DISCLOSURE_WITH_HUMAN,
    HUMAN_AVAILABLE_TEXT,
    MINI_MIRANDA_TEXT,
    DisclosureId,
    DisclosureRuleId,
    DisclosureState,
    confirms_identity,
    fires_ai_disclosure,
    fires_mini_miranda,
    is_substantive,
    requests_ai_disclosure,
)
from collector.guardrails.numeric import (
    AuthorizedFigures,
    DurationUnit,
    Figure,
    FigureKind,
    NumericRuleId,
    authorized_for,
    check_numeric,
    extract_figures,
)
from collector.guardrails.prohibited import (
    ProhibitedRuleId,
    Severity,
    Violation,
    is_prohibited,
    scan_prohibited,
)
from collector.guardrails.rings import (
    CONNECTIVE_TEXT,
    MAX_REGENERATION_STRIKES,
    SAFE_FALLBACK_TEXT,
    CallSummary,
    EscalationRecord,
    EscalationSignal,
    EscalationTrigger,
    GuardrailEvent,
    GuardrailRing,
    GuardrailState,
    InboundCheck,
    OutboundCheck,
    PreCallCheck,
    PreCallContext,
    RingRuleId,
    check_inbound,
    check_outbound,
    check_pre_call,
    detect_escalation,
    escalation_closing,
    finalize_call,
)

__all__ = [
    "AI_DISCLOSURE_TEXT",
    "AI_DISCLOSURE_WITH_HUMAN",
    "CONNECTIVE_TEXT",
    "HUMAN_AVAILABLE_TEXT",
    "MAX_REGENERATION_STRIKES",
    "MINI_MIRANDA_TEXT",
    "SAFE_FALLBACK_TEXT",
    "AuthorizedFigures",
    "CallSummary",
    "ConfirmationRuleId",
    "DisclosureId",
    "DisclosureRuleId",
    "DisclosureState",
    "DurationUnit",
    "EscalationRecord",
    "EscalationSignal",
    "EscalationTrigger",
    "Figure",
    "FigureKind",
    "GuardrailEvent",
    "GuardrailState",
    "InboundCheck",
    "NumericRuleId",
    "OutboundCheck",
    "PreCallCheck",
    "PreCallContext",
    "ProhibitedRuleId",
    "GuardrailRing",
    "RingRuleId",
    "Severity",
    "Violation",
    "authorized_for",
    "check_inbound",
    "check_numeric",
    "check_outbound",
    "check_pre_call",
    "confirmation_line",
    "detect_escalation",
    "escalation_closing",
    "extract_figures",
    "finalize_call",
    "confirms_identity",
    "fires_ai_disclosure",
    "fires_mini_miranda",
    "is_prohibited",
    "is_substantive",
    "repeats_back",
    "requests_ai_disclosure",
    "scan_prohibited",
]
