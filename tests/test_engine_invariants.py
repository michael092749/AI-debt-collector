"""T6 - invariants and purity, enforced structurally.

The tests in test_decision_engine.py check examples. These check *rules*: they
sweep the input space and assert SPEC §4.3 holds everywhere, and they assert the
engine's purity via AST inspection so it cannot quietly acquire an I/O import
six months from now.
"""

from __future__ import annotations

import ast
import itertools
from decimal import Decimal
from pathlib import Path

import pytest

from collector.decision_engine import Verdict, build_counter, validate_offer
from collector.money import Money
from collector.negotiation import NegotiationState
from collector.offers import Cadence, ConsumerProposal, Offer, Tier
from collector.policy import PolicyConfig

POLICY = PolicyConfig.default()

# A deterministic sweep rather than a random one: reproducible failures matter
# more here than input novelty, and it keeps the dependency list at zero.
TOTALS = ["0.01", "50", "200", "500", "799.99", "800", "850", "900", "1000", "1500"]
COUNTS = [1, 2, 3, 4, 5, 13, 26]
CADENCES = list(Cadence)
CAPACITIES = [None, "25", "77", "100", "250", "300", "400", "750", "900", "1000"]


def _proposals() -> list[ConsumerProposal]:
    return [
        ConsumerProposal(
            total=Money(t),
            payment_count=n,
            cadence=c,
            signaled_capacity=Money(cap) if cap else None,
        )
        for t, n, c, cap in itertools.product(TOTALS, COUNTS, CADENCES, CAPACITIES)
    ]


def _assert_offer_legal(offer: Offer, label: str) -> None:
    assert offer.smallest_payment >= POLICY.min_payment, f"{label}: sub-floor instalment"
    assert offer.total >= POLICY.settlement_floor, f"{label}: below settlement floor"
    assert offer.payment_count <= POLICY.max_payments_for(offer.tier), f"{label}: too many"
    assert offer.payment_count <= POLICY.max_installments, f"{label}: over global cap"
    assert offer.duration_days <= POLICY.max_plan_days, f"{label}: schedule too long"
    assert offer.cadence in POLICY.allowed_cadences, f"{label}: cadence not offered"
    assert sum((i.amount for i in offer.installments), Money.zero()) == offer.total, (
        f"{label}: instalments do not sum to total"
    )
    if offer.tier is not Tier.SETTLEMENT:
        assert offer.total == POLICY.original_balance, f"{label}: unauthorised discount"


class TestInvariantsHoldEverywhere:
    def test_no_verdict_ever_authorises_something_illegal(self) -> None:
        """SPEC §4.3 over the whole sweep - ~4,900 proposals."""
        state = NegotiationState.opening(POLICY)
        for p in _proposals():
            verdict = validate_offer(p, state, POLICY)
            label = f"{p.total}/{p.payment_count}/{p.cadence.value}/cap={p.signaled_capacity}"

            if verdict.outcome == "accept":
                assert p.smallest_payment >= POLICY.min_payment, label
                assert p.total >= POLICY.settlement_floor, label
                assert p.payment_count <= POLICY.max_payments_for(verdict.tier or Tier.PAY_IN_FULL)
                assert p.duration_days <= POLICY.max_plan_days, label
            else:
                assert verdict.counter is not None, f"{label}: refusal with no counter"
                _assert_offer_legal(verdict.counter, label)

    def test_every_verdict_carries_a_full_condition_trail(self) -> None:
        """The decision record must show the policy path, not just the outcome."""
        state = NegotiationState.opening(POLICY)
        for p in _proposals()[::37]:
            verdict = validate_offer(p, state, POLICY)
            assert len(verdict.conditions) >= 6
            assert len({c.rule_id for c in verdict.conditions}) == len(verdict.conditions)

    def test_build_counter_is_total_over_every_tier_and_capacity(self) -> None:
        state = NegotiationState.opening(POLICY)
        for tier, cap in itertools.product(Tier, CAPACITIES):
            offer = build_counter(state, POLICY, tier=tier, capacity=Money(cap) if cap else None)
            _assert_offer_legal(offer, f"tier={tier.name} cap={cap}")

    def test_ladder_never_moves_backwards(self) -> None:
        """A4: once conceded to a tier, the engine never counters above it."""
        state = NegotiationState.opening(POLICY).conceded_to(Tier.SETTLEMENT)
        for cap in CAPACITIES:
            offer = build_counter(state, POLICY, capacity=Money(cap) if cap else None)
            assert offer.tier >= Tier.SETTLEMENT


class TestDeterminism:
    def test_identical_inputs_give_identical_verdicts(self) -> None:
        state = NegotiationState.opening(POLICY)
        for p in _proposals()[::53]:
            runs = {validate_offer(p, state, POLICY) for _ in range(5)}
            assert len(runs) == 1, "engine is not deterministic"

    def test_verdicts_are_hashable_frozen_values(self) -> None:
        state = NegotiationState.opening(POLICY)
        v = validate_offer(ConsumerProposal(Money("1000"), 1, Cadence.IMMEDIATE), state, POLICY)
        assert isinstance(hash(v), int)
        with pytest.raises(AttributeError):
            v.tier = Tier.SETTLEMENT  # type: ignore[misc]


class TestNoFloats:
    def test_every_monetary_value_is_decimal_backed(self) -> None:
        state = NegotiationState.opening(POLICY)
        for p in _proposals()[::29]:
            verdict = validate_offer(p, state, POLICY)
            if verdict.counter:
                for inst in verdict.counter.installments:
                    assert isinstance(inst.amount.amount, Decimal)
                    assert not isinstance(inst.amount.amount, float)

    def test_money_refuses_float_construction_and_scaling(self) -> None:
        with pytest.raises(TypeError):
            Money(1.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            Money("10") * 1.5  # type: ignore[operator]


# --------------------------------------------------------------------------
# Purity, enforced by AST inspection rather than by code-review convention.
# "The LLM talks, deterministic code decides" is the whole architecture; a
# convention will not hold it.
# --------------------------------------------------------------------------

CORE_MODULES = ["money", "offers", "policy", "negotiation", "decision_engine"]

FORBIDDEN_IMPORTS = {
    "llm",
    "agent",
    "audit",
    "guardrails",
    "voice_app",
    "text_app",
    "tools",
    "sqlite3",
    "random",
    "time",
    "datetime",
    "os",
    "sys",
    "socket",
    "pathlib",
    "requests",
    "httpx",
    "anthropic",
    "openai",
    "livekit",
    "logging",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            found.add(parts[-1] if parts[0] == "collector" else parts[0])
    return found


class TestEnginePurity:
    @pytest.mark.parametrize("module", CORE_MODULES)
    def test_core_modules_import_nothing_impure(self, module: str) -> None:
        src = Path(__file__).parent.parent / "src" / "collector" / f"{module}.py"
        offenders = _imported_modules(src) & FORBIDDEN_IMPORTS
        assert not offenders, f"{module}.py imports impure modules: {sorted(offenders)}"

    def test_engine_has_no_clock_or_randomness_calls(self) -> None:
        """The clock is injected as day offsets; the engine never reads one."""
        engine = Path(__file__).parent.parent / "src" / "collector" / "decision_engine.py"
        src = engine.read_text()
        for banned in ("now(", "today(", "random", "uuid", "open(", "input("):
            assert banned not in src, f"decision_engine.py must not call {banned}"

    def test_engine_exposes_the_documented_entry_point(self) -> None:
        assert callable(validate_offer)
        assert Verdict.__dataclass_fields__.keys() >= {
            "outcome",
            "tier",
            "conditions",
            "counter",
            "rationale_code",
        }


class TestSharedVocabulary:
    """One definition per concept, across module boundaries.

    Guardrails, negotiation, and audit were built independently and each
    grew its own ``CallOutcome``/``EscalationTrigger``/``GuardrailRing``.
    Two of the three overlapped closely enough to compare unequal in silence
    rather than fail — a call that ended in agreement would have logged as if
    it had not. Identity, not just structural equality, is the assertion.
    """

    def test_call_outcome_is_defined_once(self) -> None:
        from collector import audit
        from collector.negotiation import CallOutcome

        assert audit.CallOutcome is CallOutcome

    def test_escalation_trigger_is_defined_once(self) -> None:
        from collector import audit
        from collector.guardrails import EscalationTrigger

        assert audit.EscalationTrigger is EscalationTrigger

    def test_guardrail_ring_is_defined_once(self) -> None:
        from collector import audit
        from collector.guardrails import GuardrailRing

        assert audit.GuardrailRing is GuardrailRing

    def test_every_negotiation_outcome_is_loggable(self) -> None:
        """The audit layer must be able to record any state the call can reach."""
        from collector.audit.events import CallEnded
        from collector.negotiation import CallOutcome

        for outcome in CallOutcome:
            event = CallEnded("call-1", outcome, turn_count=1, at="t1")
            assert event.outcome is outcome
