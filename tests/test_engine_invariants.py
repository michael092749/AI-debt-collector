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


# Nothing in the engine may know the brief's $1,000 or its 25%: both are read
# off the policy, so the rules are swept against balances and floors that are
# nothing like them. The odd cents are deliberate - 437.19 divides badly, and a
# 34% floor caps a plan at two payments rather than four.
BALANCES = ["437.19", "1000", "2500", "17500.55"]
FLOOR_PCTS = [Decimal("0.10"), Decimal("0.25"), Decimal("0.34")]
CAPACITY_FRACTIONS = [
    Decimal(f) for f in ("0.05", "0.26", "0.34", "0.4", "0.5", "0.75", "1", "1.5")
]


def _payments_needed(total: Money, capacity: Money) -> int:
    """Fewest payments of at most ``capacity`` that cover ``total``.

    Stated here independently of the engine: a test that imports the engine's
    own arithmetic proves only that it agrees with itself.
    """
    whole, remainder = divmod(total.amount, capacity.amount)
    return int(whole) + (1 if remainder else 0)


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


def _assert_legal_under(offer: Offer, policy: PolicyConfig, label: str) -> None:
    assert offer.smallest_payment >= policy.min_payment, f"{label}: sub-floor instalment"
    assert offer.total >= policy.settlement_floor, f"{label}: below settlement floor"
    # The ceiling, not just the floor. SPEC §4.3 reads "== ORIGINAL_BALANCE
    # unless tier is SETTLEMENT", and this sweep only ever checked the lower
    # half — so it passed 320 cases while the engine was accepting $3,000 on a
    # $1,000 debt. An invariant asserted in one direction is half an invariant.
    # Parameterized by `policy` rather than the module default, so the
    # varying-policy sweep is covered by the ceiling too.
    assert offer.total <= policy.original_balance, f"{label}: collects more than is owed"
    assert offer.payment_count <= policy.max_payments_for(offer.tier), f"{label}: too many"
    assert offer.payment_count <= policy.max_installments, f"{label}: over global cap"
    assert offer.duration_days <= policy.max_plan_days, f"{label}: schedule too long"
    assert offer.cadence in policy.allowed_cadences, f"{label}: cadence not offered"
    assert sum((i.amount for i in offer.installments), Money.zero()) == offer.total, (
        f"{label}: instalments do not sum to total"
    )
    if offer.tier is not Tier.SETTLEMENT:
        assert offer.total == policy.original_balance, f"{label}: unauthorised discount"


def _assert_offer_legal(offer: Offer, label: str) -> None:
    _assert_legal_under(offer, POLICY, label)


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

    def test_no_counter_ever_asks_for_more_per_payment_than_the_consumer_signalled(
        self,
    ) -> None:
        """The rule behind the $400 -> $400/$600 bug, over arbitrary policies.

        Scoped to the tier on the table. A tier caps its instalment count, so a
        capacity funds it only if the whole total fits in that many payments;
        where it does, the engine must find such a schedule and never counter
        with an instalment above the figure they just named. Where it does not,
        the tier is simply out of their reach and refusing it is what earns the
        step down — the engine does not take that step for them, which is what
        it used to do.

        It holds at any balance and any floor percentage, because nothing in the
        engine knows the brief's $1,000: every bound is read off the policy.
        """
        for balance, pct in itertools.product(BALANCES, FLOOR_PCTS):
            policy = POLICY.replace(original_balance=Money(balance), min_payment_pct=pct)
            for tier, fraction in itertools.product(Tier, CAPACITY_FRACTIONS):
                state = NegotiationState.opening(policy).conceded_to(tier)
                capacity = policy.original_balance * fraction
                label = f"balance={balance} pct={pct} tier={tier.name} cap={capacity}"
                offer = build_counter(state, policy, capacity=capacity)

                _assert_legal_under(offer, policy, label)
                assert offer.tier is tier, f"{label}: counter left the tier on the table"
                if tier is not Tier.SETTLEMENT:
                    assert offer.total == policy.original_balance, (
                        f"{label}: discounted without a concession on the ladder"
                    )
                # A schedule at this capacity exists only if it fits between two
                # bounds: enough instalments to cover the total at ``capacity``,
                # and few enough that each still clears the minimum payment.
                needed = _payments_needed(offer.total, capacity)
                room = int(offer.total.amount / policy.min_payment.amount)
                if needed <= min(policy.max_payments_for(tier), room):
                    assert max(i.amount for i in offer.installments) <= capacity, (
                        f"{label}: instalment above capacity when a legal schedule existed"
                    )

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
