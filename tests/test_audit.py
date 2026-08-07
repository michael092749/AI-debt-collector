"""Audit log tests — SPEC §6, build step 5.

The agreement record is the brief's deliverable ("log the final agreement"), so
these tests hold it to the vendor test from the research report: the record has
to show evaluated conditions and a policy path, survive a round trip through
SQLite with its Decimals intact, and be readable as JSON without a database
client. Everything runs offline against a tmp_path database; the real
``data/collector.db`` is never touched.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from collector.audit import (
    AgreementRecord,
    AuditStore,
    CallEnded,
    CallOutcome,
    CallStarted,
    ConsumerConfirmation,
    DecisionRecorded,
    Escalated,
    EscalationTrigger,
    GuardrailAction,
    GuardrailRing,
    GuardrailTripped,
    Speaker,
    TurnRecorded,
    to_jsonable,
)
from collector.decision_engine import RuleId, Verdict, validate_offer
from collector.money import Money
from collector.negotiation import NegotiationState
from collector.offers import Cadence, ConsumerProposal, Installment, Offer, Tier
from collector.policy import PolicyConfig

CALL_ID = "CALL-0001"

# The settlement the engine authorizes: $800 over three monthly payments,
# structured 250/250/300 because the $250 floor binds harder than the discount
# (SPEC §2.3 rule 3).
SETTLEMENT = Offer(
    tier=Tier.SETTLEMENT,
    installments=(
        Installment(Money("250.00"), 0),
        Installment(Money("250.00"), 30),
        Installment(Money("300.00"), 60),
    ),
    cadence=Cadence.MONTHLY,
)


@pytest.fixture
def policy() -> PolicyConfig:
    return PolicyConfig.default()


@pytest.fixture
def store(tmp_path: Path) -> AuditStore:
    with AuditStore(tmp_path / "collector.db", json_dir=tmp_path / "agreements") as s:
        yield s


@pytest.fixture
def lowball(policy: PolicyConfig) -> Verdict:
    """$200 up front — rejected below the $800 floor."""
    proposal = ConsumerProposal(Money("200"), 1, Cadence.IMMEDIATE)
    return validate_offer(proposal, NegotiationState.opening(policy), policy)


@pytest.fixture
def accepted(policy: PolicyConfig) -> Verdict:
    """$800 over three monthly payments — accepted at T3.

    Read against a call that has already conceded its way down to settlement:
    the same terms proposed cold are countered, not taken, because the ladder
    has not reached them yet (A7).
    """
    proposal = ConsumerProposal(Money("800"), 3, Cadence.MONTHLY)
    state = NegotiationState.opening(policy).conceded_to(Tier.SETTLEMENT)
    return validate_offer(proposal, state, policy)


@pytest.fixture
def confirmation() -> ConsumerConfirmation:
    return ConsumerConfirmation(
        confirmed=True,
        utterance="Yeah, that works. Two fifty now and the rest monthly.",
        turn_index=8,
        at="2026-01-01T12:05:00+00:00",
    )


def record_call(store: AuditStore, lowball: Verdict, accepted: Verdict) -> None:
    """A whole call: two proposals, a blocked utterance, and a dispute noted."""
    store.record(
        CallStarted(
            call_id=CALL_ID,
            account_ref="ACCT-77",
            consumer_ref="CONS-77",
            original_balance=Money("1000.00"),
            channel="text",
            at="2026-01-01T12:00:00+00:00",
        )
    )
    store.record(TurnRecorded(CALL_ID, 0, Speaker.AGENT, "This is a call about a debt.", at="t0"))
    store.record(TurnRecorded(CALL_ID, 1, Speaker.CONSUMER, "I can do two hundred.", at="t1"))
    store.record(
        DecisionRecorded(
            CALL_ID,
            1,
            ConsumerProposal(Money("200"), 1, Cadence.IMMEDIATE),
            lowball,
            at="t1d",
        )
    )
    store.record(
        GuardrailTripped(
            call_id=CALL_ID,
            turn_index=2,
            ring=GuardrailRing.DURING_CALL,
            rule_id="PROHIBITED_THREAT",
            action=GuardrailAction.BLOCKED,
            detail="threat of legal action",
            blocked_text="We'll have to take you to court.",
            at="t2",
        )
    )
    store.record(
        Escalated(
            call_id=CALL_ID,
            turn_index=3,
            trigger=EscalationTrigger.DISPUTE,
            detail="consumer questioned the balance",
            at="t3",
        )
    )
    store.record(
        TurnRecorded(CALL_ID, 4, Speaker.CONSUMER, "Eight hundred over three months?", at="t4")
    )
    store.record(
        DecisionRecorded(
            CALL_ID,
            4,
            ConsumerProposal(Money("800"), 3, Cadence.MONTHLY),
            accepted,
            at="t4d",
        )
    )


@pytest.fixture
def agreement(
    store: AuditStore, lowball: Verdict, accepted: Verdict, confirmation: ConsumerConfirmation
) -> AgreementRecord:
    record_call(store, lowball, accepted)
    record = store.finalize_agreement(
        call_id=CALL_ID,
        final_offer=SETTLEMENT,
        authorizing_verdict=accepted,
        confirmation=confirmation,
        at="2026-01-01T12:06:00+00:00",
    )
    store.record(CallEnded(CALL_ID, CallOutcome.AGREED, turn_count=9, at="t9"))
    return record


def floats_in(value: Any, path: str = "$") -> list[str]:
    """Every path in a parsed JSON tree that holds a float. Must always be empty."""
    if isinstance(value, float):
        return [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in floats_in(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in floats_in(v, f"{path}[{i}]")]
    return []


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


class TestSchema:
    def test_creates_the_four_tables(self, store: AuditStore) -> None:
        assert store.table_names() == ("agreements", "calls", "decisions", "turns")

    def test_creation_is_idempotent(self, tmp_path: Path, accepted: Verdict) -> None:
        """Reopening an existing database must not error and must not truncate."""
        path = tmp_path / "collector.db"
        with AuditStore(path, json_dir=tmp_path / "j") as first:
            first.record(
                TurnRecorded(CALL_ID, 0, Speaker.AGENT, "hello", at="t0"),
            )

        with AuditStore(path, json_dir=tmp_path / "j") as second:
            second.create_schema()  # explicit second run, same result
            second.create_schema()
            assert second.table_names() == ("agreements", "calls", "decisions", "turns")
            assert len(second.turns(CALL_ID)) == 1

    def test_uses_the_configured_path_only(self, tmp_path: Path) -> None:
        """Tests must never write to the real data/ directory (SPEC §10)."""
        with AuditStore(tmp_path / "sub" / "collector.db") as s:
            assert s.db_path == tmp_path / "sub" / "collector.db"
            assert s.json_dir == tmp_path / "sub" / "agreements"
            assert s.db_path.exists()

    def test_money_columns_are_text_never_real(self, store: AuditStore) -> None:
        """SQLite REAL is a float. Money lives in TEXT columns or it is a bug."""
        rows = store._rows("PRAGMA table_info(agreements)", ())
        types = {r["name"]: r["type"] for r in rows}
        assert types["total"] == "TEXT"
        assert "REAL" not in set(types.values())


# --------------------------------------------------------------------------
# the trace
# --------------------------------------------------------------------------


class TestTrace:
    def test_round_trips_every_event_kind(
        self, store: AuditStore, lowball: Verdict, accepted: Verdict
    ) -> None:
        record_call(store, lowball, accepted)

        assert len(store.turns(CALL_ID)) == 3
        assert len(store.decisions(CALL_ID)) == 2
        assert len(store.guardrail_events(CALL_ID)) == 1
        assert len(store.escalations(CALL_ID)) == 1
        # The timeline interleaves utterances, trips, and escalations in order.
        assert [e.EVENT_TYPE.value for e in store.trace(CALL_ID)] == [
            "turn",
            "turn",
            "guardrail_trip",
            "escalation",
            "turn",
        ]

    def test_decisions_survive_intact(
        self, store: AuditStore, lowball: Verdict, accepted: Verdict
    ) -> None:
        record_call(store, lowball, accepted)
        first, second = store.decisions(CALL_ID)

        assert first.verdict == lowball
        assert second.verdict == accepted
        assert first.proposal.total == Money("200.00")

    def test_blocked_text_is_retained_not_discarded(
        self, store: AuditStore, lowball: Verdict, accepted: Verdict
    ) -> None:
        """A blocked utterance is evidence the guardrail worked; keeping it is
        the point of the log (SPEC §5.3)."""
        record_call(store, lowball, accepted)
        (trip,) = store.guardrail_events(CALL_ID)

        assert trip.action is GuardrailAction.BLOCKED
        assert trip.blocked_text == "We'll have to take you to court."

    def test_call_lifecycle_columns_are_filled(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        del agreement
        row = store._rows("SELECT * FROM calls WHERE call_id = ?", (CALL_ID,))[0]

        assert row["original_balance"] == "1000.00"
        assert row["outcome"] == CallOutcome.AGREED.value
        assert row["turn_count"] == 9
        assert row["started_at"] == "2026-01-01T12:00:00+00:00"
        assert row["ended_at"] == "t9"

    def test_rejects_a_non_event(self, store: AuditStore) -> None:
        with pytest.raises(TypeError):
            store.record("agreed to pay")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the agreement record — the deliverable
# --------------------------------------------------------------------------


class TestAgreementRecord:
    def test_carries_tier_total_and_full_schedule(self, agreement: AgreementRecord) -> None:
        assert agreement.tier is Tier.SETTLEMENT
        assert agreement.total == Money("800.00")
        assert agreement.cadence is Cadence.MONTHLY
        assert [(str(i.amount.amount), i.due_day_offset) for i in agreement.schedule] == [
            ("250.00", 0),
            ("250.00", 30),
            ("300.00", 60),
        ]

    def test_carries_the_authorizing_condition_trail(
        self, agreement: AgreementRecord, accepted: Verdict
    ) -> None:
        """The vendor test: evaluated conditions and a policy path, not a
        transcript. Every rule the engine checked is here, passing ones too."""
        assert agreement.conditions == accepted.conditions
        assert {c.rule_id for c in agreement.conditions} == set(RuleId)
        assert all(c.passed for c in agreement.conditions)
        for c in agreement.conditions:
            assert c.actual and c.limit

    def test_carries_every_counter_offer_exchanged(self, agreement: AgreementRecord) -> None:
        assert [e.verdict.outcome for e in agreement.exchanges] == ["reject", "accept"]
        rejected = agreement.exchanges[0].verdict
        failed = [c.rule_id for c in rejected.conditions if not c.passed]
        assert RuleId.TOTAL_FLOOR in failed
        # The rejection's full trail rides along, not just its outcome.
        assert len(rejected.conditions) == len(RuleId)

    def test_carries_guardrail_events_and_escalations(self, agreement: AgreementRecord) -> None:
        assert [g.rule_id for g in agreement.guardrail_events] == ["PROHIBITED_THREAT"]
        assert [e.trigger for e in agreement.escalations] == [EscalationTrigger.DISPUTE]

    def test_carries_consumer_confirmation(self, agreement: AgreementRecord) -> None:
        assert agreement.confirmation.confirmed is True
        assert agreement.confirmation.turn_index == 8

    def test_refuses_an_unconfirmed_arrangement(self, accepted: Verdict, store: AuditStore) -> None:
        with pytest.raises(ValueError, match="unconfirmed"):
            store.finalize_agreement(
                call_id=CALL_ID,
                final_offer=SETTLEMENT,
                authorizing_verdict=accepted,
                confirmation=ConsumerConfirmation(False, "no", 8, at="t8"),
            )

    def test_refuses_authorization_by_a_rejection(
        self, lowball: Verdict, store: AuditStore, confirmation: ConsumerConfirmation
    ) -> None:
        """No path exists from "rejected" to "agreed"."""
        with pytest.raises(ValueError, match="rejected"):
            store.finalize_agreement(
                call_id=CALL_ID,
                final_offer=SETTLEMENT,
                authorizing_verdict=lowball,
                confirmation=confirmation,
            )

    def test_only_records_exchanges_that_were_logged(
        self, store: AuditStore, accepted: Verdict, confirmation: ConsumerConfirmation
    ) -> None:
        record = store.finalize_agreement(
            call_id="CALL-EMPTY",
            final_offer=SETTLEMENT,
            authorizing_verdict=accepted,
            confirmation=confirmation,
        )
        assert record.exchanges == ()
        assert record.guardrail_events == ()


# --------------------------------------------------------------------------
# serialization — no floats, exact Decimals
# --------------------------------------------------------------------------


class TestSerialization:
    def test_money_is_an_exact_decimal_string(self) -> None:
        assert to_jsonable(Money("800")) == "800.00"
        assert to_jsonable(Money("0.1") + Money("0.2")) == "0.30"

    def test_tier_serializes_by_name_not_ordinal(self) -> None:
        """Tier is an IntEnum; a log that says 3 where it means SETTLEMENT is
        not auditable."""
        assert to_jsonable(Tier.SETTLEMENT) == "SETTLEMENT"

    def test_refuses_to_serialize_a_float(self) -> None:
        with pytest.raises(TypeError, match="float"):
            to_jsonable({"total": 800.0})

    def test_no_float_anywhere_in_the_record(self, agreement: AgreementRecord) -> None:
        assert floats_in(agreement.to_json_dict()) == []
        assert floats_in(json.loads(agreement.to_json())) == []

    def test_no_float_anywhere_in_the_database(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        del agreement
        for table in store.table_names():
            for row in store._rows(f"SELECT * FROM {table}", ()):
                for key, value in dict(row).items():
                    assert not isinstance(value, float), f"{table}.{key} is a float"
                    if isinstance(value, str) and value.startswith("{"):
                        assert floats_in(json.loads(value)) == []

    def test_counter_offers_carry_a_readable_total(self, accepted: Verdict) -> None:
        offer = to_jsonable(SETTLEMENT)
        assert offer["total"] == "800.00"
        assert offer["payment_count"] == 3
        assert offer["tier"] == "SETTLEMENT"
        del accepted


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


class TestRoundTrip:
    def test_querying_the_agreement_back_reconstructs_it(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        loaded = store.agreement(CALL_ID)

        assert loaded == agreement

    def test_amounts_come_back_as_exact_decimals(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        del agreement
        loaded = store.agreement(CALL_ID)
        assert loaded is not None

        assert isinstance(loaded.total.amount, Decimal)
        assert loaded.total.amount == Decimal("800.00")
        assert str(loaded.total.amount) == "800.00"
        assert sum((i.amount for i in loaded.schedule), Money.zero()) == loaded.total
        for installment in loaded.schedule:
            assert isinstance(installment.amount.amount, Decimal)

    def test_condition_trail_survives_the_round_trip(
        self, store: AuditStore, accepted: Verdict, agreement: AgreementRecord
    ) -> None:
        del agreement
        loaded = store.agreement(CALL_ID)
        assert loaded is not None

        assert loaded.conditions == accepted.conditions
        assert loaded.exchanges[0].verdict.conditions[0].limit.startswith(">=")

    def test_listing_agreements_returns_full_records(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        assert store.agreements() == (agreement,)

    def test_missing_agreement_is_none(self, store: AuditStore) -> None:
        assert store.agreement("CALL-NOPE") is None

    def test_agreements_are_queryable_by_scalar_columns(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        """Production sampling (SPEC §7.3) needs to filter without parsing JSON."""
        row = store._rows(
            "SELECT * FROM agreements WHERE tier = ? AND total >= ?",
            (Tier.SETTLEMENT.name, "800.00"),
        )[0]

        assert row["agreement_id"] == agreement.agreement_id
        assert row["payment_count"] == 3
        assert row["final_day_offset"] == 60
        assert row["confirmed"] == 1

    def test_survives_a_reopen(
        self, tmp_path: Path, accepted: Verdict, confirmation: ConsumerConfirmation
    ) -> None:
        db = tmp_path / "collector.db"
        with AuditStore(db, json_dir=tmp_path / "agreements") as first:
            record = first.finalize_agreement(
                call_id=CALL_ID,
                final_offer=SETTLEMENT,
                authorizing_verdict=accepted,
                confirmation=confirmation,
                at="2026-01-01T12:06:00+00:00",
            )

        with AuditStore(db, json_dir=tmp_path / "agreements") as second:
            assert second.agreement(CALL_ID) == record


# --------------------------------------------------------------------------
# standalone JSON export
# --------------------------------------------------------------------------


class TestJsonExport:
    def test_writes_valid_json_next_to_the_database(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        path = store.agreement_json_path(agreement.agreement_id)

        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["call_id"] == CALL_ID

    def test_export_is_complete(self, store: AuditStore, agreement: AgreementRecord) -> None:
        """Inspectable with no SQLite client: everything a reviewer needs is in
        this one file (SPEC §6)."""
        data = json.loads(store.agreement_json_path(agreement.agreement_id).read_text())

        assert data["tier"] == "SETTLEMENT"
        assert data["total"] == "800.00"
        assert data["cadence"] == "monthly"
        assert data["schedule"] == [
            {"amount": "250.00", "due_day_offset": 0},
            {"amount": "250.00", "due_day_offset": 30},
            {"amount": "300.00", "due_day_offset": 60},
        ]
        assert [c["rule_id"] for c in data["conditions"]] == [r.value for r in RuleId]
        assert len(data["exchanges"]) == 2
        assert data["exchanges"][0]["verdict"]["rationale_code"] == "BELOW_SETTLEMENT_FLOOR"
        assert len(data["guardrail_events"]) == 1
        assert len(data["escalations"]) == 1
        assert data["confirmation"]["confirmed"] is True
        assert data["rationale_code"] == "ACCEPTED"

    def test_export_parses_back_into_the_record(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        raw = store.agreement_json_path(agreement.agreement_id).read_text()

        assert AgreementRecord.from_json(raw) == agreement


def test_default_db_path_is_under_data(tmp_path: Path) -> None:
    """SPEC §10: never write outside ./data. Asserted on the constant, not by
    opening it — this suite must not create the real database."""
    from collector.audit import DEFAULT_DB_PATH

    assert DEFAULT_DB_PATH.as_posix() == "data/collector.db"
    del tmp_path


def test_schema_declares_no_real_columns() -> None:
    """A REAL column is a float column. Money never lives in one."""
    from collector.audit import SCHEMA

    declarations = [line.split("--")[0] for line in SCHEMA.splitlines()]
    assert not [line for line in declarations if "REAL" in line]


def test_store_is_usable_without_a_context_manager(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "c.db")
    try:
        store.record(TurnRecorded(CALL_ID, 0, Speaker.AGENT, "hi", at="t0"))
        assert store.turns(CALL_ID)[0].text == "hi"
    finally:
        store.close()
    # close() shuts down the store's dedicated worker thread (see
    # test_audit_store_threading.py), so a post-close call now fails when
    # `_run` tries to submit to that dead executor rather than when sqlite3
    # notices the connection is closed.
    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        store.turns(CALL_ID)
