"""Audit log tests.

The agreement record is the brief's deliverable ("log the final agreement"), so
these tests hold it to the vendor test from the research report: the record has
to show evaluated conditions and a policy path, survive a round trip through
SQLite with its Decimals intact, and be readable as JSON without a database
client. Everything runs offline against a tmp_path database; the real
``data/collector.db`` is never touched.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from dataclasses import fields
from datetime import UTC, datetime, timedelta
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
    event_from_json,
    event_json,
    to_jsonable,
)
from collector.audit.events import proposal_from_json
from collector.audit.store import (
    DB_PATH_ENV_VAR,
    DEFAULT_DB_PATH,
    DEFAULT_RETENTION_DAYS,
    RETENTION_DAYS_ENV_VAR,
    retention_days,
)
from collector.decision_engine import RuleId, Verdict, validate_offer
from collector.money import Money
from collector.negotiation import NegotiationState
from collector.offers import Cadence, ConsumerProposal, Installment, Offer, Tier
from collector.policy import PolicyConfig

CALL_ID = "CALL-0001"

#: The four evidence tables, plus the hash chain that makes an edit to any of
#: them detectable and the log of what retention purges have deleted.
TABLES = ("agreements", "audit_chain", "calls", "decisions", "purges", "turns")

# A legal settlement: $800 over three monthly payments, each clearing the $250
# floor. Hand-built rather than engine-emitted — the engine's own even split is
# 266.67/266.67/266.66.
SETTLEMENT = Offer(
    tier=Tier.SETTLEMENT,
    installments=(
        Installment(Money("250.00"), 0),
        Installment(Money("250.00"), 30),
        Installment(Money("300.00"), 60),
    ),
    cadence=Cadence.MONTHLY,
)


#: The 2026-08-07 relay: the consumer said "$150 a month" and named no total,
#: so the tool layer assembled one and derived the count. Every field carries a
#: value other than its default, so a round trip cannot come back equal while
#: quietly dropping one.
RELAYED_150 = ConsumerProposal(
    total=Money("1000.00"),
    payment_count=7,
    cadence=Cadence.MONTHLY,
    signaled_capacity=Money("150.00"),
    amount_each=Money("150.00"),
    total_stated=False,
)

#: What the engine ruled on it, so the decision event under test is a real one
#: rather than a hand-assembled shape.
RELAYED_VERDICT = validate_offer(
    RELAYED_150,
    NegotiationState.opening(PolicyConfig.default()),
    PolicyConfig.default(),
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
    def test_creates_the_evidence_tables(self, store: AuditStore) -> None:
        assert store.table_names() == TABLES

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
            assert second.table_names() == TABLES
            assert len(second.turns(CALL_ID)) == 1

    def test_uses_the_configured_path_only(self, tmp_path: Path) -> None:
        """Tests must never write to the real data/ directory."""
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
        the point of the log."""
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
        """Production sampling needs to filter without parsing JSON."""
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


class TestTheProposalReadsBackAsTheOneRuledOn:
    """A decision record must name the consumer's own money, not an even split.

    ``ConsumerProposal`` carries two provenance fields — ``amount_each``, the
    per-payment figure the consumer actually said, and ``total_stated``, false
    when the tool layer assembled the total around a shape. They exist because
    a figure the consumer named and a figure we derived are not interchangeable
    evidence (``offers.py``), and ``smallest_payment`` is the figure the payment
    floor is answered on.

    ``to_jsonable`` writes all six fields; a decoder that reconstructs four
    puts a proposal nobody made into every reader of ``store.decisions`` and
    into the rehydrated ``AgreementRecord`` — the 2026-08-07 "$150 a month"
    shape, restored from the log as $142.85.
    """

    RELAYED = RELAYED_150

    def test_every_field_is_exercised(self) -> None:
        assert {f.name for f in fields(ConsumerProposal)} == {
            "total",
            "payment_count",
            "cadence",
            "signaled_capacity",
            "amount_each",
            "total_stated",
        }

    def test_the_decision_survives_the_json_round_trip(self) -> None:
        event = DecisionRecorded(CALL_ID, 4, self.RELAYED, RELAYED_VERDICT, at="t4d")

        assert event_from_json(event_json(event)) == event

    def test_the_figure_the_floor_is_answered_on_is_not_re_derived(self) -> None:
        """$150 a month is what they said they could produce. An even split of
        an assembled $1,000 over seven payments is $142.85 — a number nobody
        named, and a different answer from the payment floor."""
        loaded = proposal_from_json(to_jsonable(self.RELAYED))

        assert loaded.amount_each == Money("150.00")
        assert loaded.total_stated is False
        assert loaded.smallest_payment == Money("150.00")

    def test_a_row_written_before_the_provenance_fields_still_reads(self) -> None:
        """Rows already in the log carry neither key, and an evidence store that
        cannot read its own older rows back has lost the evidence."""
        older = to_jsonable(ConsumerProposal(Money("800"), 3, Cadence.MONTHLY))
        del older["amount_each"]
        del older["total_stated"]

        assert proposal_from_json(older) == ConsumerProposal(Money("800"), 3, Cadence.MONTHLY)

    def test_the_stored_decision_reads_back_intact(self, store: AuditStore) -> None:
        store.record(DecisionRecorded(CALL_ID, 4, self.RELAYED, RELAYED_VERDICT, at="t4d"))

        assert store.decisions(CALL_ID)[0].proposal == self.RELAYED


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
        this one file."""
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
        # $200 fails both the payment floor and the settlement floor. MIN_PAYMENT
        # is evaluated first because it answers the figure the consumer actually
        # named, rather than a total assembled around it.
        assert data["exchanges"][0]["verdict"]["rationale_code"] == "BELOW_MIN_PAYMENT"
        assert len(data["guardrail_events"]) == 1
        assert len(data["escalations"]) == 1
        assert data["confirmation"]["confirmed"] is True
        assert data["rationale_code"] == "ACCEPTED"

    def test_export_parses_back_into_the_record(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        raw = store.agreement_json_path(agreement.agreement_id).read_text()

        assert AgreementRecord.from_json(raw) == agreement


# --------------------------------------------------------------------------
# file permissions — the log is plaintext consumer speech
# --------------------------------------------------------------------------


def mode_of(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


class TestFilePermissions:
    """0600 on every file this store creates, 0700 on the directory.

    Not encryption — an access control, and the only one this build has. The
    rows are verbatim debt-collection transcripts; the SQLite default of
    0644 publishes them to every other account on the host.
    """

    def test_database_and_its_wal_sidecars_are_owner_only(self, tmp_path: Path) -> None:
        """The WAL holds committed rows not yet checkpointed back — the same
        transcripts, in a second file SQLite creates for itself."""
        with AuditStore(tmp_path / "data" / "collector.db") as store:
            store.record(TurnRecorded(CALL_ID, 0, Speaker.AGENT, "hello", at="t0"))
            for path in (
                store.db_path,
                Path(f"{store.db_path}-wal"),
                Path(f"{store.db_path}-shm"),
            ):
                assert path.exists(), path
                assert mode_of(path) == 0o600, path

    def test_the_data_directory_is_owner_only(self, tmp_path: Path) -> None:
        with AuditStore(tmp_path / "data" / "collector.db") as store:
            assert mode_of(store.db_path.parent) == 0o700

    def test_the_agreement_json_is_owner_only(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        path = store.agreement_json_path(agreement.agreement_id)

        assert mode_of(path) == 0o600
        assert mode_of(store.json_dir) == 0o700

    def test_a_log_written_before_this_is_tightened_on_open(self, tmp_path: Path) -> None:
        """The fix has to reach databases that already exist, or the only
        protected deployments are the ones that never took a call."""
        path = tmp_path / "collector.db"
        with AuditStore(path):
            pass
        os.chmod(path, 0o644)

        with AuditStore(path) as store:
            assert mode_of(store.db_path) == 0o600


# --------------------------------------------------------------------------
# tamper-evidence — the hash chain
# --------------------------------------------------------------------------


class TestHashChain:
    def test_a_recorded_call_verifies_clean(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        del agreement
        assert store.verify_chain() is None

    def test_the_call_row_being_rewritten_at_the_end_is_not_tampering(
        self, store: AuditStore
    ) -> None:
        """CallStarted then CallEnded legitimately UPDATE the same row. Each
        chain entry was true when it was written; only the newest is held
        against the row as it stands."""
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
        store.record(CallEnded(CALL_ID, CallOutcome.ABANDONED, turn_count=2, at="t9"))

        assert store.verify_chain() is None

    def test_a_row_mutated_behind_the_stores_back_is_caught(
        self, tmp_path: Path, lowball: Verdict, accepted: Verdict
    ) -> None:
        """The threat: someone with file access edits what was said. They can
        — that is why this is tamper-*evidence*. What they cannot do is leave
        the digests agreeing."""
        db = tmp_path / "collector.db"
        with AuditStore(db, json_dir=tmp_path / "agreements") as store:
            record_call(store, lowball, accepted)
            assert store.verify_chain() is None

            raw = sqlite3.connect(db)
            raw.execute("UPDATE turns SET text = 'I never said that' WHERE turn_index = 1")
            raw.commit()
            raw.close()

            broken = store.verify_chain()

        assert broken is not None
        assert broken.table_name == "turns"
        assert broken.reason == "payload"

    def test_an_edited_chain_entry_is_caught(self, store: AuditStore) -> None:
        """Rewriting the chain to cover a doctored row breaks the link math
        instead — the entry no longer follows from the one before it."""
        store.record(TurnRecorded(CALL_ID, 0, Speaker.AGENT, "hello", at="t0"))
        store.record(TurnRecorded(CALL_ID, 1, Speaker.CONSUMER, "hi", at="t1"))
        raw = sqlite3.connect(store.db_path)
        raw.execute("UPDATE audit_chain SET payload_hash = ? WHERE seq = 2", ("0" * 64,))
        raw.commit()
        raw.close()

        broken = store.verify_chain()

        assert broken is not None
        assert (broken.seq, broken.reason) == (2, "entry_hash")

    def test_every_evidence_table_is_chained(
        self, store: AuditStore, agreement: AgreementRecord
    ) -> None:
        del agreement
        chained = {r["table_name"] for r in store._rows("SELECT table_name FROM audit_chain", ())}

        assert chained == {"calls", "turns", "decisions", "agreements"}


# --------------------------------------------------------------------------
# retention and purge
# --------------------------------------------------------------------------


def aged_call(
    store: AuditStore,
    accepted: Verdict,
    confirmation: ConsumerConfirmation,
    call_id: str,
    at: str,
) -> AgreementRecord:
    """A whole closed-out call — turn, decision, agreement, JSON — dated ``at``."""
    store.record(
        CallStarted(
            call_id=call_id,
            account_ref="ACCT-77",
            consumer_ref="CONS-77",
            original_balance=Money("1000.00"),
            channel="text",
            at=at,
        )
    )
    store.record(TurnRecorded(call_id, 0, Speaker.CONSUMER, "Eight hundred?", at=at))
    store.record(
        DecisionRecorded(
            call_id, 0, ConsumerProposal(Money("800"), 3, Cadence.MONTHLY), accepted, at=at
        )
    )
    record = store.finalize_agreement(
        call_id=call_id,
        final_offer=SETTLEMENT,
        authorizing_verdict=accepted,
        confirmation=confirmation,
        at=at,
    )
    store.record(CallEnded(call_id, CallOutcome.AGREED, turn_count=1, at=at))
    return record


def iso_days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


class TestRetention:
    def test_the_default_window_is_three_years(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regulation F: evidence of compliance is kept for three years after
        the last collection activity, so a shorter default is a violation."""
        monkeypatch.delenv(RETENTION_DAYS_ENV_VAR, raising=False)

        assert DEFAULT_RETENTION_DAYS == 1095
        assert retention_days() == 1095

    def test_the_window_can_be_set_by_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(RETENTION_DAYS_ENV_VAR, "2555")

        assert retention_days() == 2555

    def test_an_unparseable_window_raises_rather_than_defaulting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falling back to the default would keep three years of records an
        operator believed they had deleted."""
        monkeypatch.setenv(RETENTION_DAYS_ENV_VAR, "90d")

        with pytest.raises(ValueError, match="whole number of days"):
            retention_days()


class TestPurge:
    @pytest.fixture
    def aged(
        self,
        store: AuditStore,
        accepted: Verdict,
        confirmation: ConsumerConfirmation,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[AgreementRecord, AgreementRecord]:
        """One call four years old, one from today."""
        monkeypatch.delenv(RETENTION_DAYS_ENV_VAR, raising=False)
        old = aged_call(store, accepted, confirmation, "CALL-OLD", iso_days_ago(4 * 365))
        recent = aged_call(store, accepted, confirmation, "CALL-NEW", iso_days_ago(1))
        return old, recent

    def test_deletes_the_expired_call_and_every_dependent_row(
        self, store: AuditStore, aged: tuple[AgreementRecord, AgreementRecord]
    ) -> None:
        del aged
        record = store.purge_older_than()

        assert (record.calls, record.turns, record.decisions, record.agreements) == (1, 1, 1, 1)
        assert store.turns("CALL-OLD") == ()
        assert store.decisions("CALL-OLD") == ()
        assert store.agreement("CALL-OLD") is None
        assert store._rows("SELECT * FROM calls WHERE call_id = ?", ("CALL-OLD",)) == []

    def test_keeps_everything_inside_the_window(
        self, store: AuditStore, aged: tuple[AgreementRecord, AgreementRecord]
    ) -> None:
        _, recent = aged
        store.purge_older_than()

        assert store.agreement("CALL-NEW") == recent
        assert len(store.turns("CALL-NEW")) == 1
        assert len(store.decisions("CALL-NEW")) == 1

    def test_deletes_the_agreement_json_sidecars_too(
        self, store: AuditStore, aged: tuple[AgreementRecord, AgreementRecord]
    ) -> None:
        """A record deleted from the database and left on disk as JSON has not
        been deleted."""
        old, recent = aged
        old_json = store.agreement_json_path(old.agreement_id)
        recent_json = store.agreement_json_path(recent.agreement_id)
        assert old_json.exists()

        record = store.purge_older_than()

        assert record.json_files == 1
        assert not old_json.exists()
        assert recent_json.exists()

    def test_the_purge_is_itself_recorded_and_survives_the_next_one(
        self, store: AuditStore, aged: tuple[AgreementRecord, AgreementRecord]
    ) -> None:
        """Once the evidence is gone, the record of its deletion is the only
        answer left to "where are the records?"."""
        del aged
        first = store.purge_older_than()
        second = store.purge_older_than(days=0)

        assert store.purges() == (first, second)
        assert first.retention_days == 1095
        assert first.cutoff < datetime.now(UTC).isoformat()
        assert second.calls == 1, "the second pass takes the call the first one kept"

    def test_the_chain_still_verifies_after_a_purge(
        self, store: AuditStore, aged: tuple[AgreementRecord, AgreementRecord]
    ) -> None:
        """Purging is a legitimate deletion and must not read as forgery.

        Also the canary for VACUUM: the chain references rows by rowid, and
        rowids surviving the rebuild is what keeps every entry pointing at the
        row it was written for rather than at its neighbour.
        """
        del aged
        store.purge_older_than()

        assert store.verify_chain() is None

    def test_permissions_survive_the_vacuum(
        self, store: AuditStore, aged: tuple[AgreementRecord, AgreementRecord]
    ) -> None:
        del aged
        store.purge_older_than()

        assert mode_of(store.db_path) == 0o600
        assert mode_of(Path(f"{store.db_path}-wal")) == 0o600

    def test_opening_the_store_never_purges(
        self, tmp_path: Path, accepted: Verdict, confirmation: ConsumerConfirmation
    ) -> None:
        """A silent purge on startup would be a compliance incident, not a
        feature: this is only ever the operator's explicit act."""
        db = tmp_path / "collector.db"
        with AuditStore(db, json_dir=tmp_path / "agreements") as first:
            aged_call(first, accepted, confirmation, "CALL-ANCIENT", "2015-01-01T00:00:00+00:00")

        with AuditStore(db, json_dir=tmp_path / "agreements") as second:
            assert second.agreement("CALL-ANCIENT") is not None
            assert second.purges() == ()


# --------------------------------------------------------------------------
# durability — the WAL checkpoint
# --------------------------------------------------------------------------


def traced(store: AuditStore) -> list[str]:
    """Every statement the store's connection executes from here on.

    Asserted on the statement rather than on the file: SQLite auto-checkpoints
    and removes the WAL when the last connection closes anyway, so "the WAL is
    gone afterwards" passes with the checkpoint deleted.
    """
    statements: list[str] = []
    store._run(lambda: store._conn.set_trace_callback(statements.append))
    return statements


class TestCheckpoint:
    def test_close_forces_a_checkpoint(self, tmp_path: Path) -> None:
        """A call abandoned mid-turn writes no CallEnded and no agreement, so
        close() is the only checkpoint it will ever get."""
        store = AuditStore(tmp_path / "c.db", json_dir=tmp_path / "agreements")
        store.record(TurnRecorded(CALL_ID, 0, Speaker.AGENT, "hello", at="t0"))
        statements = traced(store)

        store.close()

        assert any("wal_checkpoint" in s for s in statements)

    def test_a_call_ending_without_an_agreement_forces_a_checkpoint(
        self, store: AuditStore
    ) -> None:
        """Escalation, hang-up and refusal are exactly the commits a crash
        would lose under synchronous = NORMAL, and exactly the calls a
        compliance reviewer comes looking for."""
        statements = traced(store)

        store.record(CallEnded(CALL_ID, CallOutcome.ESCALATED, turn_count=3, at="t3"))

        assert any("wal_checkpoint" in s for s in statements)

    def test_a_turn_does_not_checkpoint(self, store: AuditStore) -> None:
        """One fsync per turn is what synchronous = NORMAL exists to avoid; a
        checkpoint per row puts it back on the voice critical path."""
        statements = traced(store)

        store.record(TurnRecorded(CALL_ID, 0, Speaker.AGENT, "hello", at="t0"))

        assert not [s for s in statements if "wal_checkpoint" in s]


def test_the_default_path_constant_is_resolved_through_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """voice_app.py passes DEFAULT_DB_PATH itself rather than calling
    default_db_path(), so $COLLECTOR_DB_PATH had no effect on the voice path at
    all: the deploy target set it and the log still landed in a CWD-relative
    data/. The constant is resolved where the path is used instead."""
    monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "elsewhere" / "collector.db"))

    with AuditStore(DEFAULT_DB_PATH, json_dir=DEFAULT_DB_PATH.parent) as store:
        assert store.db_path == tmp_path / "elsewhere" / "collector.db"
        # The sidecars follow the database, or the agreements end up on a
        # different volume from the log that references them.
        assert store.json_dir == tmp_path / "elsewhere"
        assert store.db_path.exists()


def test_default_db_path_is_under_data(tmp_path: Path) -> None:
    """Never write outside ./data. Asserted on the constant, not by
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
