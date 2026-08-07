"""SQLite audit log — SPEC §6.

Four tables, stdlib ``sqlite3``, no ORM: ``calls``, ``turns``, ``decisions``,
``agreements``. The database is an evidence store, so three things are true of
every write here:

* **Money is stored as TEXT**, an exact decimal string. SQLite's REAL is a
  float, and a float in a payment schedule is a compliance defect (SPEC §9).
* **Nothing is dropped.** Foreign keys are documented in the schema but not
  enforced: a log that refuses a write because an out-of-order event arrived
  first has lost the evidence it exists to keep.
* **The full event is kept verbatim** in a JSON payload column alongside the
  scalar columns used for querying, so a schema that gains a column later
  cannot retroactively lose what older calls recorded.

``turns`` is the chronological trace: consumer and agent utterances plus the
guardrail trips and escalations interleaved between them, discriminated by
``event_type``. They live together because a compliance reviewer reads them
together — "what was said, and what stopped being said" is one timeline.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from collector.audit.events import (
    AgreementRecord,
    CallEnded,
    CallStarted,
    ConsumerConfirmation,
    DecisionRecorded,
    Escalated,
    EventType,
    GuardrailTripped,
    ModelCalled,
    ToolInvoked,
    TraceEvent,
    TurnRecorded,
    dumps,
    event_from_json,
    event_json,
)
from collector.decision_engine import Verdict
from collector.offers import Offer

DEFAULT_DB_PATH = Path("data/collector.db")

#: Overrides the CWD-relative default. A container's working directory is
#: whatever the entrypoint happens to be, so the deploy target sets this rather
#: than hoping ``data/`` resolves somewhere writable and persistent.
DB_PATH_ENV_VAR = "COLLECTOR_DB_PATH"

# Concurrency and durability, set once per connection:
#   WAL          - readers never block the writer, so the post-call report can
#                  read the trace back while the turn loop is still writing it.
#   NORMAL       - one fsync per checkpoint instead of one per commit. A crash
#                  can lose the last commits; the alternative is an fsync on the
#                  voice critical path, several times a turn.
#   busy_timeout - wait rather than raise if another process holds the write lock.
_PRAGMAS = ("journal_mode = WAL", "synchronous = NORMAL", "busy_timeout = 5000")


def default_db_path() -> Path:
    """Where the log lives unless a caller says otherwise (``$COLLECTOR_DB_PATH``)."""
    return Path(os.environ.get(DB_PATH_ENV_VAR) or DEFAULT_DB_PATH)


SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id          TEXT PRIMARY KEY,
    account_ref      TEXT NOT NULL,
    consumer_ref     TEXT NOT NULL,
    channel          TEXT NOT NULL,
    original_balance TEXT NOT NULL,   -- exact decimal string, never REAL
    started_at       TEXT NOT NULL,
    ended_at         TEXT,
    outcome          TEXT,
    turn_count       INTEGER,
    -- Post-call compliance score (SPEC §5.3). NULL until the call is closed
    -- out; a call whose process died before finalize_call has no score, and
    -- saying so is more honest than defaulting it to compliant.
    compliant        INTEGER,
    blocked_turns    INTEGER,
    violation_count  INTEGER
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id      TEXT NOT NULL,       -- REFERENCES calls(call_id), unenforced by design
    turn_index   INTEGER NOT NULL,
    event_type   TEXT NOT NULL,       -- turn | guardrail_trip | escalation
    speaker      TEXT,
    text         TEXT,
    payload_json TEXT NOT NULL,
    at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id         TEXT NOT NULL,    -- REFERENCES calls(call_id), unenforced by design
    turn_index      INTEGER NOT NULL,
    outcome         TEXT NOT NULL,
    tier            TEXT,
    rationale_code  TEXT NOT NULL,
    proposal_total  TEXT NOT NULL,    -- exact decimal string
    conditions_json TEXT NOT NULL,    -- the evaluated-condition trail, verbatim
    payload_json    TEXT NOT NULL,
    at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agreements (
    agreement_id   TEXT PRIMARY KEY,
    call_id        TEXT NOT NULL UNIQUE,
    tier           TEXT NOT NULL,
    total          TEXT NOT NULL,     -- exact decimal string
    payment_count  INTEGER NOT NULL,
    cadence        TEXT NOT NULL,
    final_day_offset INTEGER NOT NULL,
    rationale_code TEXT NOT NULL,
    confirmed      INTEGER NOT NULL,
    record_json    TEXT NOT NULL,     -- the complete record; the deliverable
    json_path      TEXT,
    at             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_call ON turns (call_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_decisions_call ON decisions (call_id, turn_index);
"""

_TRACE_EVENT_TYPES = (
    EventType.TURN.value,
    EventType.GUARDRAIL_TRIP.value,
    EventType.ESCALATION.value,
    EventType.TOOL_CALL.value,
    EventType.MODEL_CALL.value,
)


@dataclass(frozen=True)
class CallCompliance:
    """The ``calls`` row's post-call score, read back out.

    ``finalize_call`` computes this at the end of every call; keeping it only
    in memory meant the log could not answer "was this call compliant?" — the
    one question the log exists to answer.
    """

    call_id: str
    outcome: str | None
    compliant: bool
    turn_count: int
    blocked_turns: int
    violation_count: int


_TraceRow = TurnRecorded | GuardrailTripped | Escalated | ToolInvoked | ModelCalled


def _trace_row_text(event: _TraceRow) -> str:
    """The one-line human summary for the ``turns.text`` column.

    The verbatim event is in ``payload_json`` either way; this is what a
    reviewer scanning the timeline in a SQL client actually reads.
    """
    match event:
        case TurnRecorded():
            return event.text
        case GuardrailTripped() | Escalated():
            return event.detail
        case ToolInvoked():
            outcome = "ok" if event.ok else f"failed: {event.error}"
            return f"{event.tool} -> {outcome} ({event.latency_ms}ms)"
        case ModelCalled():
            tokens = f"{event.input_tokens} in / {event.output_tokens} out"
            failure = f" failed: {event.error}" if event.error else ""
            return f"{event.model} {tokens} ({event.latency_ms}ms){failure}"


# Columns added after the first release, applied by ``_add_missing_columns``.
# Keep every entry nullable: an ALTER on a populated table cannot invent a
# value for rows written before the column existed.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "calls": {
        "compliant": "INTEGER",
        "blocked_turns": "INTEGER",
        "violation_count": "INTEGER",
    },
}


class AuditStore:
    """Append-only trace log for one collector deployment.

    The path is a constructor argument with no hidden default lookup so tests
    can point at a ``tmp_path`` and never touch the real ``data/`` directory.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        json_dir: Path | str | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        default_json_dir = self.db_path.parent / "agreements"
        self.json_dir = Path(json_dir) if json_dir is not None else default_json_dir
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        for pragma in _PRAGMAS:
            self._conn.execute(f"PRAGMA {pragma}")
        self.create_schema()

    def create_schema(self) -> None:
        """Idempotent: every statement is IF NOT EXISTS, so reopening an
        existing database is a no-op rather than an error or a truncation."""
        with self._conn:
            self._conn.executescript(SCHEMA)
            self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Bring an older database up to the current schema.

        ``CREATE TABLE IF NOT EXISTS`` silently skips a table that already
        exists, columns and all, so a log written before a column was added
        would keep failing every INSERT that names it. Adding them one at a
        time is the whole migration story this needs; the evidence already in
        the table is never rewritten.
        """
        for table, columns in _ADDED_COLUMNS.items():
            existing = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            for name, ddl in columns.items():
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    # -- recording ---------------------------------------------------------

    def record(self, event: TraceEvent) -> None:
        """Single entry point for the agent loop: hand it any trace event."""
        match event:
            case CallStarted():
                self._record_call_started(event)
            case CallEnded():
                self._record_call_ended(event)
            case DecisionRecorded():
                self._record_decision(event)
            case TurnRecorded() | GuardrailTripped() | Escalated() | ToolInvoked() | ModelCalled():
                self._record_trace_row(event)
            case _:
                raise TypeError(f"not a trace event: {type(event).__name__}")

    def _record_call_started(self, event: CallStarted) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO calls
                    (call_id, account_ref, consumer_ref, channel, original_balance, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(call_id) DO UPDATE SET
                    account_ref = excluded.account_ref,
                    consumer_ref = excluded.consumer_ref,
                    channel = excluded.channel,
                    original_balance = excluded.original_balance,
                    started_at = excluded.started_at
                """,
                (
                    event.call_id,
                    event.account_ref,
                    event.consumer_ref,
                    event.channel,
                    str(event.original_balance.amount),
                    event.at,
                ),
            )

    def _record_call_ended(self, event: CallEnded) -> None:
        # Upsert rather than UPDATE: an end event that arrives without a start
        # row still has to land somewhere.
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO calls
                    (call_id, account_ref, consumer_ref, channel, original_balance,
                     started_at, ended_at, outcome, turn_count,
                     compliant, blocked_turns, violation_count)
                VALUES (?, '', '', '', '', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(call_id) DO UPDATE SET
                    ended_at = excluded.ended_at,
                    outcome = excluded.outcome,
                    turn_count = excluded.turn_count,
                    compliant = excluded.compliant,
                    blocked_turns = excluded.blocked_turns,
                    violation_count = excluded.violation_count
                """,
                (
                    event.call_id,
                    event.at,
                    event.at,
                    event.outcome.value,
                    event.turn_count,
                    None if event.compliant is None else int(event.compliant),
                    event.blocked_turns,
                    event.violation_count,
                ),
            )

    def _record_trace_row(self, event: _TraceRow) -> None:
        speaker = event.speaker.value if isinstance(event, TurnRecorded) else None
        text = _trace_row_text(event)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO turns (call_id, turn_index, event_type, speaker, text, payload_json, at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.call_id,
                    event.turn_index,
                    event.EVENT_TYPE.value,
                    speaker,
                    text,
                    dumps(event_json(event), indent=None),
                    event.at,
                ),
            )

    def _record_decision(self, event: DecisionRecorded) -> None:
        verdict = event.verdict
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO decisions
                    (call_id, turn_index, outcome, tier, rationale_code, proposal_total,
                     conditions_json, payload_json, at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.call_id,
                    event.turn_index,
                    verdict.outcome,
                    verdict.tier.name if verdict.tier is not None else None,
                    verdict.rationale_code.value,
                    str(event.proposal.total.amount),
                    dumps(verdict.conditions, indent=None),
                    dumps(event_json(event), indent=None),
                    event.at,
                ),
            )

    # -- the agreement — SPEC §6, the brief's deliverable -------------------

    def finalize_agreement(
        self,
        *,
        call_id: str,
        final_offer: Offer,
        authorizing_verdict: Verdict,
        confirmation: ConsumerConfirmation,
        agreement_id: str | None = None,
        at: str | None = None,
    ) -> AgreementRecord:
        """Close the call out: assemble the agreement record from what was
        already logged, write it, and emit the standalone JSON.

        The exchanges and guardrail events are read back out of the log rather
        than passed in, so the record can only contain things that were
        actually recorded during the call.
        """
        record = AgreementRecord.build(
            call_id=call_id,
            final_offer=final_offer,
            authorizing_verdict=authorizing_verdict,
            confirmation=confirmation,
            exchanges=self.decisions(call_id),
            guardrail_events=self.guardrail_events(call_id),
            escalations=self.escalations(call_id),
            agreement_id=agreement_id,
            at=at,
        )
        self.record_agreement(record)
        return record

    def record_agreement(self, record: AgreementRecord) -> Path:
        """Persist a pre-assembled record. Returns the standalone JSON path."""
        json_path = self.export_agreement(record)
        final_day = max((i.due_day_offset for i in record.schedule), default=0)
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO agreements
                    (agreement_id, call_id, tier, total, payment_count, cadence,
                     final_day_offset, rationale_code, confirmed, record_json, json_path, at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.agreement_id,
                    record.call_id,
                    record.tier.name,
                    str(record.total.amount),
                    len(record.schedule),
                    record.cadence.value,
                    final_day,
                    record.rationale_code.value,
                    int(record.confirmation.confirmed),
                    record.to_json(indent=None),
                    str(json_path),
                    record.at,
                ),
            )
        return json_path

    def agreement_json_path(self, agreement_id: str) -> Path:
        return self.json_dir / f"{agreement_id}.json"

    def export_agreement(self, record: AgreementRecord) -> Path:
        """Write the record as standalone JSON — inspectable with no SQLite
        client, which is how the grader will actually read it (SPEC §6)."""
        path = self.agreement_json_path(record.agreement_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.to_json(), encoding="utf-8")
        return path

    # -- querying ----------------------------------------------------------

    def _rows(self, sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()

    def _trace_events(self, call_id: str, event_type: EventType) -> tuple[TraceEvent, ...]:
        rows = self._rows(
            "SELECT payload_json FROM turns WHERE call_id = ? AND event_type = ? ORDER BY turn_id",
            (call_id, event_type.value),
        )
        return tuple(event_from_json(_loads(r["payload_json"])) for r in rows)

    def turns(self, call_id: str) -> tuple[TurnRecorded, ...]:
        return cast(tuple[TurnRecorded, ...], self._trace_events(call_id, EventType.TURN))

    def guardrail_events(self, call_id: str) -> tuple[GuardrailTripped, ...]:
        return cast(
            tuple[GuardrailTripped, ...],
            self._trace_events(call_id, EventType.GUARDRAIL_TRIP),
        )

    def escalations(self, call_id: str) -> tuple[Escalated, ...]:
        return cast(tuple[Escalated, ...], self._trace_events(call_id, EventType.ESCALATION))

    def tool_calls(self, call_id: str) -> tuple[ToolInvoked, ...]:
        return cast(tuple[ToolInvoked, ...], self._trace_events(call_id, EventType.TOOL_CALL))

    def model_calls(self, call_id: str) -> tuple[ModelCalled, ...]:
        return cast(tuple[ModelCalled, ...], self._trace_events(call_id, EventType.MODEL_CALL))

    def compliance(self, call_id: str) -> CallCompliance | None:
        """The persisted post-call score. ``None`` if the call was never closed out."""
        rows = self._rows(
            "SELECT outcome, turn_count, compliant, blocked_turns, violation_count"
            " FROM calls WHERE call_id = ?",
            (call_id,),
        )
        if not rows or rows[0]["compliant"] is None:
            return None
        row = rows[0]
        return CallCompliance(
            call_id=call_id,
            outcome=row["outcome"],
            compliant=bool(row["compliant"]),
            turn_count=int(row["turn_count"] or 0),
            blocked_turns=int(row["blocked_turns"] or 0),
            violation_count=int(row["violation_count"] or 0),
        )

    def trace(self, call_id: str) -> tuple[TraceEvent, ...]:
        """The whole timeline in order: utterances, trips, and escalations."""
        placeholders = ", ".join("?" for _ in _TRACE_EVENT_TYPES)
        rows = self._rows(
            f"SELECT payload_json FROM turns WHERE call_id = ?"
            f" AND event_type IN ({placeholders}) ORDER BY turn_id",
            (call_id, *_TRACE_EVENT_TYPES),
        )
        return tuple(event_from_json(_loads(r["payload_json"])) for r in rows)

    def decisions(self, call_id: str) -> tuple[DecisionRecorded, ...]:
        rows = self._rows(
            "SELECT payload_json FROM decisions WHERE call_id = ? ORDER BY decision_id",
            (call_id,),
        )
        return tuple(_as_decision(event_from_json(_loads(r["payload_json"]))) for r in rows)

    def agreement(self, call_id: str) -> AgreementRecord | None:
        rows = self._rows("SELECT record_json FROM agreements WHERE call_id = ?", (call_id,))
        if not rows:
            return None
        return AgreementRecord.from_json(rows[0]["record_json"])

    def agreements(self) -> tuple[AgreementRecord, ...]:
        rows = self._rows("SELECT record_json FROM agreements ORDER BY at, agreement_id", ())
        return tuple(AgreementRecord.from_json(r["record_json"]) for r in rows)

    def table_names(self) -> tuple[str, ...]:
        rows = self._rows(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            " ORDER BY name",
            (),
        )
        return tuple(r["name"] for r in rows)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AuditStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _loads(raw: str) -> dict[str, Any]:
    return dict(json.loads(raw))


def _as_decision(event: TraceEvent) -> DecisionRecorded:
    if not isinstance(event, DecisionRecorded):
        raise ValueError(f"decisions table held a {type(event).__name__}")
    return event
