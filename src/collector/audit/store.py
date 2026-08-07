"""SQLite audit log.

Four tables, stdlib ``sqlite3``, no ORM: ``calls``, ``turns``, ``decisions``,
``agreements``. The database is an evidence store, so three things are true of
every write here:

* **Money is stored as TEXT**, an exact decimal string. SQLite's REAL is a
  float, and a float in a payment schedule is a compliance defect.
* **Nothing is dropped.** Foreign keys are documented in the schema but not
  enforced: a log that refuses a write because an out-of-order event arrived
  first has lost the evidence it exists to keep.
* **The full event is kept verbatim** in a JSON payload column alongside the
  scalar columns used for querying, so a schema that gains a column later
  cannot retroactively lose what older calls recorded.
* **Every insert is chained.** The row and its ``audit_chain`` entry commit in
  one transaction, so an edit made to the file behind this module's back is
  *detectable* — see ``AuditStore.verify_chain`` for exactly what that does and
  does not cover. It is tamper-evidence, not tamper-proofing.

``turns`` is the chronological trace: consumer and agent utterances plus the
guardrail trips and escalations interleaved between them, discriminated by
``event_type``. They live together because a compliance reviewer reads them
together — "what was said, and what stopped being said" is one timeline.

The database, its WAL sidecars, and the exported agreement JSON are created
mode 0600 inside a 0700 directory: this file is plaintext transcripts of
debt-collection calls, and the default 0644 makes them readable by every other
account on the host. That is an access control and nothing more — it does not
protect against root, against a disk image, or against a backup. Encryption is
a deployment responsibility (README, "Storage and retention").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar, cast

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
    utc_now,
)
from collector.decision_engine import Verdict
from collector.offers import Offer

DEFAULT_DB_PATH = Path("data/collector.db")

#: Overrides the CWD-relative default. A container's working directory is
#: whatever the entrypoint happens to be, so the deploy target sets this rather
#: than hoping ``data/`` resolves somewhere writable and persistent.
DB_PATH_ENV_VAR = "COLLECTOR_DB_PATH"

#: Three years, and the default must not be shortened. CFPB Regulation F
#: (12 CFR 1006.100) requires a debt collector to retain evidence of compliance
#: with the FDCPA starting from the date of the last collection activity on an
#: account and running three years past it — so a log trimmed for disk space is
#: a records-retention violation, not a housekeeping decision.
DEFAULT_RETENTION_DAYS = 1095
RETENTION_DAYS_ENV_VAR = "COLLECTOR_RETENTION_DAYS"

#: Owner-only, on everything this module creates. The rows are verbatim
#: consumer speech; 0644 publishes them to every account on the host.
_FILE_MODE = 0o600
_DIR_MODE = 0o700

#: The genesis link. A chain has to start somewhere, and starting from a
#: constant means the first entry's hash is checkable like every other.
_CHAIN_GENESIS = "0" * 64

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


def retention_days() -> int:
    """How long a call is kept (``$COLLECTOR_RETENTION_DAYS``, else 3 years).

    A misparsed value raises rather than falling back to the default: silently
    ignoring ``COLLECTOR_RETENTION_DAYS=90d`` would keep three years of records
    an operator believed they had deleted.
    """
    raw = os.environ.get(RETENTION_DAYS_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_RETENTION_DAYS
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{RETENTION_DAYS_ENV_VAR} must be a whole number of days, got {raw!r}"
        ) from exc


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
    -- Post-call compliance score. NULL until the call is closed
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

-- One entry per insert into the four tables above, appended in the same
-- transaction as the row it covers, so a row and its link commit together or
-- neither does. Editing a row without also rewriting every link after it makes
-- verify_chain() report where the file diverged from what was written.
CREATE TABLE IF NOT EXISTS audit_chain (
    seq          INTEGER PRIMARY KEY,
    table_name   TEXT NOT NULL,
    row_id       INTEGER NOT NULL,  -- the covered row's sqlite rowid
    payload_hash TEXT NOT NULL,     -- sha256 of the canonically serialized row
    prev_hash    TEXT NOT NULL,
    entry_hash   TEXT NOT NULL      -- sha256(prev_hash || payload_hash)
);

-- What a retention purge deleted, and when. Deliberately not itself purgeable:
-- once the evidence is gone, "it was deleted on schedule, and here is the
-- cutoff it was deleted against" is the only remaining answer to a regulator
-- asking where the records went.
CREATE TABLE IF NOT EXISTS purges (
    purge_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    cutoff         TEXT NOT NULL,
    retention_days INTEGER NOT NULL,
    calls          INTEGER NOT NULL,
    turns          INTEGER NOT NULL,
    decisions      INTEGER NOT NULL,
    agreements     INTEGER NOT NULL,
    json_files     INTEGER NOT NULL,
    at             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_call ON turns (call_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_decisions_call ON decisions (call_id, turn_index);
"""

#: A call is dated by its *last* activity, falling back to its start for a call
#: whose process died before it was closed out — Regulation F runs the three
#: years from the last activity on the account, so ``started_at`` alone would
#: purge a long call too early.
_EXPIRED_CALLS = "SELECT call_id FROM calls WHERE COALESCE(ended_at, started_at) < ?"

_TRACE_EVENT_TYPES = (
    EventType.TURN.value,
    EventType.GUARDRAIL_TRIP.value,
    EventType.ESCALATION.value,
    EventType.TOOL_CALL.value,
    EventType.MODEL_CALL.value,
)

_T = TypeVar("_T")


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


@dataclass(frozen=True)
class ChainBreak:
    """Where the hash chain stopped agreeing with the file, from ``verify_chain``.

    ``reason`` is one of ``prev_hash`` (this entry does not follow the one
    before it), ``entry_hash`` (the entry's own digest was recomputed or
    forged), or ``payload`` (the entry is intact but the row it covers no
    longer hashes to it — the row was edited after it was written).
    """

    seq: int
    table_name: str
    row_id: int
    reason: str


@dataclass(frozen=True)
class PurgeRecord:
    """What one ``purge_older_than`` removed. Also written to ``purges``."""

    cutoff: str
    retention_days: int
    calls: int
    turns: int
    decisions: int
    agreements: int
    json_files: int


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

    The sqlite3 connection is only ever touched from one dedicated worker
    thread, owned by this store for its entire life (``self._executor``, a
    single-worker ``ThreadPoolExecutor``). Callers — the asyncio event loop
    via ``asyncio.to_thread``, which does not guarantee the same OS thread
    across separate calls, and the plain synchronous callers in
    ``text_app.py`` — can be on any thread; every touch of ``self._conn``
    routes through ``self._run``, which hands the work to the store's own
    worker thread and blocks for the result. This keeps the public API fully
    synchronous while satisfying sqlite3's ``check_same_thread`` requirement
    that the connection only ever be used from the thread that created it.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        json_dir: Path | str | None = None,
    ) -> None:
        requested = Path(db_path) if db_path is not None else None
        # ``None`` is not the only way a caller says "wherever this deployment
        # put it": voice_app.py hands over the DEFAULT_DB_PATH constant itself,
        # so $COLLECTOR_DB_PATH had no effect on the voice path at all — the
        # deploy target set it and the log still landed in a CWD-relative
        # data/. Resolving the bare constant here makes the env var
        # authoritative at the one place the path is actually used.
        if requested is None or requested == DEFAULT_DB_PATH:
            self.db_path = default_db_path()
        else:
            self.db_path = requested
        requested_json = Path(json_dir) if json_dir is not None else None
        if requested_json is None:
            self.json_dir = self.db_path.parent / "agreements"
        elif requested_json == DEFAULT_DB_PATH.parent:
            # Same story, same call site: voice_app.py passes
            # DEFAULT_DB_PATH.parent. Follow the resolved database rather than
            # the constant, or the agreements end up on a different volume from
            # the log that references them.
            self.json_dir = self.db_path.parent
        else:
            self.json_dir = requested_json
        _secure_dir(self.db_path.parent)
        self._store_thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audit-store")
        self._conn = self._run(self._connect)
        self.create_schema()

    def _run(self, fn: Callable[[], _T]) -> _T:
        """Run ``fn`` on the store's dedicated worker thread and block for its
        result, so ``self._conn`` is only ever touched from the one thread
        that created it — regardless of which thread the caller is on.

        Reentrancy-aware: if this is already running as a task on the store's
        own worker thread (e.g. a wrapped method calling another wrapped
        method), run ``fn`` inline instead of submitting. With
        ``max_workers=1``, submitting from inside the one worker thread while
        it is busy running this very call would block forever waiting on a
        queue only that same, currently-busy thread could ever drain.
        """
        if self._store_thread is not None and threading.current_thread() is self._store_thread:
            return fn()

        def _task() -> _T:
            self._store_thread = threading.current_thread()
            return fn()

        return self._executor.submit(_task).result()

    def _connect(self) -> sqlite3.Connection:
        # Create the file here, at 0600, rather than letting sqlite3 create it
        # at 0666 & ~umask and chmod'ing afterwards: between those two moments
        # the transcripts are world-readable, and that window opens on every
        # cold start. The chmod still runs, for a log that predates this.
        os.close(os.open(self.db_path, os.O_RDONLY | os.O_CREAT, _FILE_MODE))
        _secure_file(self.db_path)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        for pragma in _PRAGMAS:
            conn.execute(f"PRAGMA {pragma}")
        self._secure_wal_sidecars()
        return conn

    def _secure_wal_sidecars(self) -> None:
        """Lock down ``-wal`` and ``-shm``, which SQLite creates, not us.

        The WAL holds committed rows that have not been checkpointed back into
        the database yet — the same transcripts, in a second file. Its mode is
        whatever the umask allowed when SQLite created it, so restricting the
        database alone would leave the recent end of the log readable.
        """
        for suffix in ("-wal", "-shm"):
            _secure_file(Path(f"{self.db_path}{suffix}"))

    def create_schema(self) -> None:
        """Idempotent: every statement is IF NOT EXISTS, so reopening an
        existing database is a no-op rather than an error or a truncation."""

        def _create() -> None:
            with self._conn:
                self._conn.executescript(SCHEMA)
                self._add_missing_columns()

        self._run(_create)

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

    # -- tamper-evidence ---------------------------------------------------

    def _append_chain(self, table: str, row_id: int) -> None:
        """Append the chain entry covering ``table``'s row ``row_id``.

        Must be called from inside the ``with self._conn`` block that wrote the
        row: a row that committed without its link, or a link without its row,
        is indistinguishable from tampering, and a crash between two separate
        commits would manufacture exactly that. Reading the previous entry and
        writing the next one is a read-modify-write, which is safe here only
        because every caller reaches this through ``_run`` on the store's one
        writer thread.
        """
        row = self._conn.execute(f"SELECT * FROM {table} WHERE rowid = ?", (row_id,)).fetchone()
        payload_hash = _payload_hash(table, row)
        previous = self._conn.execute(
            "SELECT entry_hash FROM audit_chain ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        prev_hash = _CHAIN_GENESIS if previous is None else str(previous["entry_hash"])
        self._conn.execute(
            "INSERT INTO audit_chain (table_name, row_id, payload_hash, prev_hash, entry_hash)"
            " VALUES (?, ?, ?, ?, ?)",
            (table, row_id, payload_hash, prev_hash, _entry_hash(prev_hash, payload_hash)),
        )

    def _last_rowid(self) -> int:
        """The rowid just inserted. Not ``cursor.lastrowid``, which is typed
        optional and is undefined after the UPDATE arm of an upsert."""
        return int(self._conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def verify_chain(self) -> ChainBreak | None:
        """Re-walk the hash chain and return the first broken link, or ``None``.

        This makes tampering **detectable, not impossible**. Anyone who can
        write to the file can still edit a row; what they cannot do is edit one
        and leave the digests agreeing, without also rewriting every entry
        after it. Two things are checked, in ``seq`` order:

        * the link — ``prev_hash`` equals the previous entry's ``entry_hash``,
          and ``entry_hash`` equals ``sha256(prev_hash || payload_hash)``;
        * the row — for the *latest* entry naming a given row, the live row
          still hashes to the recorded ``payload_hash``. Only the latest,
          because a ``calls`` row is legitimately rewritten when the call ends;
          the earlier entries record earlier states of it, and each one was
          true when it was written.

        What this does not do: prove nothing was deleted. A row removed behind
        the store's back leaves its chain entry orphaned rather than breaking
        any hash, and is not reported here — because ``purge_older_than``
        deletes rows legitimately and must not make the chain read as forged.
        An orphaned entry is still evidence a row existed; noticing it is a
        reviewer's comparison of the tables against the chain, not this method.
        """

        def _verify() -> ChainBreak | None:
            entries = self._conn.execute("SELECT * FROM audit_chain ORDER BY seq").fetchall()
            latest = {(e["table_name"], e["row_id"]): e["seq"] for e in entries}
            prev_hash = _CHAIN_GENESIS
            for entry in entries:
                table, row_id, seq = entry["table_name"], entry["row_id"], entry["seq"]
                if entry["prev_hash"] != prev_hash:
                    return ChainBreak(seq, table, row_id, "prev_hash")
                if entry["entry_hash"] != _entry_hash(prev_hash, entry["payload_hash"]):
                    return ChainBreak(seq, table, row_id, "entry_hash")
                prev_hash = str(entry["entry_hash"])
                if latest[(table, row_id)] != seq:
                    continue
                row = self._conn.execute(
                    f"SELECT * FROM {table} WHERE rowid = ?", (row_id,)
                ).fetchone()
                if row is not None and _payload_hash(table, row) != entry["payload_hash"]:
                    return ChainBreak(seq, table, row_id, "payload")
            return None

        return self._run(_verify)

    # -- recording ---------------------------------------------------------

    def record(self, event: TraceEvent) -> None:
        """Single entry point for the agent loop: hand it any trace event."""

        def _dispatch() -> None:
            match event:
                case CallStarted():
                    self._record_call_started(event)
                case CallEnded():
                    self._record_call_ended(event)
                case DecisionRecorded():
                    self._record_decision(event)
                case (
                    TurnRecorded()
                    | GuardrailTripped()
                    | Escalated()
                    | ToolInvoked()
                    | ModelCalled()
                ):
                    self._record_trace_row(event)
                case _:
                    raise TypeError(f"not a trace event: {type(event).__name__}")

        self._run(_dispatch)

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
            self._append_chain("calls", self._call_rowid(event.call_id))

    def _call_rowid(self, call_id: str) -> int:
        """``calls`` is upserted, so the rowid has to be looked up rather than
        read off the cursor: on the UPDATE arm there was no insert."""
        row = self._conn.execute("SELECT rowid FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        return int(row[0])

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
            self._append_chain("calls", self._call_rowid(event.call_id))
        # The call is over — escalation, hang-up, refusal, or agreement. Under
        # `synchronous = NORMAL` (see _PRAGMAS) everything this call committed
        # is still only in the page cache, so an OS crash or a killed container
        # loses precisely the calls nobody chose to end well. record_agreement
        # already forces one for the agreed path; the other endings are the
        # ones a compliance reviewer comes looking for, and they had none.
        self._checkpoint()

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
            self._append_chain("turns", self._last_rowid())

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
            self._append_chain("decisions", self._last_rowid())

    # -- the agreement — the brief's deliverable ---------------------------

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

        def _persist() -> Path:
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
                self._append_chain("agreements", self._last_rowid())
            # `synchronous = NORMAL` is right for the turn loop — an fsync
            # several times a turn sits on the voice critical path. It is wrong
            # for this write. The agreement is the deliverable and
            # this is the last commit of a successful call, so under NORMAL it
            # would sit in the page cache until the next checkpoint: an OS crash
            # or power loss in the seconds after the consumer agreed would lose
            # the record *and* the negotiation that produced it, leaving nothing
            # to notice the gap with. This runs after the call, not during it,
            # so the fsync costs nothing that matters.
            self._checkpoint()
            return json_path

        return self._run(_persist)

    def agreement_json_path(self, agreement_id: str) -> Path:
        return self.json_dir / f"{agreement_id}.json"

    def export_agreement(self, record: AgreementRecord) -> Path:
        """Write the record as standalone JSON — inspectable with no SQLite
        client, which is how the grader will actually read it."""
        path = self.agreement_json_path(record.agreement_id)
        _secure_dir(path.parent)
        # os.open with the mode, not write_text-then-chmod: this file is the
        # whole agreement — consumer utterances, the negotiation, the lot — and
        # a 0644 file that is chmod'ed a millisecond later was still readable.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(record.to_json())
        _secure_file(path)  # O_CREAT leaves an already-existing file's mode alone
        return path

    # -- querying ----------------------------------------------------------

    def _rows(self, sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
        return self._run(lambda: self._conn.execute(sql, params).fetchall())

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

    def purges(self) -> tuple[PurgeRecord, ...]:
        """Every purge this log has been through, oldest first."""
        rows = self._rows("SELECT * FROM purges ORDER BY purge_id", ())
        return tuple(
            PurgeRecord(
                cutoff=r["cutoff"],
                retention_days=int(r["retention_days"]),
                calls=int(r["calls"]),
                turns=int(r["turns"]),
                decisions=int(r["decisions"]),
                agreements=int(r["agreements"]),
                json_files=int(r["json_files"]),
            )
            for r in rows
        )

    # -- retention ---------------------------------------------------------

    def purge_older_than(self, days: int | None = None) -> PurgeRecord:
        """Delete every call whose last activity predates ``days`` ago, with
        its turns, decisions, agreement, and exported agreement JSON.

        Explicit only. Nothing calls this on open, on close, or on a timer:
        deleting evidence is the one operation this log must never perform as a
        side effect of starting up, and ``collector-purge`` exists so that it
        is somebody's decision with a date on it. ``days`` defaults to
        ``retention_days()`` — three years, per Regulation F.

        The foreign keys in SCHEMA are documented and deliberately unenforced
        (see the module docstring), so there is no ON DELETE CASCADE to lean
        on: the dependents are deleted explicitly, in the same transaction as
        the parent, or the log keeps turns belonging to a call that is gone.

        The purge is itself audited — counts and cutoff into ``purges``, which
        no purge deletes — and the ``audit_chain`` entries of the deleted rows
        are left in place, so the chain still verifies and the gap is visible.
        """
        window = retention_days() if days is None else days
        cutoff = (datetime.now(UTC) - timedelta(days=window)).isoformat()

        def _purge() -> PurgeRecord:
            sidecars = [
                r["json_path"]
                for r in self._conn.execute(
                    f"SELECT json_path FROM agreements WHERE call_id IN ({_EXPIRED_CALLS})",
                    (cutoff,),
                )
            ]
            # Files first: the sidecar is a copy of `agreements.record_json`,
            # so a crash between here and the commit leaves a file that can be
            # re-exported. The reverse order loses the rows and leaves the
            # transcripts lying in the directory, which is the failure that
            # matters.
            json_files = _remove_sidecars(sidecars)
            with self._conn:
                counts = {
                    table: self._conn.execute(
                        f"DELETE FROM {table} WHERE call_id IN ({_EXPIRED_CALLS})", (cutoff,)
                    ).rowcount
                    for table in ("turns", "decisions", "agreements")
                }
                calls = self._conn.execute(
                    "DELETE FROM calls WHERE COALESCE(ended_at, started_at) < ?", (cutoff,)
                ).rowcount
                record = PurgeRecord(
                    cutoff=cutoff,
                    retention_days=window,
                    calls=calls,
                    turns=counts["turns"],
                    decisions=counts["decisions"],
                    agreements=counts["agreements"],
                    json_files=json_files,
                )
                self._conn.execute(
                    "INSERT INTO purges (cutoff, retention_days, calls, turns, decisions,"
                    " agreements, json_files, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.cutoff,
                        record.retention_days,
                        record.calls,
                        record.turns,
                        record.decisions,
                        record.agreements,
                        record.json_files,
                        utc_now(),
                    ),
                )
                # Chained like any other row: a purge record that can be edited
                # to say it deleted nothing is not a record of anything.
                self._append_chain("purges", self._last_rowid())
            # Outside the transaction — VACUUM cannot run inside one. Without
            # it the deleted transcripts stay legible in the file's free pages,
            # which is not a deletion. It rebuilds the tables while preserving
            # rowids, which the chain's row references depend on; the test
            # "the chain still verifies after a purge" is the canary if a
            # future SQLite ever stops preserving them.
            self._conn.execute("VACUUM")
            self._secure_wal_sidecars()  # VACUUM and the checkpoint rebuild them
            _secure_file(self.db_path)
            self._checkpoint()
            return record

        return self._run(_purge)

    # -- lifecycle ---------------------------------------------------------

    def _checkpoint(self) -> None:
        """Force the WAL out to the database file.

        Only ever at the end of something — never per row. `synchronous =
        NORMAL` (see _PRAGMAS) exists precisely so the turn loop does not fsync
        several times a turn, and a checkpoint per insert would spend that
        budget on the voice critical path.
        """
        self._conn.execute("PRAGMA wal_checkpoint(FULL)")

    def close(self) -> None:
        def _close() -> None:
            # The last chance to get the WAL onto disk under the store's own
            # control, and the only one a call that was abandoned mid-turn ever
            # gets: no CallEnded, no agreement, nothing else to trigger it.
            self._checkpoint()
            self._conn.close()

        self._run(_close)
        self._executor.shutdown(wait=True)

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


def _payload_hash(table: str, row: sqlite3.Row) -> str:
    """SHA-256 over the row, canonically serialized.

    Canonical means: the table name plus every column, keys sorted, no
    incidental whitespace. Two runs hashing the same row have to produce the
    same digest, or the chain reports tampering every time it is checked and
    nobody looks at it twice.
    """
    # sqlite3.Row iterates its *values*, so `.keys()` here is not the redundant
    # call SIM118 takes it for.
    columns = {key: row[key] for key in row.keys()}  # noqa: SIM118
    canonical = json.dumps(
        {"table": table, "row": columns},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _entry_hash(prev_hash: str, payload_hash: str) -> str:
    return hashlib.sha256((prev_hash + payload_hash).encode("utf-8")).hexdigest()


def _secure_dir(path: Path) -> None:
    """Create ``path`` if it is absent, and make it owner-only.

    Raises rather than carrying on when it cannot: the container runs as an
    unprivileged UID (Dockerfile), so a data volume mounted with the wrong
    owner arrives here — and a collector that keeps taking calls while writing
    world-readable transcripts into it is the exact failure this prevents.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, _DIR_MODE)
    except PermissionError as exc:
        raise PermissionError(
            f"cannot restrict {path} to {_DIR_MODE:o}: the audit log holds verbatim call"
            " transcripts and must not be world-readable. Give the process's own UID"
            " ownership of this directory (the image runs as UID 10001)."
        ) from exc


def _secure_file(path: Path) -> None:
    if path.exists():
        os.chmod(path, _FILE_MODE)


def _remove_sidecars(paths: Iterable[str | None]) -> int:
    """Delete the exported agreement JSON for purged calls. ``json_path`` is
    nullable, and a file an operator already moved is not an error."""
    removed = 0
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw)
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def purge_main(argv: list[str] | None = None) -> int:
    """``collector-purge`` — delete calls past the retention window.

    A console script and nothing else. Purging is destructive and irreversible,
    so it is an operator's deliberate act on a schedule they own, never a
    process lifecycle hook (see ``AuditStore.purge_older_than``).
    """
    parser = argparse.ArgumentParser(
        prog="collector-purge",
        description="Delete calls older than the retention window, with their turns,"
        " decisions, agreements, and exported agreement JSON. Destructive.",
    )
    parser.add_argument("--db", type=Path, default=default_db_path())
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"Retention window (default: ${RETENTION_DAYS_ENV_VAR}, else {DEFAULT_RETENTION_DAYS}"
        " — three years, the Regulation F floor).",
    )
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"no log at {args.db}", file=sys.stderr)
        return 1
    with AuditStore(args.db) as store:
        record = store.purge_older_than(args.days)
    print(
        f"purged {record.calls} calls ended before {record.cutoff} "
        f"({record.retention_days}d): {record.turns} turns, {record.decisions} decisions, "
        f"{record.agreements} agreements, {record.json_files} json files"
    )
    return 0
