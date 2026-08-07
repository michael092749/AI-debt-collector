"""Regression tests for AuditStore's cross-thread sqlite3 safety.

AuditStore's sqlite3 connection has ``check_same_thread=True`` (stdlib
sqlite3's default). In production, ``voice_app.py`` constructs ``AuditStore``
once on the asyncio event-loop thread but then drives every store-touching
call -- ``open_call()``, ``turn()``, ``close()`` -- through
``asyncio.to_thread(...)``, which runs on asyncio's shared
``ThreadPoolExecutor`` and does NOT guarantee the same OS thread across
separate calls. That made the very first audit write raise
``sqlite3.ProgrammingError``, before a single audit event or agreement was
ever written on a voice call.

The fix (``AuditStore._run`` in ``store.py``) gives ``AuditStore`` its own
dedicated single-worker ``ThreadPoolExecutor`` that owns the sqlite3
connection for the store's entire life; every touch of ``self._conn`` is
routed through that one worker thread and blocked on, regardless of which
thread the caller is on.

These tests use explicit ``threading.Thread``, not ``asyncio.to_thread``:
asyncio's shared pool can nondeterministically reuse the same OS thread
across two calls and produce a false pass. An explicit, separate thread
deterministically reproduces the exact failure geometry.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from collector.audit import (
    AuditStore,
    CallEnded,
    CallOutcome,
    CallStarted,
    ConsumerConfirmation,
    DecisionRecorded,
)
from collector.decision_engine import Verdict, validate_offer
from collector.money import Money
from collector.negotiation import NegotiationState
from collector.offers import Cadence, ConsumerProposal, Installment, Offer, Tier
from collector.policy import PolicyConfig

CALL_ID = "CALL-THREAD-0001"

SETTLEMENT = Offer(
    tier=Tier.SETTLEMENT,
    installments=(
        Installment(Money("250.00"), 0),
        Installment(Money("250.00"), 30),
        Installment(Money("300.00"), 60),
    ),
    cadence=Cadence.MONTHLY,
)


def _run_on_new_thread(fn: Callable[[], None]) -> None:
    """Run ``fn()`` on a brand-new, dedicated thread and block for it,
    re-raising any exception on the calling (test) thread.

    A join timeout turns a real deadlock into a fast test failure instead of
    a hung CI job.
    """
    caught: list[BaseException] = []

    def target() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            caught.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive(), "worker thread did not finish -- possible deadlock"
    if caught:
        raise caught[0]


@pytest.fixture
def store(tmp_path: Path) -> AuditStore:
    # Constructed on the test's own (main) thread, exactly as voice_app.py
    # constructs AuditStore on the asyncio event-loop thread.
    with AuditStore(tmp_path / "collector.db", json_dir=tmp_path / "agreements") as s:
        yield s


@pytest.fixture
def policy() -> PolicyConfig:
    return PolicyConfig.default()


@pytest.fixture
def accepted(policy: PolicyConfig) -> Verdict:
    """$800 over three monthly payments -- accepted."""
    proposal = ConsumerProposal(Money("800"), 3, Cadence.MONTHLY)
    return validate_offer(proposal, NegotiationState.opening(policy), policy)


def test_record_from_a_different_thread_than_the_constructor(store: AuditStore) -> None:
    """The bug, reproduced: AuditStore() on one thread, .record() on another,
    raised sqlite3.ProgrammingError -- confirmed against the unfixed store.py
    before this fix landed (see PR description / task report for the
    traceback). Assert on the actual row content, not just "no exception":
    money must survive the thread hop as an exact TEXT string.
    """
    _run_on_new_thread(
        lambda: store.record(
            CallStarted(
                call_id=CALL_ID,
                account_ref="ACCT-1",
                consumer_ref="CONS-1",
                original_balance=Money("800.00"),
                channel="voice",
                at="t0",
            )
        )
    )
    _run_on_new_thread(
        lambda: store.record(
            CallEnded(call_id=CALL_ID, outcome=CallOutcome.AGREED, turn_count=1, at="t1")
        )
    )

    rows = store._rows("SELECT * FROM calls WHERE call_id = ?", (CALL_ID,))
    assert len(rows) == 1
    assert rows[0]["original_balance"] == "800.00"
    assert rows[0]["outcome"] == CallOutcome.AGREED.value


def test_finalize_agreement_from_a_different_thread_does_not_deadlock(
    store: AuditStore, accepted: Verdict
) -> None:
    """finalize_agreement() calls four already-wrapped methods (decisions(),
    guardrail_events(), escalations(), record_agreement()) sequentially from
    its caller's thread. Driving a whole call sequence from one
    non-constructor thread is the realistic voice-pipeline shape
    (asyncio.to_thread eventually calling close(), which finalizes), and is
    the scenario that would wedge forever if AuditStore._run's reentrancy
    guard were wrong: with max_workers=1, a second submit from inside the
    one worker thread while it is already busy running this call would block
    on a queue only that same, currently-busy thread could ever drain.
    """
    confirmation = ConsumerConfirmation(
        confirmed=True, utterance="Yes, let's do that.", turn_index=1, at="t1"
    )

    def call_sequence() -> None:
        store.record(
            CallStarted(
                call_id=CALL_ID,
                account_ref="ACCT-1",
                consumer_ref="CONS-1",
                original_balance=Money("800.00"),
                channel="voice",
                at="t0",
            )
        )
        store.record(
            DecisionRecorded(
                CALL_ID,
                0,
                ConsumerProposal(Money("800"), 3, Cadence.MONTHLY),
                accepted,
                at="t0d",
            )
        )
        store.finalize_agreement(
            call_id=CALL_ID,
            final_offer=SETTLEMENT,
            authorizing_verdict=accepted,
            confirmation=confirmation,
            at="t1",
        )

    _run_on_new_thread(call_sequence)

    record = store.agreement(CALL_ID)
    assert record is not None
    assert record.call_id == CALL_ID
    assert record.confirmation.confirmed
    assert record.total == SETTLEMENT.total


def test_close_from_a_different_thread_does_not_hang_or_raise(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "c.db", json_dir=tmp_path / "agreements")
    _run_on_new_thread(store.close)
