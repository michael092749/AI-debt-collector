# Plan — negotiation-ladder defects from the 2026-08-07 transcript review

Scope approved 2026-08-08. Baseline: 925 passed on `integration` @ `76dea1f`.

Items 1 and 2 of the original review list (dollars-and-cents composition, capacity
ratchet) were verified **already fixed** by `76dea1f` and are not in this plan.
Item 3 (echoing consumer-stated figures) is already implemented as
`consumer_stated_money()`; the punchy-opener and no-narration prompt rules are
already present at `llm/base.py:240` and `llm/base.py:179`.

Everything below was reproduced against current HEAD before being written down.

---

## Task 1 — A repeated proposal must not re-buy a ladder step

**Defect (reproduced).** `propose_offer` then the *same* `$100 / 1 payment /
immediate` five times walks the ladder PAY_IN_FULL → DOWNPAYMENT_PLUS_ONE →
SETTLEMENT → PAYMENT_PLAN. The $800 settlement floor is disclosed on the third
utterance of the same number. The consumer never negotiated.

`tools.py:584` steps down whenever `_already_ruled_on(state, proposal)` — correct
for *a* repeat (they heard the counter and held), but it re-fires on every later
repeat of the same terms, so the ladder is a free elevator.

**Acceptance criteria.**
- Repeating identical terms earns **at most one** ladder step across the whole call.
- Repeating $100 five times never reaches SETTLEMENT, so the floor is never spoken.
- A consumer naming *genuinely new* terms still earns their step (A7 preserved).
- `_held_to` in the engine keeps working: a consumer who holds to legal terms
  through a counter still gets them accepted.

**Verify:** new test in `tests/test_engine_invariants.py`; full suite green.

---

## Task 2 — Never counter below an otherwise-legal proposal's total

**Defect (reproduced).** With the ladder at SETTLEMENT, a fully legal
`$1,000 / 4 monthly / $250 each` proposal returns
`outcome=counter, rationale=PREFERRED_TIER_AVAILABLE` with an **$800** counter.
The agent tries to talk the consumer down $200 from their own full-balance offer.

Cause: the `LADDER` condition (`decision_engine.py:199-204`) compares tier rank
only, never totals, and `_tier_total` ranks an $800 settlement above a $1,000 plan.

**Acceptance criteria.**
- A proposal that fails **only** `LADDER` and whose total is `>=` the counter's
  total is never answered with a lower-total counter.
- The `_held_to` protections stay intact — a sharpened re-proposal
  ("settle at $900" → "$800, then") must still not walk us down.
- No change to proposals that fail a hard floor.

**Verify:** new test in `tests/test_decision_engine.py`; full suite green.

---

## Task 3 — Settlement opens above the floor

**Defect (reproduced).** `_tier_total` (`decision_engine.py:374-376`) returns
`policy.settlement_floor` for SETTLEMENT, so the first settlement offer *is* the
maximum authorized discount. There is no room left to concede within the tier.

**Approved change:** open the settlement tier above the floor and hold the floor
back for a later concession. `max_settlement_discount` (20%) remains the hard
ceiling — the opening discount is a new, smaller policy value.

**Acceptance criteria.**
- First settlement offer total is strictly above `settlement_floor` and at or below
  `original_balance`.
- The floor is still reachable as a later in-tier concession.
- No offer ever discounts more than `max_settlement_discount`.
- `_is_concession` still reports movement for the in-tier step.

**Verify:** new tests in `tests/test_decision_engine.py` + invariant that no
authorized offer breaches the floor; full suite green.

---

## Task 4 — Fallback must not repeat itself

**Defect.** `SAFE_FALLBACK_TEXT` (`rings.py:60`) is one static string returned
unconditionally at `rings.py:803`. Two consecutive guard trips speak it verbatim
twice, which reads as "the robot didn't hear me". LiveKit's prompting guide calls
for phrase variation across turns; nothing here varies.

`CONNECTIVE_TEXT` and `MAX_REGENERATION_STRIKES` already exist and are not changed.

**Acceptance criteria.**
- The same recovery line is never spoken twice in a row.
- Selection is deterministic (audit chain is hash-sensitive — no RNG).
- Every rotation variant passes the numeric and prohibited-language guards.
- Escalation on a repeat trip does not re-open a standing offer for bargaining.

**Verify:** new tests in `tests/test_guardrails.py`; full suite green.

**STATUS: not landed — needs a decision.** Implemented and reverted. A
three-line `RECOVERY_LINES` rotation on a `fallbacks_spoken` counter works and
its own tests pass, but it collides with a deliberate, tested contract in
`TestRepeatedFallbacksEscalateToTheStandingOffer`
(`tests/test_instrumentation.py`), which already carries a partial fix:

- trip 1 asks the open question, trip 2 restates the standing offer
  (`agent.py:1386`, `_consecutive_fallbacks`), trips 3+ return to the open
  question — `lines[2:] == [SAFE_FALLBACK_TEXT, SAFE_FALLBACK_TEXT]` (:1708)
  asserts that verbatim repeat directly.
- `:1697` asserts "the count restarts, it does not accumulate" after a turn
  that speaks, which contradicts a counter that never resets.

Two questions to settle before rebuilding it:
1. Should trips 3+ rotate, or keep repeating the open question?
2. Should the rotation reset after a successful turn, or persist for the call?

Published guidance is one-sided (never repeat a recovery prompt verbatim;
taper and escalate within 2-3 strikes), which argues for rotate + persist —
but that rewrites eight assertions in compliance-adjacent tests, so it is a
decision to take deliberately rather than absorb into a green suite.

---

## Out of scope, found during research — flagged, not built

- `voice_app.py` `TURN_HANDLING` never sets `discard_audio_if_uninterruptible`,
  which defaults to `True`. With `allow_interruptions=False` on the disclosure,
  anything the consumer says during it is **discarded, not queued**.
- `asyncio.sleep(GREETING_PAUSE_SECONDS)` hand-rolls the SDK's
  `min_consecutive_speech_delay`.
- `pyproject.toml` floors `livekit-agents>=1.5.2` while 1.6.8 is installed and
  several used features post-date the floor.
