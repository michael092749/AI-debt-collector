# TODO — Decision Engine (SPEC steps 1–2)

Plan: `plan.md` · Spec: `../SPEC.md`
**Status: engine complete.** 48 tests green · ruff clean · `mypy --strict` clean on all 5 core modules.

## Done

- [x] **T1** — Thinnest path: accept full payment
  - [x] `money.py` — Decimal-backed `Money`, rejects `float`
  - [x] `offers.py` — `Tier`, `Cadence`, `Installment`, `Offer`, `ConsumerProposal`
  - [x] `policy.py` — §2.1 constants + `PolicyConfig`, all limits derived not hardcoded
  - [x] `negotiation.py` — minimal `NegotiationState` (per plan §1.1)
  - [x] `decision_engine.py` — `validate_offer()`, `Verdict` + `Condition` + `RuleId`
  - [x] Test: full $1,000 → accept, `tier=PAY_IN_FULL`, conditions populated

- [x] **T2** — Reject + counter machinery
  - [x] `MIN_PAYMENT` / `SETTLEMENT_FLOOR` rules emit structured `Condition`s
  - [x] `RationaleCode` closed enum
  - [x] Counter construction; returned counter self-satisfies §4.3
  - [x] Test: $50/$200/$500/$799.99 → reject, each with a legal counter
  - [x] Added: `_effective_capacity` infers capacity from the proposal's own smallest
        installment when none is stated

- [x] **T3** — Tier 2: downpayment + one, maximized
  - [x] `downpayment == clamp(capacity, $250, $750)`
  - [x] Test: capacity $900 → $750 cap; capacity $100 → falls through, no sub-floor split

- [x] **CHECKPOINT 1** — floors, counters, two tiers ✓

- [x] **T4** — Tier 3: settlement ≤20% off, ≤3 payments
  - [x] Boundary: $799.99 reject / $800.00 accept
  - [x] Every installment ≥ $250 (200/300/300 illegal) — see note below
  - [x] Installments sum exactly (no Decimal drift)

- [x] **T5** — Tier 4: payment plan + impossible-schedule counter
  - [x] Weekly-over-3-months rejected → counter 4 weekly × $250, duration 21 days
  - [x] No accepted plan exceeds 4 installments (verified for counts 5–14)
  - [x] Disallowed/incoherent cadence corrected, never silently accepted

- [x] **T6** — Invariants + purity
  - [x] ~4,900-proposal sweep over all §4.3 invariants; verified non-vacuous
        (680 accepts across all 4 tiers, 2,080 rejects, 40 counters)
  - [x] No-float-money tests
  - [x] AST import-purity test across all 5 core modules
  - [x] Determinism + frozen/hashable verdict tests
  - [x] `mypy --strict` clean

- [x] **CHECKPOINT 2** — engine complete and certified ✓

### Deviation from plan (T4)
Plan specified the $800/3 settlement split as exactly `250/250/300`. That over-specified
SPEC §2.3 rule 3, whose actual rule is "every installment ≥ $250". Implemented an even
split (`266.67/266.67/266.66`) which satisfies the rule and is easier for a consumer to
agree to; the floor-pinned split is used only when an even one would dip below $250.
Test asserts the rule, not the example.

## Deferred (SPEC steps 3–10)

- [x] **Step 3 — full `negotiation.py`: ladder advance, concession tracking** (17 tests)
  - [x] `CallOutcome` + `Round`; every transition returns a new state
  - [x] Concessions are earned: `advance_ladder()` is a no-op without a recorded refusal
  - [x] Exactly one tier per refusal; bottoms out at `PAYMENT_PLAN`; never moves back up (A4)
  - [x] Terminal states (`AGREED` / `ESCALATED` / `NO_AGREEMENT`) refuse further negotiation
  - [x] `escalate()` deliberately reachable from any state — safety valve, per A6
  - [x] Round cap (`max_negotiation_rounds=8`, new in `PolicyConfig`) — unbounded
        haggling is badgering however politely phrased
- [ ] Step 4 — `guardrails/`: prohibited persuasion, numeric authorization, disclosures, escalation
- [ ] Step 5 — `audit/`: SQLite + agreement record with decision trail
- [ ] Step 6 — `llm/mock_client.py`, `tools.py`, `agent.py`, `text_app.py`
- [ ] Step 7 — `llm/anthropic_client.py`
- [ ] Step 8 — `tests/evals/`: adversarial certification
- [ ] Step 9 — `voice_app.py`: LiveKit pipeline
- [ ] Step 10 — README
