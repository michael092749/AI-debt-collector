# Plan — Decision Engine (SPEC.md build order, steps 1–2)

**Spec:** `../../SPEC.md` · **Scope:** value layer + `decision_engine.py`, test-first
**Out of scope:** guardrails, audit, LLM, agent loop, voice (steps 4–9)

---

## 1. Dependency graph

```
                       money.py
                    (Decimal money type)
                          │
                          ▼
                       offers.py
        (Tier, Installment, Schedule, Offer, ConsumerProposal)
                    │             │
          ┌─────────┘             └─────────┐
          ▼                                 ▼
      policy.py                      negotiation.py
  (constants, PolicyConfig)      (NegotiationState — minimal)
          │                                 │
          └──────────────┬──────────────────┘
                         ▼
                 decision_engine.py
              validate_offer() -> Verdict
```

Import direction is one-way, top to bottom. `decision_engine.py` is a leaf: nothing imports
back into it, and it imports nothing from `llm/`, `agent.py`, `audit/`, or `voice_app.py`.

### 1.1 Correction to SPEC §11

The spec's build order lists `negotiation.py` at step 3, but `validate_offer()` takes a
`NegotiationState` (SPEC §4.1) — step 2 cannot compile without it. **Resolution:** a minimal
`NegotiationState` moves into step 2 carrying only what the engine reads:

```python
@dataclass(frozen=True)
class NegotiationState:
    ladder_floor: Tier  # best tier still on the table (A4 monotonicity)
    signaled_capacity: Money | None  # what the consumer said they can afford
    offers_made: tuple[Offer, ...]
```

Concession-tracking behavior, turn counting, and ladder-advance rules stay at step 3.

### 1.2 Slicing rationale

The spec's "value layer, then engine" is a horizontal slice. Re-cut vertically: every task
below delivers a complete `ConsumerProposal → Verdict` path that runs and is tested. T1
builds the thinnest possible end-to-end path; each later task adds one tier or one rule
class through the full stack.

---

## 2. Tasks

### T1 — Thinnest end-to-end path: accept a full payment
**Builds:** `money.py`, `offers.py`, `policy.py`, minimal `negotiation.py`, `decision_engine.py` skeleton

The first complete proposal→verdict path. A consumer offering the full $1,000 in one payment
gets `outcome="accept"`, `tier=PAY_IN_FULL`, and a populated `conditions` trail.

**Acceptance criteria**
- `Money` wraps `Decimal`; construction from `float` raises `TypeError`
- `validate_offer(full_payment, fresh_state, policy)` → `outcome="accept"`, `tier=PAY_IN_FULL`
- `Verdict.conditions` is non-empty and contains every rule evaluated, passing ones included
- `Verdict` and all value objects are frozen dataclasses

**Verify:** `uv run pytest tests/test_decision_engine.py -k full_payment` green

---

### T2 — Rejection + the counter machinery
**Builds:** floor rules, `rationale_code`, counter construction

A lowball proposal is rejected with a structured reason *and* a legal counter-offer. This is
the brief's core requirement — "validated **and countered** by logic outside the agent."

**Acceptance criteria**
- $50 and $200 proposals → `outcome="reject"`, `counter is not None`
- The failing `Condition` names the rule (`MIN_PAYMENT` / `SETTLEMENT_FLOOR`) with `actual`
  and `limit` populated as human-readable strings
- The returned `counter` itself satisfies every §4.3 invariant
- `rationale_code` is drawn from a closed enum — never a free-text string

**Verify:** `uv run pytest tests/test_decision_engine.py -k reject` green

---

### T3 — Tier 2: downpayment + one, with the downpayment maximized
**Acceptance criteria**
- Total is exactly $1,000 across exactly 2 installments, no discount (A3)
- Given signaled capacity `c`, `downpayment == clamp(c, $250, $750)`
- Capacity of $900 → downpayment capped at $750 (remainder must clear the $250 floor)
- Capacity of $100 → T2 is infeasible; engine falls through to a lower tier, never emits a
  sub-floor installment

**Verify:** `uv run pytest tests/test_decision_engine.py -k downpayment` green

> **CHECKPOINT 1** — floors, counters, and two tiers working. Review before continuing.

---

### T4 — Tier 3: settlement, ≤20% off, ≤3 payments
**Acceptance criteria**
- Accepted total ∈ `[$800.00, $1000.00)`; $799.99 rejected, $800.00 accepted (boundary)
- ≤ 3 installments, each ≥ $250, duration ≤ 3 months (A2)
- **$800 over 3 payments structures as 250/250/300, never 200/300/300** (§2.3 rule 3)
- Installments sum to the total exactly — no Decimal rounding drift

**Verify:** `uv run pytest tests/test_decision_engine.py -k settlement` green

---

### T5 — Tier 4: payment plan, no discount, and the impossible-schedule counter
**Acceptance criteria**
- Total is exactly $1,000; cadence ∈ {weekly, biweekly, monthly}; duration ≤ 3 months
- **A weekly-over-3-months request (~13 payments of $77) is rejected**, and the counter is a
  legal structure — 4 weekly payments of $250 completing in ~1 month (§2.3 rule 1)
- No accepted plan anywhere exceeds 4 installments
- Cadence outside the allowed set is rejected, not silently coerced

**Verify:** `uv run pytest tests/test_decision_engine.py -k plan` green

---

### T6 — Invariants + purity, enforced structurally
The safety net. Rules proven across *all* inputs rather than the examples above.

**Acceptance criteria**
- Property tests: for any generated input, a non-reject `Verdict` satisfies every §4.3
  invariant (installment ≥ floor, total ≥ settlement floor, count ≤ tier max, duration ≤ 3
  months, cadence allowed, installments sum exactly)
- Float-money test: no `float` reaches any monetary field anywhere in the module
- **Import-purity test**: `decision_engine` imports nothing from `llm`, `agent`, `audit`,
  `voice_app`, `sqlite3`, `datetime.now`, or the network. Asserted by AST inspection so it
  cannot silently rot.
- Determinism: same `(proposal, state, policy)` → byte-identical `Verdict`, repeatedly
- `uv run mypy --strict src/collector/decision_engine.py` clean

**Verify:** `uv run pytest && uv run ruff check . && uv run mypy src`

> **CHECKPOINT 2** — engine complete and certified. Review before step 3 (negotiation ladder).

---

## 3. Risks

| Risk | Mitigation |
|---|---|
| Decimal rounding leaves installments not summing to total | Largest-remainder allocation; exact-sum assertion in T4/T6 |
| Counter algorithm silently violates A4 monotonicity | `ladder_floor` read on every call; dedicated T3 test |
| Tier fall-through picks a worse tier than necessary | Tiers evaluated strictly in preference order; asserted in T3/T5 |
| "Pure" engine acquires an I/O import later | AST-based import test in T6, not a code-review convention |

## 4. Definition of done (all tasks)

Tests pass · no regressions · `ruff` + `mypy --strict` clean on the engine ·
behavior confirmed against §2.3's four structural consequences.
