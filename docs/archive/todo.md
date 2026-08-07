# TODO — Decision Engine (SPEC steps 1–2)

Plan: `plan.md` · Spec: `../../SPEC.md`
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
- [x] **Step 4 — `guardrails/`: prohibited persuasion, numeric authorization, disclosures, escalation** (124 tests)
- [x] **Step 5 — `audit/`: SQLite + agreement record with decision trail** (36 tests)
- [x] **Step 6/7 — `llm/mock_client.py`, `tools.py`, `agent.py`, `text_app.py`, `llm/anthropic_client.py`** (43 tests)
- [x] **Step 8 — `tests/evals/`: adversarial certification**
  - [x] `personas.py` — all eight SPEC §7.2 personas, each with live `instructions` and a
        deterministic `fallback_script`
  - [x] `simulator.py` — live Claude-as-consumer when a key is configured, scripted replay
        otherwise, decided once by whether `AnthropicClient()` constructs
  - [x] `test_scenarios.py` — transcript-level invariants (no prohibited phrase, no
        unauthorized figure, no sub-floor agreement, disclosure ordering, escalation
        where expected/forbidden, identity gating, overall compliance) parametrized
        over all eight personas
  - [x] Verified non-vacuous: `evasive` produces 6 agent lines with identity never
        confirmed (real assertions, not an early `return`); every persona reaches a
        terminal outcome offline
  - **Caveat, not a defect**: offline (no key) the consumer is `MockLLMClient`-driven
    end-to-end, and the mock only ever reads figures back from engine-authored offers —
    so `test_no_unauthorized_figure_is_ever_spoken` is structurally guaranteed to pass
    without a key, not just observed to pass. The invariant only gets genuinely
    adversarial pressure in live mode (`ANTHROPIC_API_KEY` set). README's Tier-2
    description already says this; noted here so it isn't mistaken for full offline
    certification.
- [x] **Step 9 — `voice_app.py`: LiveKit pipeline** — `llm_node` overridden to bypass the
      framework's own LLM/function-calling and hand the transcribed utterance straight to
      `NegotiationAgent.turn()`; Deepgram STT / Cartesia TTS are the framework's. Verified:
      clean `mypy --strict` (22 source files) and `import collector.voice_app` succeeds.
      Not verified: an actual LiveKit room/call (needs `LIVEKIT_*` + `DEEPGRAM_API_KEY` +
      `CARTESIA_API_KEY`, none configured in this environment).
- [x] **Step 10 — README** — architecture, policy table, commands, testing strategy,
      setup, assumptions, boundaries.

**337 tests passing. `ruff` clean. `mypy --strict` clean across all 22 source files.**
All ten SPEC steps complete.
