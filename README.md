# Voice Debt-Collection Negotiation Agent

A voice agent that calls a consumer 180+ days delinquent on a $1,000 debt and closes the
highest-value agreement they'll actually honor — without threats, false urgency, or invented
consequences.

> **The LLM talks. Deterministic code decides.**

The model handles understanding, phrasing, and rapport. Every amount, schedule, discount, and
concession is computed by a pure Python decision engine the model calls as a tool, mid-call. The
model may never originate a number — this is enforced mechanically, not requested in a prompt.

This README is the whole contract: the policy the engine enforces, the guardrails, the record it
writes, and how each of those is tested.

## What's actually graded

Three requirements, each satisfied by a specific, isolated piece of the system:

| Requirement | Where |
|---|---|
| Proposed amounts validated and countered by logic **outside the agent, mid-call** | `decision_engine.py` — pure function, called as a tool, returns a verdict with every evaluated condition |
| Compliance guardrails that fail regardless of conversion | `guardrails/` — three rings enforced in code, including a numeric guard that blocks any unauthorized figure before it reaches TTS |
| The final agreement, logged | `audit/` — SQLite + a standalone JSON record with the full decision trail |

The test for the decision record: *"Show me the decision record. If it's a transcript, the model
decided. If it's evaluated conditions and a policy path, the engine did."*

## Quick start

```bash
uv sync                            # everything below needs no keys at all
uv run pytest                      # 747 tests, offline
uv run collector-text              # negotiate in a terminal, mock model, no keys
```

Try it:

```
you> Yes, speaking.
you> I can do $50 a month.
you> No, too much.
you> Okay.
```

Expect: an AI disclosure in the opening line, the Mini-Miranda immediately after identity is
confirmed, every dollar figure traceable back to an `[engine]` line with `--verbose`, and a
closed agreement no lower than $800 total / $250 per payment.

## The policy

Every *policy* limit is a field or property of `PolicyConfig` in `policy.py`, and nowhere
else. (The turn-loop caps are separate and live in `agent.py`: `MAX_TOOL_ROUNDS = 4`,
`MAX_REGENERATION_STRIKES = 2`.)

| Field | Value | Source |
|---|---|---|
| `original_balance` | $1,000.00 | brief |
| `min_payment_pct` | 0.25 | brief |
| `min_payment` | $250.00 | derived — 25% of the *original* balance (**A1**) |
| `max_settlement_discount` | 0.20 | brief |
| `settlement_floor` | $800.00 | derived |
| `max_installments` | 4 | derived — balance ÷ floor |
| `max_plan_days` | 92 | brief — "3 months max" |
| `max_settlement_payments` | 3 | brief |
| `allowed_cadences` | immediate, weekly, biweekly, monthly | brief |
| `max_negotiation_rounds` | 8 | not the brief — past this the call closes out, because an
unbounded negotiation is badgering however politely it is phrased |

Outcome tiers, in preference order:

| Tier | Total | Payments | Per-payment | Duration |
|---|---|---|---|---|
| T1 — pay in full | $1,000 | 1 | $1,000 | immediate |
| T2 — downpayment + one | $1,000 | 2 | each ≥ $250, first maximized | ≤ 3 months |
| T3 — settlement | $800–$999.99 | ≤ 3 | each ≥ $250 | ≤ 3 months |
| T4 — payment plan | $1,000 | ≤ 4 | each ≥ $250 | ≤ 3 months |

Four structural rules fall out of those constants. They are not separate policy — they are
arithmetic, and each one is a test:

1. **Max 4 payments, ever.** $1,000 / $250 = 4. "Weekly over a year" (≈52 payments) or "weekly
   over 3 months" (≈13) is *impossible*, so the engine rejects it. What it counters *with* is
   the current ladder position, not the nearest legal structure (**A7**): 4 weekly payments of
   $250 is a T4 counter, and the ladder only reaches T4 after three earned concessions.
2. **$800 is the hard minimum total.** Below it, rejected at every tier. There is no path to
   $500.
3. **The per-payment floor binds harder than the discount.** Every instalment of an $800
   settlement must clear $250, so 200/300/300 is illegal. The engine splits evenly —
   266.67/266.67/266.66 — and leads with a capacity the consumer named when that still
   leaves every later instalment above the floor: $300 signalled gives 300/250/250.
4. **T2 maximizes the downpayment.** Given signalled capacity `c`, the counter is
   `downpayment = clamp(c, 250, 750)` with the remainder in one further payment.

The agent walks *down* this ladder as the consumer pushes back and never back up. Concessions
are earned by a refusal on record, never volunteered.

### Assumptions

The brief is ambiguous in places. Each reading below is resolved in code, and the source cites
these by ID.

| | Assumption |
|---|---|
| **A1** | "25% minimum" is 25% of the *original* $1,000 balance, not of the agreed total — a fixed $250 floor at every tier. The stricter reading, and the only one implemented. It is why a plan tops out at 4 installments and why $50 → $200 lowballs never move the floor. |
| **A2** | Settlement duration is capped at 3 months. The brief caps T3 payment *count* and is silent on duration; letting a discounted settlement stretch a year while a full-balance plan caps at 3 months would be incoherent. |
| **A3** | Discounts apply only at T3. T2 and T4 are full-balance by definition. |
| **A4** | Concessions are monotonic. The ladder runs T1→T4 and never back up, and a withdrawn term is never re-offered. Negotiation integrity, not a brief requirement. |
| **A5** | No payment capture. The agent closes and logs an *agreement*; it never touches a card or bank account. Keeps PCI scope at zero. |
| **A6** | Escalation terminates the call. No human queue in this build: on dispute, hardship, distress, attorney, or cease-and-desist the agent stops negotiating, says a human will follow up, and writes an `escalation` record. A real deployment swaps in a warm transfer; the trigger logic is identical. |
| **A7** | Legal is not the same as agreeable. A proposal that clears every floor but sits below the current ladder position is *countered* at the ladder rather than accepted (`LADDER` condition, rationale `PREFERRED_TIER_AVAILABLE`). Without this the first figure a consumer names decides the call. Exception: a tier they already proposed, that was countered for the ladder alone, on terms no worse than before — they have heard the better ask and held to legal terms. Known limit: the engine is turn-free, so a model that validates the same proposal twice inside one turn satisfies that repeat without the counter ever reaching the consumer's ear. `MAX_TOOL_ROUNDS` bounds the repeats inside a turn and `max_negotiation_rounds` the call as a whole; closing it properly needs a turn index the engine deliberately does not have. |

## Architecture

```
consumer speech
      │
      ▼
┌─────────────┐   perceive → guard (inbound: escalation only, never blocks) ┐
│  agent.py   │                                                             │
│ turn loop   │   think → tool ──────► decision_engine.py (pure)            │
│             │              │         validate_offer() / build_counter()  │
│             │◄─────────────┘         returns Verdict: outcome, tier,     │
│             │                        every evaluated Condition, counter  │
│             │   guard (outbound: prohibited phrases, numeric              │
│             │   authorization, disclosure ordering) → speak               │
└─────────────┘
      │
      ▼
audit/store.py — SQLite + JSON, full decision trail per agreement
```

- **`decision_engine.py`** is pure: no I/O, no LLM, no clock, no imports from anything above it.
  Enforced by an AST test (`tests/test_engine_invariants.py`), not just convention. Its output is
  a `Verdict`: an outcome (`accept`/`counter`/`reject`), the tier, *every* rule evaluated with
  its actual and limit values, an engine-computed counter, and a stable rationale code the model
  phrases but never invents.
- **`tools.py`** is the whitelist. The model can only take the six actions defined there —
  `validate_consumer_offer`, `propose_offer`, `record_refusal`, `concede`, `confirm_agreement`,
  `end_call` — and every one returns `{"ok": false, ...}` rather than raising, because a bad
  cadence or a typo must not end a phone call.
- **`guardrails/`** — three rings:
  - **pre-call**: account loaded, calling window open, no cease-and-desist on file.
  - **during-call**, on every candidate agent sentence before it's spoken: prohibited persuasion
    (threats, false urgency, invented consequences), numeric authorization (every figure must be
    in the set the engine currently authorizes), and disclosure ordering (Mini-Miranda before
    substance, AI disclosure at open and on request). A trip blocks the turn and regenerates it
    with the violation named; two strikes falls back to a scripted, figure-free line.
  - **post-call**: scores the whole trace and writes the agreement.
- **`llm/`** — one `LLMClient` protocol, four implementations: `MockLLMClient` (scripted,
  deterministic, no key needed — what the whole test suite runs against), `AnthropicClient`
  (`claude-sonnet-5`, real reasoning, the certified path), and `OpenRouterClient` /
  `LiveKitClient`, two opt-in alternate routes sharing an OpenAI-shaped request mapper
  (`openai_shape.py`). The agent loop is written against the protocol only, so swapping
  between them changes nothing above `llm/`. Only the Anthropic route has been through the
  adversarial pass.
- **`text_app.py`** and **`voice_app.py`** are two transports over the identical core. Text is a
  terminal loop with no audio, no keys, no network. Voice is a LiveKit Agents worker — Deepgram
  STT and Cartesia TTS are the framework's; the turn itself is not. `voice_app.py` overrides the
  framework's `llm_node` to hand the transcribed utterance straight to the same
  `NegotiationAgent.turn()` the text app calls, bypassing LiveKit's own LLM/function-calling
  entirely. What reproduces in the terminal reproduces on the phone, because only the transport
  differs.

## Commands

| Command | Purpose |
|---|---|
| `uv sync` | Install (add `--extra dev` for pytest/ruff/mypy) |
| `uv run collector-text` | Interactive negotiation in a terminal — mock model, no keys |
| `uv run collector-text --claude --verbose` | Same call, real model, with the engine trace shown |
| `lk agent dev src/collector/voice_app.py` | LiveKit voice worker, dev mode — see below |
| `uv run pytest` | Tier-1 suite, offline — the `evals` marker is deselected by default |
| `uv run pytest -m evals tests/evals` | Tier-2 adversarial evals, explicitly |
| `uv run collector-agreements` | Dump every agreement record + its decision trail as JSON |
| `uv run collector-purge` | Delete calls past the retention window — destructive, see below |
| `uv run ruff check src tests && uv run mypy src` | Lint + strict types |

## Running the voice worker

```bash
cp .env.example .env               # fill in LiveKit + Anthropic keys
set -a; source .env; set +a        # `lk` reads the shell, not .env
lk agent dev src/collector/voice_app.py
```

`uv run collector-voice dev` still works, but the framework's own dev mode is deprecated in
favour of the `lk` CLI. `lk agent dev` won't auto-detect this entrypoint — it only looks for
`agent.py` / `src/agent.py` — so the path is given explicitly. The worker then waits to be
dispatched a job; it does not join a room on its own. Dispatch metadata that is present but
malformed raises rather than falling back to the fixture consumer: a live call must never
proceed under the wrong name.

**LiveKit Cloud recording is off by default**, and the default does not move. This project's
`AuditStore` is the compliance record, and nothing on the call collects consent for a second
copy of the consumer's audio held by anyone else. `COLLECTOR_VOICE_RECORDING` opts in per
deployment:

| Value | Uploaded to LiveKit Cloud |
|---|---|
| unset / `off` | **Default.** Nothing. |
| `diagnostics` | Pipeline traces and agent-server logs. No audio, no transcript. |
| `full` | Everything — audio, transcript, traces, logs (the SDK's own default). |

Use `full` only for a local call with no real consumer. Anything unrecognized records nothing.

## Storage and retention

`data/collector.db` is the evidence store: verbatim consumer and agent speech, every guardrail
trip, every decision, every agreement. Four things are true of it.

**Permissions.** The database, its `-wal`/`-shm` sidecars, and the exported agreement JSON are
created mode `0600` inside a `0700` directory. The store refuses to start if it cannot set that
— a data volume mounted with the wrong owner is a misconfiguration, not something to write
transcripts through.

**Encryption is the deployment's job, and it is not optional.** *The mounted data volume must be
on an encrypted filesystem* — dm-crypt/LUKS, an encrypted EBS volume, or the equivalent. There
is no SQLCipher and no field-level encryption here; that was decided deliberately, in favour of
OS permissions plus volume-level encryption. Say the limit plainly: `0600` separates accounts on
one host. **It does not protect the data from root, from a stolen or imaged disk, from a volume
snapshot, or from a backup.** Anything that reads the block device reads the calls.

**Tamper-evidence, not immutability.** Every insert appends an entry to `audit_chain` —
`entry_hash = sha256(prev_hash || payload_hash)` — in the same transaction as the row it covers.
`AuditStore.verify_chain()` re-walks it and returns the first broken link, so a row edited behind
the store's back is *detectable*. It is not prevented: anyone who can write to the file can still
change it. What they cannot do is change it and leave the digests agreeing without rewriting
every entry after it. Deletion is the acknowledged gap — a removed row leaves an orphaned chain
entry rather than a broken hash, because purging is legitimate (below) and must not read as
forgery.

**Retention.** `COLLECTOR_RETENTION_DAYS` defaults to **1095 days**. CFPB Regulation F
(12 CFR 1006.100) requires retaining evidence of compliance until three years after the last
collection activity on an account, so a shorter value is a records-retention decision someone
has to own. Nothing purges automatically — not on open, not on a timer. `collector-purge` is an
explicit command: it deletes expired calls with their turns, decisions, agreements, and agreement
JSON files, `VACUUM`s, and writes what it deleted (counts and cutoff) to the `purges` table,
which no purge deletes.

```bash
uv run collector-purge --days 1095     # or omit --days for $COLLECTOR_RETENTION_DAYS
```

Money is stored as an exact decimal string, never a float, everywhere — the database, the JSON
record, the span attributes. A float in a payment schedule is a compliance defect, not a
rounding nit, so `Money` refuses to be constructed from one at all.

## Testing

`uv run pytest` — **747 tests, no API key, no network.** `addopts = "-m 'not evals'"`
deselects the 73 tier-2 evals; run those explicitly with `uv run pytest -m evals
tests/evals`.

- **Tier 1** (`tests/`) — unit and property tests over the decision engine, guardrails, and
  negotiation ladder, plus the full agent loop against the mock client. Includes a ~4,900-case
  sweep asserting the total rule holds across the whole input space, and an AST test that fails
  if `decision_engine.py` ever acquires an I/O import.
- **Tier 2** (`tests/evals/`) — eight adversarial personas (lowballer, impossible-schedule, rage,
  evasive/silent, hardship, verbal dispute, jailbreaker, serial re-negotiator) driven by
  `claude-sonnet-5` for genuinely unpredictable pressure when a key is configured, with scripted
  transcripts as the offline fallback. Offline it is deterministic and green; against live
  models it is a real gate and can fail — a `coherence_judge` fail on the rage persona is a
  result, not a flake. Assertions run
  over the *whole transcript*: no prohibited phrase, no unauthorized figure, no agreement below
  the floors, disclosures fired in order, escalation where the persona calls for it, and the
  post-call compliance verdict itself. One transcript per persona is built once per session and
  shared by both eval modules.
- **LLM judges** (`tests/evals/test_judges.py`) — the invariants above are asserted with this
  project's own detectors, because a judge that is right *most* of the time is the wrong
  instrument for a rule that must hold *every* time. What a detector cannot see is tone and
  intent, so LiveKit's `JudgeGroup` runs alongside them with two of the eight built-in judges:
  `safety_judge` (unauthorized advice, improper disclosure, missed escalation, harmful language)
  and `coherence_judge`. Each was checked against a known-good transcript and a deliberately
  abusive one before being trusted. `relevancy_judge` was tried and removed: it *passed* a
  transcript containing arrest threats and a brushed-off disclosure of suicidal ideation (abuse
  is on-topic), then *failed* a clean lowballer call for "repeatedly ignoring the user's offers"
  — which is what a policy floor looks like from outside. The judges see only what the consumer
  heard, and the assertion is `none_failed` rather than `all_passed`, so a `maybe` on a scripted
  transcript is not a build break. Needs `LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`; skips cleanly
  without them.
- **Tier 3** (production sampling) — out of scope to run; the log schema supports it.

Verified at runtime, not just in tests: a consumer who signals $750 down closes T2 at
$750 + $250 (the downpayment clamp — a smaller signal, e.g. $500, closes just as legally at
$500 + $500); "I want to pay weekly for a year" is rejected, and countered at whatever tier the ladder
has reached — 4 weekly payments of $250 once it is at T4; a lowballer offering $50 then $200 is refused both
times and closes at the $1,000 floor; hardship and dispute both escalate immediately with no
agreement recorded.

## Observability

Every line the worker logs carries `call_id`, `room`, `account_ref`, `channel` and `llm_route`
as structured fields, so one call can be pulled out of a worker handling several. The consumer's
*name* is deliberately not among them: it buys nothing a reader of the audit store cannot get,
and logs leave the process by routes that carry none of the consent posture the store does.
Turn latency is measured from the framework's own metrics (`e2e_latency`, and `llm_node_ttft`,
which on this design is the whole seven-round-trip turn rather than time-to-first-token).

**Tracing is off by default and it fails loudly.** `COLLECTOR_TRACING` accepts `off`/unset and
`otlp` and nothing else: any other value raises at start-up rather than reading as "off", and
`otlp` without `OTEL_EXPORTER_OTLP_ENDPOINT` raises too, because the exporter's own default is
localhost and would silently black-hole every span.

With `otlp` set, the whole call is one trace: a `collections_call` root span with every model
call, tool call, engine verdict, guardrail trip and escalation under it, exported over OTLP/HTTP
to any compatible backend (Langfuse, Grafana Tempo, Honeycomb). The LiveKit Agents SDK's own
session spans are pointed at the same backend through
`livekit.agents.telemetry.set_tracer_provider`.

**No consumer data reaches a span.** No transcript, no debtor name, no dollar amount — the span
carries rule ids, outcome codes, token counts and latencies, plus `call_id` and `turn_index` to
find the row in the audit store that does hold the values. See `src/collector/tracing.py`.

## Setup

```bash
cp .env.example .env
```

| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | `--claude` mode, tier-2 evals with live adversarial pressure, and any real voice call |
| `COLLECTOR_MODEL` | Overrides the Anthropic model id (default `claude-sonnet-5`) |
| `COLLECTOR_LLM` | Which route the voice worker uses: `anthropic` (default), `openrouter`, `livekit` |
| `OPENROUTER_API_KEY` | `--claude`'s alternate route (`--openrouter`, `COLLECTOR_LLM=openrouter`) |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | `collector-voice` (transport, STT, TTS), the LLM judges, and `--livekit` / `COLLECTOR_LLM=livekit` |
| `COLLECTOR_DB_PATH` | Moving the audit log off the CWD-relative `data/` — set it to the encrypted volume |
| `COLLECTOR_RETENTION_DAYS` | What `collector-purge` deletes against (default 1095 — see Storage and retention) |
| `COLLECTOR_VOICE_RECORDING` | LiveKit Cloud upload mode (default off — see Running the voice worker) |
| `COLLECTOR_TRACING` | OpenTelemetry span export. `off` (default) or `otlp` — see Observability |
| `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` | Where `COLLECTOR_TRACING=otlp` sends spans, and how it authenticates |

Everything through the tier-2 eval suite runs with none of the above set — the personas fall
back to scripted transcripts and the LLM judges skip.

## Boundaries

No payment capture — the agent closes and logs an *agreement*, never collects a card or bank
account (keeps PCI scope at zero). No human transfer queue in this build — dispute, hardship,
distress, attorney representation, or cease-and-desist all end the call immediately with an
`escalation` record; a real deployment swaps in a warm transfer without touching the trigger
logic. No real outbound calling, ever — nothing in this repo dials a real number.
