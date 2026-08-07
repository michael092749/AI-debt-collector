# Voice Debt-Collection Negotiation Agent

A voice agent that calls a consumer 180+ days delinquent on a $1,000 debt and closes the
highest-value agreement they'll actually honor — without threats, false urgency, or invented
consequences.

> **The LLM talks. Deterministic code decides.**

The model handles understanding, phrasing, and rapport. Every amount, schedule, discount, and
concession is computed by a pure Python decision engine the model calls as a tool, mid-call. The
model may never originate a number — this is enforced mechanically, not requested in a prompt.

`SPEC.md` is the full contract. This is the map.

## Documentation

| Doc | What it is |
|---|---|
| `SPEC.md` | The full contract — policy, tiers, guardrail rings, the ten build steps, and §2.4's numbered assumptions (`A1`–`A6`) that the code cites by ID |
| `VOICE_QUICKSTART.md` | Getting the voice worker onto a real LiveKit room: keys, `.env`, running the worker, dispatching a job, troubleshooting |
| `OBSERVABILITY.md` | Why this repo does *not* export OTel traces, and the three gaps it fixed instead |
| `issues.md` | Open and closed code-review follow-ups (`C1`–`C8`, `R1`–`R5`), cited by ID from source comments |
| `docs/brief.md` | The original assignment, verbatim — what `SPEC.md` is an interpretation of |
| `docs/research/` | Background research the design argues against, plus architecture diagrams |
| `docs/archive/` | Finished build artifacts: the step 1–2 plan, the ten-step todo, and a superseded handoff. Historical — not maintained |

## What's actually graded

The brief tests three things, each satisfied by a specific, isolated piece of the system:

| Brief requirement | Where |
|---|---|
| Proposed amounts validated and countered by logic **outside the agent, mid-call** | `decision_engine.py` — pure function, called as a tool, returns a verdict with every evaluated condition |
| Compliance guardrails that fail regardless of conversion | `guardrails/` — three rings enforced in code, including a numeric guard that blocks any unauthorized figure before it reaches TTS |
| The final agreement, logged | `audit/` — SQLite + a standalone JSON record with the full decision trail |

The reviewer's test for the decision record, from the background research this project is built
against: *"Show me the decision record. If it's a transcript, the model decided. If it's
evaluated conditions and a policy path, the engine did."*

## Quick start

```bash
uv sync                            # steps 1-8 need no keys at all
uv run pytest                      # 538 tests, offline
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

## Policy, in one table

| Tier | Total | Payments | Per-payment | Duration |
|---|---|---|---|---|
| T1 — pay in full | $1,000 | 1 | $1,000 | immediate |
| T2 — downpayment + one | $1,000 | 2 | each ≥ $250, first maximized | ≤ 3 months |
| T3 — settlement | $800–$999.99 | ≤ 3 | each ≥ $250 | ≤ 3 months |
| T4 — payment plan | $1,000 | ≤ 4 | each ≥ $250 | ≤ 3 months |

The floor ($250 = 25% of the *original* $1,000 balance, fixed at every tier — SPEC §2.4 A1) is
why a plan can never have more than 4 payments, and why there is no path to a $500 deal: $800
is the hard minimum, always. The agent walks *down* this ladder as the consumer pushes back and
never back up (A4) — concessions are earned by a refusal on record, never volunteered.

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
  Enforced by an AST test (`tests/test_engine_invariants.py`), not just convention.
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
  adversarial pass — the alternates are noticeably less reliable on the mandatory AI-disclosure
  opening line (see `VOICE_QUICKSTART.md`), which is a compliance surface, not a latency one.
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
| `lk agent dev src/collector/voice_app.py` | LiveKit voice worker, dev mode — needs `.env` exported into the shell (`VOICE_QUICKSTART.md` §5) |
| `uv run pytest` | Tier-1 + tier-2 suite, offline (adversarial evals fall back to scripted personas without a key) |
| `uv run collector-agreements` | Dump every agreement record + its decision trail as JSON |
| `uv run ruff check src tests && uv run mypy src` | Lint + strict types |

## Testing strategy

- **Tier 1** (`tests/`) — pure unit and property tests over the decision engine, guardrails, and
  negotiation ladder, plus the full agent loop against the mock client. No API key, ever.
- **Tier 2** (`tests/evals/`) — eight adversarial personas from SPEC §7.2 (lowballer,
  impossible-schedule, rage, evasive/silent, hardship, verbal dispute, jailbreaker, serial
  re-negotiator) driven by `claude-sonnet-5` for genuinely unpredictable pressure when a key is
  configured, with scripted transcripts as the offline fallback — so `uv run pytest` stays green
  either way. Assertions run over the *whole transcript*: no prohibited phrase, no unauthorized
  figure, no agreement below the floors, disclosures fired in order, escalation where the
  persona calls for it, and the post-call compliance verdict itself.
- **Tier 3** (production sampling) — out of scope to run; the log schema supports it.

Verified at runtime, not just in tests: a consumer who signals $750 down closes T2 at
$750 + $250 (the downpayment clamp — a smaller signal, e.g. $500, closes just as legally at
$500 + $500); "I want to pay weekly for a year" is rejected and countered with 4 weekly
payments of $250 (the only legal structure); a lowballer offering $50 then $200 is refused both
times and closes at the $1,000 floor; hardship and dispute both escalate immediately with no
agreement recorded.

## Setup

```bash
cp .env.example .env
```

| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | `--claude` mode, tier-2 evals with live adversarial pressure, and any real voice call |
| `OPENROUTER_API_KEY` | `--claude`'s alternate route (`--openrouter`, `COLLECTOR_LLM=openrouter`) |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | `collector-voice` (transport, STT, TTS), and `--livekit` / `COLLECTOR_LLM=livekit` |

Everything through the tier-2 eval suite runs with none of the above set.

## Assumptions

The brief is ambiguous in a few places; SPEC §2.4 resolves each and states why. The one with the
widest blast radius: **"25% minimum" is read as 25% of the *original* $1,000 balance**, not of
whatever total gets agreed to — a fixed $250 floor at every tier. The stricter reading was
chosen and confirmed; it's also why a payment plan tops out at 4 installments and why
$50 → $200 lowball offers never move the floor.

## Boundaries

No payment capture — the agent closes and logs an *agreement*, never collects a card or bank
account (keeps PCI scope at zero). No human transfer queue in this build — dispute, hardship,
distress, attorney representation, or cease-and-desist all end the call immediately with an
`escalation` record; a real deployment swaps in a warm transfer without touching the trigger
logic. No real outbound calling, ever — nothing in this repo dials a real number.
