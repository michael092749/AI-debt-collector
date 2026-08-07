# Production LLM Agent Systems — Research Report
**Prepared for the Corafone voice-agent task (July 2026 brief) — but applicable to any production AI agent deployment.**

---

## 1. The single most important finding

Every credible production deployment converges on one architectural principle:

> **The LLM talks. Deterministic code decides.**

Zowie (in production across four regulated markets for collections) states it plainly: "The model phrases; the engine decides." The language model handles understanding and conversation; every offer, limit, discount, and escalation is computed by a deterministic **Decision Engine** outside the model. Their test for any vendor: *"Show me the decision record for a negotiated arrangement. If they show you a transcript, the model decided. If they show you evaluated conditions and a policy path, the engine did."* [^16^]

This maps **exactly** onto the Corafone brief's requirement: *"The consumer's proposed amount must be validated and countered by logic outside the agent, mid-call."* The brief is testing whether you know this principle. Everything below follows from it.

Why it matters: in regulated conversations, a model "trying to be helpful" improvises discounts, extensions, or threats that policy doesn't allow. No output filter catches every improvised kindness or menace — so you don't let the model make those decisions at all. [^16^]

---

## 2. What a production LLM system actually looks like — the layers

ByteByteGo's "Typical AI Agent Stack" frames it well: an **Agent Runtime** running a think→tool→observe loop, fed by a Model Layer, a Tool Layer, and a Memory Layer — all wrapped by an **Observability & Safety Layer** that keeps the system "debuggable, evaluable, cost-aware, and safe." [^4^]

Concretely, for a production agent you need seven things. Keep each one simple.

### 2.1 Guardrails — three rings, enforced in code, not prompts

The industry standard is pre-call / during-call / post-call rings (Skit.ai's model): [^14^]

| Ring | What runs | Enforced by |
|---|---|---|
| **Pre-action** | Identity verification, eligibility, prompt-injection/PII screening, "should this call even happen" | Deterministic code + rule engine. Fast and cheap (regex/classifiers), no LLM in the hot path [^1^] |
| **During (runtime)** | Prohibited-phrase blocking (threats, false urgency, invented consequences), required disclosures (Mini-Miranda), sentiment/hardship/dispute detection → escalation | Output streamed through a **rule engine before it's spoken**; decision logic called as a tool, never computed by the LLM [^12^][^13^][^14^] |
| **Post-action** | 100% transcription & compliance scoring, drift detection, immutable audit log, agreement logging | Async eval pipeline over traces [^14^] |

Key lessons from industry case studies:
- **"Guardrails before action, not after output."** Enforce authorization at the tool-execution layer — by the time you filter a response, the agent already said it. [^8^] For voice: screen the TTS text *before* synthesis where possible.
- **Prohibited phrases should be *never generated*, not filtered after.** DROS blocks threats/garnishment/legal language "at the model layer — not filtered after, never generated." [^13^]
- **Whitelist, don't blacklist.** "The agent can ONLY do A, B, C" — not "can do anything except X." [^3^]
- **Escalation is a guardrail.** Dispute, hardship, distress, or low confidence → immediate human handoff with full context passed over. [^13^][^14^]

### 2.2 Context management

Four memory layers, but you only need three for a single-call agent: [^3^]
1. **Account context loaded before the call** — balance, delinquency age, prior contact attempts, broken promises, dispute flags, language preference. DROS "doesn't dial cold"; every call starts with full account context. [^13^][^15^]
2. **Conversation memory** — the live transcript within the context window. For a 5–15 minute call this fits easily; summarize only if calls run long.
3. **Working state outside the context window** — negotiation state machine (offers made, counters, agreed terms) persisted in a store (Redis/Postgres), not in the prompt. This is what survives a dropped call and feeds the audit trail.
4. (Cross-call long-term memory — out of scope for the task, trivial to add later via Postgres.)

Rule of thumb: **business state lives in a database, conversation lives in the context window.** Never let the transcript be the only record of what was agreed.

### 2.3 Tools & orchestration

- Tool count small (<10): a single agent with a tool belt is the right pattern — router/specialist patterns are over-engineering here. [^3^]
- The **Decision Engine is a tool**: `validate_offer(amount, schedule)` → returns accept/counter/reject with reasons. The LLM calls it mid-call, gets a deterministic verdict, and phrases it. This directly satisfies the brief's "validated and countered by logic outside the agent."
- Every tool invocation treated as a policy decision: validate parameters, check authorization, log input/output/latency. [^10^]
- Human-in-the-loop via interrupt/resume with a persistent checkpointer (LangGraph's model: `interrupt()` pauses, state saved to Postgres, `Command(resume=...)` continues). [^20^][^24^]

### 2.4 Observability

Instrument from day one with OpenTelemetry/OpenInference traces. Capture per call: [^1^][^5^]
- Every LLM call (prompt, completion, tokens, cost, latency)
- Every tool call (arguments, result, latency)
- Guardrail events (what was blocked, what disclosure fired, what escalated)
- Decision-engine verdicts (the evaluated conditions, not just the outcome)
- Session-level grouping (the whole call as one trace)

**89% of teams with production agents have observability, but only 52% have evals — that gap is where quality dies.** [^8^]

Platform choice (keep it simple): [^2^]
- **Langfuse** — open-source, self-hostable, framework-agnostic via OTel. Default pick.
- **LangSmith** — if you're on LangGraph and want the deepest integration + replay-against-new-models.
- Helicone for a 5-minute proxy-based start; Arize Phoenix if eval rigor becomes the priority.

### 2.5 Evals — three tiers [^8^]

1. **PR-time checks** (fast, deterministic): does the agent call the right tools? Does the decision engine reject a $100 offer on a $1,000 debt (below the 25% minimum)? Does a threat ever pass the guardrail in a scripted scenario?
2. **Nightly/weekly regression**: LLM-as-judge over a golden dataset of simulated calls — including **adversarial consumers** (angry, evasive, confused, jailbreak-y, offering $50, asking for 90% off). Score: compliance, goal progression, tone, correct use of the decision engine.
3. **Continuous production evals**: sample 5–10% of live traces, LLM-judge for compliance drift and hallucinated terms; alert on failures, feed them back into the golden dataset. [^7^]

Domu's lesson is the gold standard for the Corafone task: **adversarial certification before go-live** — stress-test the agent through thousands of simulated edge cases (hardship claims, verbal disputes, rage, silence) and only launch once it holds policy. *"You don't launch and hope; you audit and then launch."* [^12^] The brief literally warns you: *"we will not be a cooperative consumer."*

Start small: 10 realistic scenarios + 1 end-to-end eval beats a perfect dataset you never build. [^7^]

### 2.6 The voice layer (specific to this task)

Production voice agents in 2026 use a **cascade architecture**: STT → LLM → TTS, each component swappable. Speech-to-speech only where naturalness is the product. [^21^][^22^]

- **Latency is the whole game:** sub-500ms perceived response. Under ~300ms feels human; over 1.5s they hang up. [^21^]
- Orchestration: **LiveKit Agents** (production-grade, WebRTC, telephony via SIP/Twilio) or **Pipecat** (most flexible, Python-first). Managed platforms (Vapi, Retell) get a phone number live in hours — fine for a 2-day deadline, but you trade control of the guardrail/decision layers. [^21^][^22^][^23^]
- Typical components: Deepgram (STT), your LLM, Cartesia/ElevenLabs (TTS).
- Interruption handling (barge-in) and turn-taking matter more than voice beauty — "we will not be a cooperative consumer" means they *will* talk over the agent.

### 2.7 Compliance guardrails for collections (the domain layer)

Encode as code, not prompt text: [^11^][^14^]
- Required disclosures (Mini-Miranda: "this is an attempt to collect a debt…"), AI disclosure where required
- No threats, harassment, false statements, false urgency, invented consequences (FDCPA §806/807 — and the brief's own fail condition)
- No legal/financial advice
- Dispute/hardship keywords → change call state, escalate, fence off payment capture
- Call-time windows & frequency caps (in real deployments; less relevant to the demo)
- 100% recording + transcript + agreement log (the brief's "log the final agreement" — write it to Postgres + a structured JSON record with the decision-engine trail)

---

## 3. Recommended setup — concrete and minimal

| Layer | Choice | Why |
|---|---|---|
| Voice orchestration | **LiveKit Agents** (Python) + Twilio SIP, or **Vapi** if speed dominates | Production-grade, telephony path, components swappable [^21^][^22^] |
| STT / TTS | Deepgram / Cartesia (or ElevenLabs) | Low-latency streaming [^22^] |
| Agent logic | **LangGraph** (state machine, not free ReAct loop) | Negotiation is a structured flow: verify → disclose → negotiate → confirm → log. Explicit states = controllable, testable, checkpointed [^8^][^20^] |
| Decision engine | Plain Python module exposed as a tool | Deterministic: full pay / downpayment+1 / settlement ≤20% off ≤3 payments / plan ≤3 months, min payment ≥25%. Zero LLM involvement. |
| Guardrails | Custom rule engine + regex/classifier pre-TTS check; disclosure state enforced by graph | "Never generated, never spoken" [^13^] |
| State & agreement log | Postgres (+ Redis for live call state) | Checkpointer for LangGraph, audit trail, agreement record |
| Observability | **Langfuse** (self-host or cloud) | Traces every LLM/tool/guardrail event; feeds evals [^2^][^5^] |
| Evals | Pytest (deterministic) + Langfuse LLM-as-judge + adversarial call simulator (an LLM playing an uncooperative consumer) | Three-tier eval strategy [^7^][^8^][^12^] |
| Deploy | Docker on a single VM/Fly.io/Render; webhook for telephony | Keep it boring |

Deliberately **excluded** (over-complication traps): multi-agent architectures, vector DBs/RAG, MCP servers, message queues, Kubernetes. A single-call negotiation agent needs none of them.

---

## 4. Diagrams

Two diagrams accompany this report:

1. **`infrastructure_diagram.svg`** — the deployed system: telephony → voice pipeline → agent runtime → decision engine / guardrails / state, wrapped by observability & evals.
2. **`data_flow_diagram.svg`** — one conversational turn end-to-end: audio in → STT → guardrail pre-check → LLM → tool call to decision engine → guardrail post-check → TTS → audio out, with async logging and evals.

---

## 5. Lessons learned from industry case studies (condensed)

1. **Separate deciding from talking** (Zowie). The single highest-leverage decision. [^16^]
2. **Certify before launch with adversarial simulation** (Domu — 30% fewer complaints per 100 calls at a top-5 US fintech after deployment). [^12^]
3. **Guardrails wrap the whole call: pre, during, post** (Skit.ai — including a published "Agent Card" stating what the agent will never do). [^14^]
4. **Escalation is a feature, not a failure** — dispute/hardship/distress → human, with full context handoff (DROS, Skit). [^13^][^14^]
5. **Compliance lives in the platform layer, not the prompt** — "these aren't prompts; they're blocks on the dialer logic itself." [^12^]
6. **100% auditability beats sampling** in regulated domains — every call transcribed, scored, and preserved with an immutable log. [^12^][^14^]
7. **Latency budget drives every voice decision** — sub-500ms or callers hang up; cascade STT→LLM→TTS is the proven default. [^21^]
8. **Evals before deployment, not after the incident** — the 89% observability / 52% evals gap "is where production quality dies." [^8^]
9. **Keep the architecture boring** — single agent + tool belt + deterministic decision engine beats multi-agent novelty for a well-defined domain. [^3^]
10. **Speed without control is a false economy** (ByteByteGo) — the observability & safety layer is what separates a demo from a deployment. [^4^]

---

## Sources
[^1^]: Arthur — Evaluating AI Agents in Production (arthur.ai, Jun 2026)
[^2^]: Digital Applied — Agent Observability Platforms 2026: LangSmith, Langfuse, Arize (Apr 2026)
[^3^]: HyperTrends — Building Production AI Agent Systems: Architecture Patterns That Scale (Apr 2026)
[^4^]: ByteByteGo EP218 — The Typical AI Agent Stack, Explained (Jun 2026)
[^5^]: Langfuse — AI Agent Observability, Tracing & Evaluation (Jul 2026)
[^7^]: Scorable — How to Build Eval-Driven AI Observability for Agents (May 2026)
[^8^]: The AI Engineer — The AI Agents Stack, 2026 Edition (Mar 2026)
[^10^]: AppSecEngineer — How to Design Guardrails for Secure and Scalable AI Agents (Mar 2026)
[^11^]: Sumeru Digital — AI Voice Agent Development for Debt Collection (Jul 2026)
[^12^]: Domu — The 8 Gen AI Voice Agents Actually Built for Collections (Jul 2026)
[^13^]: DROS AI — Context-Aware Voice AI Agents for Debt Collection
[^14^]: Skit.ai — Responsible Voice AI for Debt Collection (Jul 2026)
[^15^]: PR Newswire — DROS.ai launches voice agents with compliance guardrails (Jun 2026)
[^16^]: Zowie — AI Debt Collection (2026): How AI Agents Work Every Account (Jun 2026)
[^20^]: LangChain Docs — Human-in-the-loop middleware (2026)
[^21^]: Forasoft — Build and Deploy LiveKit AI Voice Agents: The 2026 Playbook (Jul 2026)
[^22^]: Gradium — Best API to Build an AI Voice Agent in 2026 (Jul 2026)
[^23^]: Thinnest.ai — Open-Source Voice AI Frameworks Ranked 2026 (Apr 2026)
[^24^]: Towards AI — LangGraph Human-in-the-Loop (Jun 2026)
