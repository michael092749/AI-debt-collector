# Voice pipeline quickstart

Gets `collector-voice` running against a real LiveKit room with live STT/TTS and Claude
reasoning. No real phone number is ever dialed — nothing in this repo does outbound telephony
(see README "Boundaries"). You talk to the agent through a LiveKit room instead.

## 1. Accounts / keys needed

| Service | What it's for | Where to get it |
|---|---|---|
| Anthropic | The agent's reasoning (`AnthropicClient`, `claude-sonnet-5`) | console.anthropic.com → API Keys |
| LiveKit Cloud | Room/session transport, STT, TTS, and optionally the reasoning model | cloud.livekit.io → create a project → Settings → Keys |

STT (Deepgram Nova-3) and TTS (Cartesia Sonic-3) are served through LiveKit Inference, so the
LiveKit project credentials cover them — no Deepgram or Cartesia account is needed. The same is
true of the reasoning model if you set `COLLECTOR_LLM=livekit` (Gemini 3 Flash), which drops the
Anthropic key too; see the note under "Model routes" below before doing that on a real call.

## 2. Install

```bash
uv sync --extra dev
```

## 3. Configure

```bash
cp .env.example .env
```

Fill in all four values in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

### Model routes

`COLLECTOR_LLM` picks the reasoning backend for `collector-voice` (`text_app.py` takes the same
choices as CLI flags):

| Value | Backend | Key needed |
|---|---|---|
| unset (default) | `claude-sonnet-5` via Anthropic | `ANTHROPIC_API_KEY` |
| `openrouter` | `claude-sonnet-5` via OpenRouter | `OPENROUTER_API_KEY` |
| `livekit` | `google/gemini-3-flash-preview` via LiveKit Inference | none beyond `LIVEKIT_*` |

Only the default is certified. `MAX_TOOL_ROUNDS` and the regeneration-strike budget were tuned
against Claude, so re-run `tests/evals/` and the ADVERSARIAL_TESTING pass before either alternate
route carries a real consumer call.

## 4. Sanity-check the core first (no keys needed for this part)

```bash
uv run collector-text
```

If the negotiation logic doesn't behave here, it won't behave on voice either — `voice_app.py`
is the same `NegotiationAgent.turn()` underneath, just with audio in/out instead of a terminal.

## 5. Start the worker

```bash
lk agent dev src/collector/voice_app.py
```

`uv run collector-voice dev` still runs, but the framework's own `dev` mode is deprecated in
favor of the `lk` CLI wrapper (hot-reload has moved there too). `lk agent dev` won't auto-detect
this entrypoint on its own — it only looks for `agent.py`/`src/agent.py` — so the path must be
given explicitly. It also reads `LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET` from your
actual shell environment, not from `.env` (that's `lk`'s own project auth, separate from how
`voice_app.py` loads `.env` via `python-dotenv`), so export them first:

```bash
set -a; source .env; set +a
lk agent dev src/collector/voice_app.py
```

This connects the worker to your LiveKit project and waits to be dispatched a job (see step 6 —
it will not join a room on its own). Leave it running.

By default, the worker disables LiveKit Cloud's recording/observability upload for every call
(see `issues.md` R2 — this project's own `AuditStore` is the compliance deliverable, and nothing
here has consumer consent for a second copy of their audio). That also means the
[Agent Console](https://cloud.livekit.io/projects/p_/agents)'s Events/Audio/Metrics panes and the
[Agent insights](https://cloud.livekit.io/projects/p_/sessions) timeline won't have anything to
show for a dev session. `COLLECTOR_VOICE_RECORDING` opts in:

| Value | Uploaded to LiveKit Cloud |
| --- | --- |
| unset / `off` | **Default.** Nothing. |
| `diagnostics` | Pipeline traces + agent-server logs. No audio, no transcript. |
| `full` | Everything — audio, transcript, traces, logs (the SDK's own default). |

Use `full` only for a **local, no-real-consumer** call in the Console. `diagnostics` gets the
insights timeline for latency and failures without a third-party copy of the consumer's words —
still a deliberate product decision, not a default. Anything unrecognized records nothing.

Whatever the worker logs at is what LiveKit Cloud collects when `logs` is on (`info` and above by
default); `LIVEKIT_LOG_LEVEL=debug` or `--log-level debug` widens it. Every line this project
emits carries `call_id`, `room`, `account_ref`, `channel`, and `llm_route` as structured fields,
so one call can be pulled out of a worker handling several. The consumer's *name* is deliberately
not among them — see `OBSERVABILITY.md` §G4.

By default the worker reasons with `AnthropicClient`. `COLLECTOR_LLM` routes the same calls
elsewhere — see "Model routes" above.

Earlier spot-testing (`collector-text --openrouter`, several runs) showed the opening line
missing the mandatory AI disclosure and tripping the guardrail's safe fallback in roughly half of
attempts. Root-caused via live probing rather than by tuning: the model was disclosing being an
AI on nearly every generated candidate ("this is an AI calling on behalf of..."), but
`_AI_DISCLOSURE_RE` in `guardrails/disclosures.py` only matched first-person phrasing ("I'm an
AI...") and a fixed set of noun compounds ("AI assistant/agent/..."), not that third-person
copular form. Fixed by adding the missing branch (see `tests/test_guardrails.py`,
`TestDisclosureDetection::test_third_person_ai_disclosures_are_recognized`). Two things this
*wasn't*: reasoning effort (`reasoning_tokens: 0` in the response usage at every effort level —
see `openrouter_client.py`'s module docstring) and OpenRouter-specific unreliability — the
regex gap would have blocked the same phrasing from any provider.

## 6. Join a room to talk to it

The worker registers under the dispatch name `collections-negotiator` (`AGENT_NAME` in
`voice_app.py`), which puts it on **explicit** dispatch: opening a room does not summon it. A job
has to name it. That is deliberate — automatic dispatch carries no job metadata, so every call
would run as the fixture consumer, and an agent that auto-joins any room that happens to open can
place a collections call nobody authorized.

Dispatch a job, then join the room it names — e.g. with LiveKit's
[Agents Playground](https://agents-playground.livekit.io) pointed at your `LIVEKIT_URL` project.
Speak into your mic; you should hear the opening AI disclosure line back.

```bash
lk dispatch create \
  --agent-name collections-negotiator \
  --room my-test-room \
  --metadata '{"consumer_name": "Dana Whitfield", "account_ref": "ACCT-4471"}'
```

Metadata is how the call learns whose account it is. Omit `--metadata` and the call falls back to
the fixture consumer "Dana Whitfield" / `ACCT-4471`; supply it but malformed and the worker
refuses the call outright rather than dial under the wrong identity (`issues.md` R1).

## 7. What to expect

Same script as the text client: AI disclosure in the opening line, Mini-Miranda right after
identity confirmation, no dollar figure that isn't traceable to the decision engine, agreement
never below $800 total / $250 per payment. Check the audit trail after the call:

```bash
uv run collector-agreements
```

## 8. Production mode

```bash
uv run collector-voice start
```

Same worker, without the `dev` CLI's local debugging affordances — point real LiveKit dispatch
at it once you're past manual testing.

## Troubleshooting

- **Worker connects but no audio**: STT and TTS run on LiveKit Inference, so there is no separate
  speech key to check — confirm `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` are set and that the
  project has Inference enabled.
- **`RuntimeError: ANTHROPIC_API_KEY is not set`**: voice mode always uses a real backend, never
  the mock — this key is not optional for `collector-voice` the way it is for `collector-text`,
  unless `COLLECTOR_LLM` selects another route (see "Model routes" above).
- **Nothing joins the room**: the worker registers under the dispatch name `collections-negotiator`
  (`AGENT_NAME` in `voice_app.py`), so it is on **explicit** dispatch — opening a room is not enough,
  a job has to name it. Dispatch with
  `lk dispatch create --agent-name collections-negotiator --room <room> --metadata '{"consumer_name": "...", "account_ref": "..."}'`,
  or from a token's `roomConfig.agents`. Also confirm the worker is running *and* connected — check
  its terminal output for connection errors against `LIVEKIT_URL`.
- **`anthropic.BadRequestError: messages: at least one message is required`** on the very first
  call of `open_call()`: fixed. `AnthropicClient` used to map the sole system-prompt message into
  the API's separate `system` field, leaving an empty `messages` array on the call that opens
  every conversation; `_to_anthropic` now appends a synthetic `<call_started>` turn whenever the
  mapped conversation would otherwise be empty. `openrouter_client.py`'s `_to_openrouter` carries
  the equivalent nudge (triggered whenever every message so far is system-role, since a
  regeneration retry produces the same all-system shape) — the chat completions format doesn't
  reject an empty-of-dialogue request the way the Messages API does, but an all-system context is
  still a degenerate one to ask a dialogue model to open from. If you see this error, you're on an
  older working tree than this doc.
