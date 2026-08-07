# Observability evaluation — voice pipeline (2026-08-07)

Scope: do we need to add observability to `src/collector/voice_app.py`? Prompted by
`docs.livekit.io/deploy/observability/tracing` (OTel trace export). Evaluated against
the pinned `livekit-agents==1.6.8` **source in `.venv`**, not from memory, per
`CLAUDE.md`. Companion to `issues.md` (R2 set the current recording posture).

## Verdict

**Do not wire the OTLP/Langfuse exporter.** It is the wrong tool for this repo's
shape, and adopting it would re-open the exact consent question `issues.md` R2
closed. See "Why not OTel export" below.

**Do fix three concrete gaps**, none of which need a new backend, a new
dependency, or a network egress decision. G1 is a compliance defect, not a
monitoring nice-to-have.

## Current posture (verified)

| Surface | State |
| --- | --- |
| `AuditStore` (SQLite + agreement JSON) | The SPEC §6 compliance record. Complete for the happy path. |
| LiveKit Cloud insights (audio/transcript/traces/logs) | **Off.** `voice_app.py:248` passes `record=dev_recording`, `False` by default (R2). |
| OTel spans | **Not exported anywhere.** With `record=False`, `JobContext.init_recording()` (`job.py:761-790`) returns before `_setup_cloud_tracer()`, so no exporter is installed. Spans are created and dropped. |
| Latency | `agent.py:_timed_respond` logs `elapsed_ms` per LLM call to a stdlib logger. Nothing measures a **turn**. |
| Anything else | Worker stdout/stderr. That is the whole production picture today. |

*(Row 2 and 3 of that table still hold. Rows 4 and 5 were the state before the
2026-08-07 second pass — see G4–G8 below, which is where the picture stands now.)*

## G1 — the transient-error apology is spoken but never audited (compliance)

`voice_app.py:126-136`: when `NegotiationAgent.turn()` raises, the handler yields
`_TRANSIENT_ERROR_APOLOGY` straight to TTS. That path never reaches
`NegotiationAgent._record_spoken()` (`agent.py:460`), which is the only writer of
`TurnRecorded(speaker=AGENT)`.

Two consequences, same root cause:

1. **A spoken line is missing from the timeline.** `turn()` already recorded the
   consumer's utterance and incremented `_turn_index` before any point it could
   raise (`agent.py:223-234`). The audit shows consumer turn N, then consumer turn
   N+1, with **agent silence between them** — while the consumer actually heard a
   sentence. `store.py`'s own docstring states the contract this breaks: *"what was
   said, and what stopped being said" is one timeline.* The agent's `self.messages`
   also never sees the apology, so the model's context diverges from the call.
2. **The call undercounts its own turns.** `self.turns.append(turn)` is at
   `agent.py:260`, after `_act()`. A raise skips it, so `close()` emits
   `CallEnded(turn_count=len(self.turns))` and `CallReport.turns` both excluding
   every errored turn.

This is the single most valuable thing on this page and it is unrelated to LiveKit
tooling. Fix: give `NegotiationAgent` a narrow public method that records a
code-authored spoken line and accounts for the turn (`_speak_verbatim` already does
the first half for the escalation closing), and call it from the `except` branch
before yielding.

**Call it off the event loop.** `AuditStore.record()` routes through
`self._executor.submit(...).result()` (`store.py:_run`), which *blocks* for the
store's worker thread. The `except` branch runs on the event loop, so it must be
`await asyncio.to_thread(...)` — the same wrapping every other store touch in
`voice_app.py` already uses (`:207`, `:231`), for the reason stated at `:122-125`.

## G2 — no measurement of turn latency, on a design that can stack 7 LLM calls

Worst case inside one voice turn, from the constants:

- `MAX_TOOL_ROUNDS = 4` (`agent.py:73`) → 4 `respond()` calls
- the "ask once more for words" call (`agent.py:312`) → 5
- `MAX_REGENERATION_STRIKES = 2` (`rings.py:48`) → up to 7

Seven *sequential*, blocking Anthropic round trips between the consumer finishing a
sentence and hearing anything. On a phone call that is dead air measured in seconds.
Nothing currently measures it end to end.

The framework already computes this for free and it survives the `llm_node`
override:

- `llm_gen_data.ttft` is set on the first chunk our async generator yields
  (`generation.py:229-235`). Because the override yields only after the whole
  blocking `turn()` completes, **`llm_node_ttft` is a direct measurement of total
  `NegotiationAgent.turn()` latency** — the 7-call worst case, exactly.
- `e2e_latency` = agent playback start − user stop-speaking
  (`agent_activity.py:3403-3406`). Computed in the reply path from TTS playback, so
  it is independent of the `llm_node` override.

Both land on `ChatMessage.metrics` and are readable from the `conversation_item_added`
event. Cost: a handful of lines, no new dependency, no data leaving the process,
fully compatible with `record=False`.

## G3 — the `llm_node` span is a black box (only matters if tracing is ever added)

`generation.py:174` wraps the *node callable*, so the override does get a span. But
everything this project actually cares about — tool rounds, engine verdicts,
guardrail trips, regeneration strikes — happens inside `turn()` on a worker thread
and produces no spans. A trace would show one opaque multi-second `llm_node` and
nothing about why.

Worth noting if tracing is ever adopted: `asyncio.to_thread` copies the contextvars,
so spans started inside `turn()` **would** nest correctly under `llm_node`. The
plumbing is not the obstacle; the exporter decision is.

Minor, related: `open_call()` runs at `voice_app.py:207`, before `session.start()`,
so the opening line's LLM call parents under `job_entrypoint`, not `agent_session`.

## G4 — logs were unstructured, so none of the above was queryable (2026-08-07)

Second pass, prompted by `docs.livekit.io/deploy/observability/insights`. G2 landed the
turn-latency measurement but emitted it as `"turn latency e2e=%s turn_ms=%s ..."`.
Under `collector-voice start` the framework formats records as JSON
(`cli/log.py:setup_logging`, non-devmode → `JsonFormatter`), so that whole
sentence arrives as one opaque `message` string. Nothing could ask
`e2e_ms > 2000`. Same for `agent.py`'s `elapsed_ms` line.

**Verified, not assumed** (`cli/log.py`):

* `_merge_record_extra` copies every non-reserved `record.__dict__` key into the
  output — **no allowlist**. So `extra={...}` becomes real top-level JSON fields.
* Dev mode's `ColoredFormatter` renders the same dict after the message via its
  `%(extra)s` slot, so nothing is lost at the terminal either.

Every log call in `agent.py` and `voice_app.py` now uses `extra=`, with a short
event name as the message (`turn_complete`, `outbound_blocked`, `turn_latency`).

## G5 — the call had no correlator on its own log lines

`ctx.log_context_fields` was `{"room": ...}` and was set before
`_consumer_context()` parsed, so the R1 refusal path logged without an account.

**Verified**: `JobContext._on_setup` (`job.py:224-228`) installs its filter on the
**root** logger's handlers, not on `livekit.*`. So the fields land on
`collector.agent` and `collector-voice` records too — this is what makes them
worth setting. (The `agent.py` lines are emitted from an `asyncio.to_thread`
worker; `contextvars` are copied into it, and in the production `PROCESS`
executor the filter skips the context comparison entirely.)

Now `room`, `call_id`, `channel`, `llm_route`, and — after parsing —
`account_ref`. **Not** `consumer_name`: it buys nothing a reader of the audit
store cannot get, and logs leave the process by stdout and any configured drain,
which carry none of the consent posture R2 set for consumer speech.

## G6 — everything outside `llm_node` was dark

`llm_node`'s `try`/`except` (R5/C4) covers the reasoning step and nothing else. An
STT stream that never returns a transcript, a TTS failure mid-sentence, a false
interruption — the framework logs its own view, but nothing tied any of it to the
call. `_log_session_events()` registers four handlers:

| Event | Logged | Why |
| --- | --- | --- |
| `error` | source/error class, `recoverable` | STT/TTS/LLM failures the session catches. Level follows `recoverable` — unrecoverable TTS means the consumer hears silence. |
| `user_transcription_timeout` | `speech_duration_ms` | The consumer spoke and STT returned nothing. Otherwise the call just goes quiet with no cause. |
| `agent_false_interruption` | `resumed` | Noise cut the agent off. Fires with the stock session — `false_interruption_timeout` is `2.0` and `resume_false_interruption` `True` in `turn.py:_INTERRUPTION_DEFAULTS`, so the handler is live without any turn-handling config. |
| `user_input_transcribed` | `chars`, finals only | Confirms STT is producing. **Metadata only** — see below. |

Plus `_log_session_usage()` in the shutdown callback: one line per model from
`session.usage`, a local rollup that works with `record=False`. It covers STT and
TTS through Inference and **not** the negotiation model — `llm_node` bypasses the
framework's LLM, so Anthropic's tokens never reach the session's collector. It is
a speech-pipeline usage line, not the call's cost.

## G7 — the log said nothing about *why* the agent said what it said

The richest thing this project knows about a turn — the engine's verdict, the
guardrail trips, the regeneration strikes — was written to SQLite and nowhere
else. That is the right home for the record, but during a live call the log is
the only visible surface. `agent.py` now emits, log-only with no control-flow
change: `turn_complete` (tools run, verdict outcomes, rationale codes,
regeneration count, identity state), `outbound_blocked` (rule ids, strike number,
whether the fallback fired), and `escalated` (trigger).

**Codes only, never prose.** The candidate text a guardrail refused is exactly the
text that must not be repeated, and the escalation detail embeds the consumer's
own words. Both stay in the audit store — `blocked_text` is the field built to
hold them.

Asserted, not just commented — and asserted against `record.__dict__`, not
`caplog.text`. That distinction is the whole test: pytest's formatted text is
`LEVEL logger:file:line message` and never includes the `extra` dict, so a
`caplog.text` check would pass even if the blocked candidate had been logged as
`extra={"blocked_text": ...}`. The boundary is between the *taxonomy* and the
*utterance*, not the topic: `THREAT_GARNISHMENT` is a rule id and belongs in the
log; "garnish your wages" is the consumer-facing sentence and does not.

## G8 — the `record` parameter now has a three-way answer

`_recording_options()` reads `COLLECTOR_VOICE_RECORDING`: `off` (default),
`diagnostics` (`{"audio": False, "transcript": False, "traces": True, "logs":
True}`), `full`. This replaces `COLLECTOR_VOICE_DEV_RECORDING=1`.

**The default does not move**, and unrecognized values fail closed to `off` —
the direction a typo should resolve when the alternative is uploading a
consumer's speech. `diagnostics` is the granular middle ground this document
already named; it is narrower than what R2 declined, but still a decision to make
per deployment rather than drift into.

## Why not OTel export

1. **It reproduces R2.** Spans carry `ATTR_CHAT_CTX` (the full chat context) and
   `ATTR_RESPONSE_TEXT` (`generation.py:191-207, 224`) — i.e. the consumer's words.
   Shipping those to Langfuse or any SaaS is the same third-party copy of consumer
   speech that R2 declined to make without a consent flow. The only difference is
   that it is text rather than audio.
2. **The `redaction` recording option does not help here.** `RecordingOptions`
   (`agent_session.py:99-124`) does have a `redaction` key, undocumented on the docs
   page. But `job.py:838` only stamps `ATTRIBUTE_REDACTION_ENABLED` into
   `_otel_metadata()`, which feeds `_setup_cloud_tracer()`. It is a **LiveKit Cloud
   pipeline** feature. A user-supplied `TracerProvider` exporting elsewhere gets no
   redaction at all.
3. **It is not dependency-free.** `opentelemetry-sdk` / `-exporter-otlp` are in
   `.venv` *transitively* via `livekit-agents`, but absent from `pyproject.toml`.
   Importing them in `voice_app.py` creates an undeclared dependency that breaks on
   the next `livekit-agents` bump. If ever adopted, declare them in the same change.
4. **Cloud traces and self-hosted export are likely mutually exclusive, not
   additive** — `_setup_cloud_tracer` installs its own provider. Confirm before
   assuming both can run.

There is no production deployment here whose incidents this would diagnose. The
graded deliverable is the SQLite evidence store, and it is already the richer
record — it holds verdicts, rationale codes, and guardrail trips that no LiveKit
span models.

If a real deployment later needs it, the honest granular middle ground is
`record={"audio": False, "transcript": False, "traces": True, "logs": True}` —
keeps pipeline traces and runtime logs in LiveKit Cloud (30-day retention, and note
the free **Build** plan's model-improvement program retains anonymized data longer),
drops the audio copy. That is still a consent decision, not a default flip.

## Recommended changes, in priority order

1. **G1** — *Done.* `NegotiationAgent.record_fallback_speech()` records the apology
   and accounts for the turn; `llm_node`'s `except` branch calls it via
   `asyncio.to_thread` before yielding. Tracked as `issues.md` C4.
2. **G2** — *Done.* `_log_turn_latency()` registers a `conversation_item_added`
   handler logging `e2e_latency`, `llm_node_ttft`, and `tts_node_ttfb` per assistant
   turn. No new dependency, nothing leaves the process, works with `record=False`.
3. **Tests for `voice_app.py`** — *Done.* `tests/test_voice_app.py` covers the C4
   error path, the latency hook, and `_consumer_context`'s R1 refusal path. Driven by
   constructing `CollectorAgent` directly and calling `llm_node` with a hand-built
   `ChatContext`; no room, no network, no API keys.

4. **G4–G7** — *Done (2026-08-07, second pass).* Structured `extra=` fields on every
   log call, enriched `log_context_fields`, four session-event handlers plus a
   usage rollup, and a per-turn engine/guardrail line in `agent.py`. No new
   dependency, no network egress, all compatible with `record=False`.
5. **G8** — *Done.* `COLLECTOR_VOICE_RECORDING=off|diagnostics|full`, default `off`.
6. **Tests** — *Done.* `TestTurnLogging` in `tests/test_agent_loop.py` (4) and the
   recording-mode and session-event blocks in `tests/test_voice_app.py` (14).
   487 passing, `ruff` and `mypy --strict` clean.

Not recommended: OTLP export, Langfuse, flipping `record`'s default.

## Still open

- **G3** stands as written — if tracing is ever adopted, the internals of `turn()`
  need their own spans or the trace shows one opaque multi-second `llm_node`.
  G7's `turn_complete` line narrows this in practice: with `COLLECTOR_VOICE_RECORDING=
  diagnostics`, the agent-server logs land in the same LiveKit Cloud timeline as the
  spans, so the opaque `llm_node` span now has an explanatory log line beside it at
  the same timestamp. Not a substitute for spans — no nesting, no per-round-trip
  breakdown — but it is most of the diagnostic value.
- **`ErrorEvent` and `agent_false_interruption` are wired but not exercised against a
  live pipeline.** Both are reachable in tests only by emitting the event by hand,
  which proves the handler and its fields, not that the framework raises it when
  Deepgram actually fails. The defaults that make the false-interruption path live
  were read from `turn.py:_INTERRUPTION_DEFAULTS` rather than observed.
- **`_log_session_usage` is untested.** It runs in the shutdown callback and reads
  `session.usage`, which is empty for a text-mode session with no STT or TTS
  attached — there is nothing meaningful to assert offline.
- **`make_session_report()` was considered and declined.** It dumps the full
  conversation history to a file: a second copy of consumer speech and a strictly
  poorer duplicate of the `AuditStore`, which already holds verdicts, rationale
  codes, and guardrail trips it does not model. Same reasoning as R2.
- **Everything on this page is working-tree-only, and the set is now larger.**
  `voice_app.py`, `issues.md`, this file, `VOICE_QUICKSTART.md`, and
  `tests/test_voice_app.py` are all untracked in git; `agent.py` and
  `tests/test_agent_loop.py` are tracked-and-modified. The second pass deepened the
  coupling: `agent.py` now also carries `_log_turn()` and the `outbound_blocked` /
  `escalated` lines, and `tests/test_agent_loop.py::TestTurnLogging` asserts on them.
  Those two files *can* land alone without breaking (the logging is self-contained),
  but `record_fallback_speech()` would still be in history with no caller. Land the
  set together.
- Not evaluated here: the layout/packaging findings from the `agent-starter-python`
  comparison. Only the `.gitignore` fix (`.env.*` + `!.env.example`) landed alongside
  this work, because it is a pure secret-handling guard with no behaviour change. The
  matching `load_dotenv(".env.local")` change was reverted: `anthropic_client.py:53`
  and `openrouter_client.py:60` each call bare `load_dotenv()`, so changing only
  `voice_app.py` would make `collector-voice` and `collector-text` read different env
  files. Fix all three together or none. The larger finding — `voice_app.py`, the
  `Dockerfile`, `README.md`, and `livekit.toml` all untracked, so a fresh clone has no
  voice entrypoint — is separate and unaddressed.
