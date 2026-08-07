# Issues — Code Review Follow-up (2026-08-06)

Source: ad-hoc code review of the uncommitted voice-pipeline work (`voice_app.py`,
`store.py`, `prohibited.py`, `rings.py`). Companion to `ADVERSARIAL_TESTING.md` (which
covered an earlier, separate guardrail-certification pass — all 21 of *those* findings are
already remedied). Tracking doc per user request; all items below fixed with a regression
test, per this repo's Prove-It convention. `ruff` and `mypy --strict` clean.

> **`ADVERSARIAL_TESTING.md` is lost.** It was never committed and is not recoverable from
> git or any worktree, yet roughly 35 comments and test docstrings across `src/` and
> `tests/` still cite it by finding ID (`C1`, `H2`, `M6`, …). Those IDs are now orphaned
> identifiers: they still label *which* finding a given line defends against, and the
> regression tests themselves remain the executable record, but the prose describing each
> finding is gone. The citations were deliberately left in place rather than stripped —
> removing them would delete the only surviving trace of the mapping. If that pass is ever
> re-run, write its report back to `ADVERSARIAL_TESTING.md` and keep the same IDs.

## Critical

- [x] **C1 — voice transport cannot place a single call.** `AuditStore` now owns a
      dedicated single-worker `ThreadPoolExecutor` for its whole life; every touch of
      `self._conn` routes through a reentrancy-aware `_run()` helper that hands the work
      to that one worker thread and blocks for the result. Public API stays fully
      synchronous — no caller (including `text_app.py`) had to change. Regression test:
      `tests/test_audit_store_threading.py`, constructs on one thread, drives
      record/query/close from another, asserts rows landed.
- [x] **C2 — the C5 fix blocked reassurances.** Dropped `won'?t|` from
      `_CLAUSE_BREAK_RE`'s subject+verb alternation in `prohibited.py` (it was the
      negation cue itself getting eaten by the clause split, not a real clause boundary).
      `"We won't sue you."` etc. no longer block; `"We will sue you."` etc. still do.
      **Known, intentionally out-of-scope residual**: mode 2b, `"I'm not saying we'll
      take you to court."` (cue in a discarded clause head), still blocks — needs
      forward-scoping negation, a different algorithm, not attempted here.
- [x] **C3 — widened escalation regexes ended live calls on non-triggers.** All four
      reproduced false positives fixed in `rings.py`, each with a true-positive check
      confirming the real trigger still fires:
      - `"Don't call my work number, use my cell."` — merged the two optional-trailer
        lines into one requiring a mandatory trailing word (`me`/`again`/`any more`/
        `back`); still fires on bare `"Don't call me."` (the canonical FDCPA phrasing).
        **Residual**: `"don't call me at work, use my cell"` still false-positives
        (literally contains "call me") — not chased further.
      - `"I laid off the sauce last year."` — deleted the redundant bare `laid off`
        line; the anchored `i got laid off` line still catches the real signal.
      - `"I can't handle this payment amount."` — made the `handle` branch's trailing
        `any more` mandatory, matching the `take` branch beside it.
      - `"This phone isn't mine, it's my wife's."` — **deviated from the originally
        planned fix** (see below).

- [x] **C4 — the R5 apology is spoken but never audited.** (Filed and fixed 2026-08-07;
      analysis in `OBSERVABILITY.md` §G1.) The `except` branch R5 added to `llm_node`
      (`voice_app.py:126-136`) yields `_TRANSIENT_ERROR_APOLOGY` straight to TTS
      without ever reaching `NegotiationAgent._record_spoken()`. Two consequences:
      the audit timeline shows agent silence where the consumer heard a sentence
      (`turn()` records the consumer utterance and increments `_turn_index` before
      any point it can raise), and `self.turns.append(turn)` at `agent.py:260` is
      skipped, so `CallEnded(turn_count=...)` and `CallReport.turns` undercount every
      errored turn. R5 correctly reasoned that nothing is left *inconsistent* for the
      next turn; it did not account for the record of this one. Fix: a narrow public
      method on `NegotiationAgent` that records the code-authored line and accounts
      for the turn, called via `await asyncio.to_thread(...)` — `AuditStore.record()`
      blocks on its worker thread and the `except` branch runs on the event loop.
      **Landed**: `NegotiationAgent.record_fallback_speech()` (`agent.py`), called off
      the event loop from `llm_node`'s `except` branch before the apology is yielded.
      Regression tests: `tests/test_voice_app.py` — the apology is recorded as an
      agent turn, `CallReport.turns` counts it, it reaches the model's context, and a
      successful turn is still recorded exactly once (the guard against double-recording).
      **Caveat**: the fix is working-tree-only. `voice_app.py` (the caller) and
      `tests/test_voice_app.py` are untracked in git while `agent.py` is tracked, so
      until they are committed together this checkbox does not describe the repository
      — a clone would get `record_fallback_speech()` with no caller and no test.

## Required

- [x] **R1 — `_consumer_context` fails open.** No dispatch metadata (local/manual
      testing) still falls back to the fixture consumer, as documented in
      `VOICE_QUICKSTART.md`. Metadata that IS present but malformed (bad JSON, non-dict,
      wrong-typed/empty fields) now raises `_ConsumerContextError`, caught in
      `entrypoint()`, which refuses the call instead of silently substituting a
      different real consumer's identity.
- [x] **R2 — `session.start()` recording was implicit.** Verified against the pinned
      `livekit-agents==1.6.8` (`inspect.signature`, matches current docs): `record:
      NotGivenOr[bool | RecordingOptions]`, defaulting to "record everything" (audio,
      transcripts, traces, logs uploaded to LiveKit Cloud) when omitted. Now passed
      explicitly as `record=False` in `voice_app.py`.
      **New stated assumption (same posture as A1–A6 in docs/archive/HANDOFF.md)**: this project's
      own `AuditStore` is the SPEC's compliance deliverable; nothing here collects
      consent for a second, third-party copy of consumer audio, so LiveKit Cloud's own
      observability recording is disabled rather than left to the SDK default. If a
      product/legal decision later wants LiveKit Cloud recording too, that needs an
      explicit consent flow (see the `docs.livekit.io` recording-consent recipe) — not a
      silent default flip.
- [x] **R3 — `tests/evals/` was in the default pytest collection.** Added a `pytest.mark.
      evals` marker (`tests/evals/test_scenarios.py`) and `addopts = "-m 'not evals'"`
      in `pyproject.toml`. Bare `pytest`/`uv run pytest` now excludes it (confirmed: 361
      run, 65 deselected); explicit `pytest -m evals tests/evals` still runs them.
- [ ] **R4 — `simulator.py` live path has zero test coverage** in this environment (no
      `ANTHROPIC_API_KEY` configured). Already documented as a caveat in
      `docs/archive/todo.md` step 8 / `docs/archive/HANDOFF.md`; not fixable without a live key. No action
      planned — tracked for visibility only.
- [x] **R5 — no `try`/`except` around `turn()` in `llm_node`.** Wrapped the
      `asyncio.to_thread(self._negotiation_agent.turn, ...)` call; on any exception,
      logs, speaks a short apology, and lets the call continue on the next turn (safe:
      `turn()`'s only mutations before a point it could raise are recording the
      consumer's utterance and updating guardrail state — nothing left inconsistent).

## Verified clean (no action)

- [x] C7 — replay returns a byte-identical payload, no duplicate `DecisionRecorded`.
- [x] C8 — dependencies check out; no secrets or real PII committed.

## Deviation from the original plan

**Issue 5 / "isn't mine"** — the DISPUTE fix originally planned as
`r"\b(?:this|that|it)\s+(?:isn'?t|is\s+not)\s+(?:even\s+)?(?:my|mine)\b"` breaks a
**pre-existing, already-passing** test (`tests/test_guardrails.py`:
`"This account isn't even mine."` must fire DISPUTE). The subagent that implemented this
fix caught the conflict via a full-suite run, consulted its own advisor, and chose to
preserve the existing test (a missed DISPUTE is a worse compliance failure than the
false positive being fixed) over the exact planned regex. Landed fix:
```python
r"\b(?:this|that|it)(?:\s+(?:account|debt|bill|balance))?\s+"

(r"(?:isn'?t|is\s+not)\s+(?:even\s+)?(?:my|mine)\b",)
```
— an optional debt-referent noun between the pronoun and the negation. Fixes the
reproduced false positive (`"This phone isn't mine..."` → clean) while keeping
`"This account isn't even mine."` firing. **Residual**: a noun outside the small
allowlist (`account`/`debt`/`bill`/`balance`) between the pronoun and `isn't` — e.g.
`"This charge isn't mine"` — still misses. Same posture as the C4/H4 scope notes already
in `ADVERSARIAL_TESTING.md`: the closed-enumeration regex approach is structurally
undersized for open-ended paraphrase; this pass closes the reproduced gaps, not the
category.

## Speech pipeline — LiveKit docs review (2026-08-07)

Reviewed `voice_app.py` against the `livekit-agents==1.6.8` source in `.venv`, per
`CLAUDE.md`. Findings below; the two marked `[x]` are landed.

- [x] **S1 — preemptive generation ran `turn()` on uncommitted transcripts.** The SDK
      default is `{'enabled': True, ..., 'max_retries': 3}`, and the preflight path
      reaches `Agent.llm_node` whenever `_vad_base_turn_detection` holds — which it does
      for the default `inference.TurnDetector()`. Verified in the installed source:
      `_pipeline_reply_task_impl` calls `perform_llm_inference(node=llm_node, ...)` at
      `agent_activity.py:3100`, ~100 lines *before* it awaits
      `speech_handle._wait_for_scheduled()` at `:3200`. For a pure text→text node a
      discarded generation costs only tokens; `turn()` advances `_turn_index`, records a
      `ConsumerUtterance`, and can move the concession ladder. And `llm_node` runs it
      under `asyncio.to_thread`, which **cannot be cancelled** — the SDK cancels the
      `SpeechHandle` while `turn()` completes anyway. "I can do two fifty" finalizes, the
      ladder moves, "…a month" arrives, the generation is invalidated, the ladder moves
      again. **Landed**: `TURN_HANDLING` in `voice_app.py`, passed to `AgentSession(...)`.
      Tests in `tests/test_voice_app.py` assert the constant and that `entrypoint()`
      wires it — no test in this repo can reach the live path, since preemptive
      generation fires from `audio_recognition.py` and needs an audio stream, while the
      suite runs the session in text mode. **Not verified on a live call.**
- [x] **S2 — prompt examples are unsayable above 8 words.** `check_outbound` defaults
      `confidential_reference` to `SYSTEM_PROMPT`, and `_LEAK_MIN_WORDS = 8`, so any
      8-word run written into the prompt becomes blockable as leaked prompt text. This
      makes a "Pauses and filler words" section a trap: an example phrase written there
      as a model of good speech is exactly the phrase the model reproduces. Caught in
      practice — a draft quoting `"Yeah, um... so, here's what I can do."` in full tripped
      `CONFIDENTIAL_TEXT_LEAKED` when spoken back. **Landed**: the section's examples are
      kept to a word or two, and `tests/test_guardrails.py` pins it (red against the
      original draft, green now).
      Related: SSML `<break time="300ms"/>` trips `SUSPICIOUS_DIGIT_BOUNDARY` on the
      numeric guard — the digits inside the tag read as an unauthorized figure. The
      docs conflict here (the prompting guide says "Cartesia supports SSML directly";
      the Cartesia page documents only IPA overrides and pronunciation dictionaries, and
      never mentions `<break>`), but it is moot: tags are disqualified for this agent
      regardless, and the prompt already forbids them.
- [ ] **S3 — `allow_interruptions=False` on the opening silences STT.** With the default
      `interruption.discard_audio_if_uninterruptible: True`, `agent_activity.py:1365-1397`
      substitutes silence frames on the STT path while an uninterruptible speech plays.
      VAD still sees real audio; STT does not. A consumer saying "wrong number" or "this
      isn't Dana" during the mini-Miranda is never transcribed — gone, not delayed. That
      trades directly against the identity gate C1 added. Reading the disclosure through
      may well be the compliant choice; the point is that it is currently an implicit
      default rather than a decision. `{"interruption": {"discard_audio_if_uninterruptible":
      False}}` is the lever. **No action taken — needs a product/compliance call.**
- [ ] **S4 — `endpointing.max_delay` is 2.5s.** Because a streaming turn detector is
      active, `_resolve_endpointing` selects `_STREAMING_ENDPOINTING_DEFAULTS`
      (`{'mode': 'fixed', 'min_delay': 0.3, 'max_delay': 2.5, 'alpha': 0.9}`) — not the
      `0.5 / 3.0` the tuning page's table lists, which is stale for this configuration.
      A consumer doing mental arithmetic past 2.5s is cut off, and half of "two fifty a
      month" is a different number. Raising it costs up to 1.5s more worst-case dead air.
      A feel judgment; `_log_turn_latency` already emits the data to decide on.
- [x] **S5 — Deepgram keyterm biasing.** `deepgram/nova-3` reports `keyterms=True`;
      `extra_kwargs={"keyterm": [...]}` merges through `_keyterms_extra_for_model`
      (`inference/stt.py:210-247` maps any `deepgram/` model onto the `keyterm` extra;
      ceiling 100 terms / 1200 chars per `:79`). **Landed**: `STT_KEYTERMS` in
      `voice_app.py`, passed to `inference.STT(...)` in `CollectorAgent.__init__`.
      Three groups, each earning its place: cadence (`biweekly` — routinely heard as
      "by weekly", and a wrong cadence is a wrong schedule), escalation triggers
      (`cease and desist`, `identity theft`, `validation`, `retained`, `attorney`,
      `bankruptcy`, `garnishment`, `chemo`, `terminally`, `suicidal`, `disability` —
      each verified to fire `detect_escalation`), and domain vocabulary. Common words
      Nova-3 already handles are deliberately excluded: boosting the common case buys
      nothing and dilutes the terms that need it.
      **The rule that is not a preference**: no dollar amount, ever. Biasing the
      recognizer toward the figures this agent is authorized to offer would bias
      transcription of what the *consumer* said toward those same figures —
      manufacturing agreement to a number they never spoke, upstream of every guardrail
      and invisible in the audit log, since the false figure would be recorded as what
      they said. Pinned by `test_no_keyterm_carries_a_figure`, verified red against an
      injected `"$250"`. Effect on live recognition accuracy is **not measured** here —
      no audio fixtures in this environment; the tests pin the configuration, not the
      transcription quality.

Rejected, with reasons: **Cartesia `emotion`** applies persuasion pressure the
prohibited-persuasion guard inspects *text* to catch — an unauditable channel downstream
of the guard. **`tts_text_transforms=[text_transforms.replace(...)]`** runs after
`llm_node` yields, i.e. mutates guardrail-approved text after approval.
**`keyterm_detection`** forwards up to 12 messages of consumer speech to a second
third-party model per turn — the posture R2 declined. **Noise cancellation** is the
biggest remaining STT-accuracy lever for telephony but processes raw consumer audio on
LiveKit Cloud; same consent decision, not a tuning one. **The turn-detector plugin** is
redundant — `AgentSession()` already instantiates `inference.TurnDetector()` and Silero
VAD, so the docs' "recommended starting config" is a no-op here.
**`use_tts_aligned_transcript`** is currently a no-op (Cartesia reports
`aligned_transcript=False` without `add_timestamps`) and cuts no latency regardless.
**TTS caching** cannot apply to the opening — `open_call()` generates it per call.

## Notes

- The tree was moving during the original review (a concurrent session editing
  `agent.py`/`disclosures.py`/`numeric.py`, and separately `text_app.py` +
  `llm/openrouter_client.py`); none of those files were touched by this remediation pass.
- A separate `Issues.md` (capital I) used to sit beside this file, holding the raw review
  transcript as pasted. It was deleted in the 2026-08-07 docs cleanup: the paste was
  corrupted mid-transfer (truncated and word-mangled in its last third), so it could not
  serve the provenance purpose it was kept for. This file is the maintained tracking doc.
