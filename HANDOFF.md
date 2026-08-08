# Handoff: live-call negotiation fixes, disclosure repair, and a route migration off LiveKit Inference

**Generated**: 2026-08-08 20:56 UTC
**Branch**: `integration` (`main` is identical — both at `159f386`, both pushed)
**Status**: In Progress — five defects fixed and deployed; three open, none of them compliance-visible speech

## Goal

Started from a live transcript where the agent ignored the consumer's "two payments of $500 each" and confirmed its own $250/$750 instead. Widened as more live calls surfaced: the agent falling back to a non-sequitur after identity confirmation, then flattening an uneven payment split, plus getting production running at all after LiveKit Inference credits ran out.

## Completed

- [x] **`$500 × 2` overwritten by the standing offer** (`aa895d7`) — `tools.py` guard swapped a consumer-named split for the standing schedule. Now stands down when `proposal.amount_each` is set.
- [x] **Mini-Miranda fallback** (`8c41068`) — agent said `SAFE_FALLBACK_TEXT` after "Yes." on every call. Deterministic repair prepends the canonical notice.
- [x] **`$400/$600` flattened to `$500/$500`** (`96b0519`) — `Offer.from_proposal` split evenly regardless of the consumer's own first payment.
- [x] **LLM route moved to OpenRouter + DeepSeek V4 Flash** (`4eb7d0f`) — LiveKit Inference credits exhausted.
- [x] **STT/TTS moved to Deepgram + Cartesia plugins** (`cc2ad44`) — same exhausted pool was killing speech in both directions.
- [x] Production deployed and smoke-tested end-to-end: zero 429s, TTS confirmed synthesizing (`sonic-3`, `characters_count: 6` = `"Hello."`).
- [x] **Agent accepted without saying the figures** (`159f386`) — deterministic read-back on both `turn()` and `stream_turn`, suppressed on a scripted close. See Key Decisions.
- [x] **`signaled_capacity` was dropped by the relay** (`159f386`) — the tool description read as an affordability *ceiling*, so a named first payment was omitted and the uneven-split fix above was fed `None`, re-flattening the consumer's schedule. Live on DeepSeek V4 Flash: relayed 2/10 before, 8/8 after.

## Not Yet Done

- [ ] **Model invents balances.** Produced `$1,573.97` against a $1,000 account. Numeric guard catches it every time (correct), but it costs the turn → fallback. Cheapest fix: have the regeneration note state the *authorized* balance, not just prohibit invented ones (`agent.py` ~line 1283, `"...do not restate any figure the engine has not returned."`).
- [ ] **`tests/evals/` never run against DeepSeek V4 Flash.** Route is uncertified; `voice_app._llm_client` docstring says explicitly it must not carry production on a smoke test alone. Six trials ≠ certification.
- [ ] **Audit storage still ephemeral** (pre-existing, from previous handoff) — SPEC §6 compliance record writes to container disk, lost on every redeploy.
- [ ] Lower LLM latency — see Failed Approaches; no verified win yet.

## Failed Approaches (Don't Repeat These)

**1. Disabling `reasoning` on OpenRouter to cut latency.** ~2× faster, but A/B on the real turn (6 reps each):

| | Reached consumer | Invented a figure | Median |
|---|---|---|---|
| `reasoning: {effort: "low"}` | 6/6 | 2/6 | 9528 ms |
| `reasoning: {enabled: false}` | 2/6 | 4/6 | 4939 ms |

Halves latency, guts accuracy, doubles the invented-balance defect. **Reverted.** Do not re-try without re-running this A/B (`scratchpad/reasoning_ab.py` pattern).

**2. Hard-pinning OpenRouter to BaseTen** (`provider: {order: ["baseten"], allow_fallbacks: false}`) — BaseTen is genuinely fastest (p50 323 ms vs 570–1000 ms across the other 23 providers) but returns `429 Provider returned error` under a hard pin. With `allow_fallbacks: true` it did not beat default routing in benchmark. Unresolved.

**3. Switching LiveKit Inference to Gemini 3.1 Flash Lite** because the dashboard showed it "available". The UI shows *catalog membership, not credit*. Tested `gemini-3-flash-preview`, `gemini-3.1-flash-lite`, `gemini-3-flash`, `gemini-2.5-flash-lite` — all identical `429 MaxGatewayCredits`. The credit pool is model-agnostic ($2.50/month, Build plan, shared across LLM+STT+TTS **and** across all the user's free projects, resets 1st of month, no rollover). No model swap inside LiveKit Inference can help.

**4. Quoting the Mini-Miranda in the system prompt** so the model would say it verbatim. Cannot be done: `test_no_required_script_reads_as_a_prompt_leak` forbids it, because 8 shared words make *speaking* the line trip `CONFIDENTIAL_TEXT_LEAKED`. Verified the trap is real. This is why the repair is deterministic code, not prompt text.

**5. Extending the Mini-Miranda repair to `MINI_MIRANDA_OUT_OF_ORDER`.** Scored better on raw reach (2/3 vs 1/3) but produced a stutter: *"This is an attempt to collect a debt... This is a communication from a debt collector. This is an attempt to collect a debt..."* — that rule means a complete notice is already present, just late. Also rejected an "excise and move the notice" variant: the span from attempt-clause start to purpose-clause end swallows any substantive text between the halves (`"This is an attempt to collect a debt. Your balance is $1,000. Any information..."`).

**6. `lk agent update-secrets COLLECTOR_LLM=openrouter`** — bare `KEY=VALUE` parses as the *working-dir* arg, silently falls back to `.env`, and uploads everything in it. Then `--secrets ... --overwrite` **replaces the entire secret set** rather than merging, and comma-separated pairs in one `--secrets` only registered the first. Correct form is repeated flags:
```bash
lk agent update-secrets --overwrite --secrets K1=V1 --secrets K2=V2   # ALL secrets, every time
```

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Deterministic Mini-Miranda prepend, not prompt steering | Wording is fixed by law; prompt can't quote it (see Failed #4). Matches repo principle: "The LLM talks. Deterministic code decides." |
| Repair only when the notice is **absent** | `OUT_OF_ORDER` means one is already there → stutter (Failed #5) |
| Repair refuses turns carrying `UNAUTHORIZED_AMOUNT` | Prefixing a disclosure onto an invented figure launders it past the guard that caught it |
| Uneven-split fix lives in `tools.py`, not `offers.py` | The payment floor decides it, and `offers.py` can't see `PolicyConfig`. MIN_PAYMENT was ruled against the *even* split, so a front-loaded schedule is unchecked until `tools.py` |
| Read-back is code-supplied, not prompt-steered | Same reasoning as the Mini-Miranda: the model was asked for the figures and did not produce them, and the guard has no rule for what a sentence omits |
| The read-back is never appended to `SAFE_FALLBACK_TEXT`/`CONNECTIVE_TEXT` | Live trial produced "...What would work for you? Just to confirm: $1,000.00 today." — the recovery line reopens the negotiation and the read-back shuts it, in one breath. The debt stays owed for the next real turn |
| `_accepted` is held across turns, not reset per turn | A turn that swallows the terms does not cancel the obligation to say them |
| `deepseek-v4-flash-0731` pinned, not `~...-latest` | The alias re-points to whatever ships next — swapping the live model with no deploy and no trial |
| STT/TTS on direct plugins | One pooled LiveKit allowance covering LLM+STT+TTS is a single point of failure for the whole voice path |
| Reverted the concurrent session's prompt rewrite (`b33e4d9`) before re-landing it scoped (`410ecab`) | It broke production 0/4; reverting first stopped the bleeding, then it was re-applied with the never-compress rule scoped to the AI disclosure only |

## Current State

**Working**: Production `pQcZDJSPA7Pn` @ `159f386`, healthy. LLM = OpenRouter/DeepSeek V4 Flash. STT = Deepgram `nova-3` plugin. TTS = Cartesia `sonic-3` plugin. Nothing in the call path touches the exhausted LiveKit credit pool. 955 tests, `ruff`, `mypy` all clean.

**Broken**: Invented balances (see Not Yet Done). LiveKit Inference credits remain at zero until the 1st of the month — the turn detector and adaptive-interruption models are free on deployed agents, so this no longer blocks calls, but any code re-introducing `inference.STT`/`inference.TTS`/`inference.LLM` will 429 instantly.

**Uncommitted Changes**: None. Working tree clean, `main` == `integration` == `96b0519`, both pushed.

## Files to Know

| File | Why It Matters |
|------|----------------|
| `src/collector/tools.py` | `_validate_consumer_offer` — the standing-offer guard, `_consumer_led_schedule`, accepted-offer construction. Both offer bugs lived here |
| `src/collector/agent.py` | `_guard_and_speak` (~1229) — regeneration loop + the Mini-Miranda repair; `_only_missing_notice` |
| `src/collector/decision_engine.py` | Pure verdict layer. `effective_capacity`, `build_counter`, `_allocate` (leads a counter with a named capacity — the rule `_consumer_led_schedule` mirrors) |
| `src/collector/voice_app.py` | STT/TTS wiring (~375), `_llm_client` route dispatch (~331) |
| `src/collector/llm/openrouter_client.py` | `MODEL`, `EFFORT` — do not set reasoning to disabled (Failed #1) |
| `src/collector/guardrails/disclosures.py` | `MINI_MIRANDA_TEXT` + the regexes that decide whether it fired |

## Code Context

**The repair that fixes the fallback** (`agent.py`, inside `_guard_and_speak`'s loop, before `self.guard = check.state`):
```python
if not check.allowed and not completed_notice and self._only_missing_notice(check):
    candidate = f"{MINI_MIRANDA_TEXT} {candidate}"
    completed_notice = True
    continue  # re-guard; no strike, no model round-trip


@staticmethod
def _only_missing_notice(check: OutboundCheck) -> bool:
    rules = {v.rule_id for v in check.blocking_violations}
    return rules == {DisclosureRuleId.MINI_MIRANDA_NOT_FIRED}  # NOT out-of-order
```

**The uneven-split fix** (`tools.py`) — returns `None` whenever their split can't be written legally:
```python
def _consumer_led_schedule(proposal, policy) -> tuple[Money, ...] | None:
    down = proposal.signaled_capacity
    if down is None or proposal.payment_count < 2 or down < policy.min_payment:
        return None
    remainder = proposal.total - down
    if remainder.amount <= 0:
        return None
    rest = remainder.allocate(proposal.payment_count - 1)
    if min(rest) < policy.min_payment:
        return None
    return (down, *rest)
```
The standing-offer guard now requires `proposal.amount_each is None and not consumer_led`.

**The detector the model kept missing** — requires this literal phrase:
```python
_MINI_MIRANDA_ATTEMPT_RE = re.compile(r"attempt\s+to\s+collect\s+a\s+debt", re.IGNORECASE)
```
`"This call is from a debt collector"` does NOT match → `MINI_MIRANDA_NOT_FIRED`.

## Resume Instructions

1. Verify baseline: `.venv/bin/python -m pytest tests -q -p no:randomly && .venv/bin/python -m ruff check . && .venv/bin/python -m mypy`
   - Expected: `955 passed`, `All checks passed!`, `Success: no issues found in 28 source files`
2. Reproduce the silent-accept bug offline (no API cost):
   ```bash
   set -a; . .env; set +a
   COLLECTOR_LLM=openrouter .venv/bin/collector-text --livekit --no-store --verbose
   ```
   Say `Yes.` → `No.` → `I can pay four hundred today and six hundred next month.`
   - Expected: confirmation names `$400.00` and `$600.00`
   - If the agent accepts without figures: that's the open bug — the model got `you_must_confirm` and ignored it
3. For live trials, use the harness pattern from this session: monkeypatch `rings.check_outbound` to collect `blocking_violations`, run N agents through `open_call()` + `turn("Yes.")`, count how many spoke vs hit `SAFE_FALLBACK_TEXT`. Always report the count; a single trial proves nothing on a stochastic route.
4. Deploy only after tests + a live trial batch:
   ```bash
   git push origin integration && git push origin main
   lk agent deploy            # builds from the WORKING DIRECTORY, not HEAD
   lk agent versions          # match on git_commit
   ```
5. Smoke test: `timeout 75 lk agent logs > /tmp/x.txt &` then
   `lk dispatch create --agent-name collections-negotiator --room smoke-<sha>`, then `lk room delete smoke-<sha>`
   - Expected: `llm_route: "openrouter"`, a `tts_usage` line with `model: "sonic-3"`, zero `429`
   - The `tts_usage` metric only flushes at session teardown — delete the room to see it

## Setup Required

- `.env` needs: `OPENROUTER_API_KEY`, `CARTESIA_API_KEY`, `DEEPGRAM_API_KEY`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL`
- `ANTHROPIC_API_KEY` is present but **capped until 2026-09-01** (`400 ... regain access on 2026-09-01`). `--claude` and judge-backed evals will fail until then.
- Deployed agent secrets (5): `COLLECTOR_LLM=openrouter`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `CARTESIA_API_KEY`, `DEEPGRAM_API_KEY`. Never set `LIVEKIT_URL`/`API_KEY`/`API_SECRET` as secrets — injected at runtime.

## Edge Cases & Error Handling

- Consumer names a downpayment that starves the tail (`"$800 today"` on $1,000, $250 floor) → `_consumer_led_schedule` returns `None`, even split stands. Covered by `test_a_named_downpayment_never_starves_the_tail`.
- Consumer says `"two payments"` with no figure → still resolves to the standing offer, not an even split. The guard only stands down for consumer-*authored* figures.
- Model invents a figure on a turn also missing the notice → repair declines, both guards fire, turn is lost to the fallback. Intentional.
- Model relays `total` and `amount_each` together → tool returns an error rather than guessing.

## Warnings

- **`lk agent deploy` builds from the working directory, not `HEAD`.** Uncommitted and untracked files ship. This is how an unreviewed prompt rewrite reached production this session. Deploy a clean tree, or `lk agent deploy /path/to/worktree`.
- **A second Claude session is working in this same checkout** (peer at `uds:/run/user/1000/cc-socks/2191478.sock`), and owns `llm/base.py` prompt wording. It has committed its work, but check `git status` before staging — **stage explicit paths, never `git add -A`**.
- **A formatter hook strips imports added before their first use.** Add the usage in the same edit, or re-add the import afterwards. Bit this session twice (`Installment`, `OutboundCheck`).
- **Never paste 8+ contiguous words of the system prompt into agent speech** — blocks as `CONFIDENTIAL_TEXT_LEAKED`. Applies to example sentences written *into* the prompt too.
- `main` was never force-moved past `b8e5589`; it was merged then fast-forwarded so both branches share one head. Keep doing that rather than `git branch -f` onto a diverged main.
- The `$2.50` LiveKit allowance is shared across **all** the user's free projects and resets the 1st. Upgrading to Ship ($50/mo) is the only way to restore it sooner.
