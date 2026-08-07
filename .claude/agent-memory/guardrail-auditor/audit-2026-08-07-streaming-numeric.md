---
name: audit-2026-08-07-streaming-numeric
description: Confirmed guardrail bypasses and false positives found auditing branch `develop` (streaming path + tightened numeric guard) on 2026-08-07, with exact reproductions.
metadata:
  type: project
---

Audit of `develop` (c903743, 45065ea, 1107e83 on c4df454) in worktree
`lexical-greeting-tower`. All findings reproduced against live code; the 337-test
suite passes with every one of them present.

**Why:** the suite's fixtures (`MINI_MIRANDA_TEXT`, `AI_DISCLOSURE_TEXT`) are each a
*single sentence*, so per-sentence guarding is never exercised against multi-sentence
phrasing. That is the structural reason the streaming defects are invisible to CI.

**How to apply:** re-check these before any future pass on `agent.py` streaming or
`guardrails/numeric.py`; do not re-derive.

## Confirmed decision-layer failure (worst)
- `decision_engine.py::_evaluate` `NO_UNAUTHORIZED_DISCOUNT` uses
  `proposal.total >= policy.original_balance`; SPEC §4.3 requires `==`. No upper bound
  anywhere. Repro end-to-end with `MockLLMClient`: consumer says
  *"I can do a thousand a month for three months."* → `parse_proposal` → total $3,000,
  3 monthly → verdict **accept**, `confirm_agreement` ok, agreement record written for
  **$3,000 on a $1,000 debt**. Pre-existing (decision_engine unchanged in this diff).

## Confirmed detection-layer bypasses
- **Consumer-figure laundering**, `numeric.py::_from_verdict` harvests
  `f"{condition.actual} {condition.limit}"`. `actual` is consumer-originated. Proposal
  $25/wk × 40 → authorized money gains `$25.00`, counts gains `40`, durations gains
  `273`. Isolated from the counter offer (counter only yields 250/1000). Agent may then
  say "I could do $25 a week" / "we could do 40 payments". Pre-existing mechanism, but
  now load-bearing because the base set was tightened onto it.
- Harvesting `limit` re-authorizes $250/$800/$1,000/92-days on the **first**
  `validate_consumer_offer`, so the tightening delays policy limits by one tool call
  rather than gating them on an offer.
- **Date demotion across a sentence split**, `agent.py::_split_sentences` +
  `_SENTENCE_END`. `"Your first payment is due Jan. 15."` splits at the abbreviation →
  `"...due Jan."` + `"15."`; `15` is BARE < `_MONEY_INFERENCE_FLOOR` → WARN not BLOCK.
  Whole-turn guard blocks it as `UNAUTHORIZED_DATE`. Nothing in `agent.py` ever passes
  `extra_dates`, so `authorized.dates` is always empty — the asymmetry is real.

## Confirmed streaming-only false positives (both abort to SAFE_FALLBACK_TEXT)
- **AI-disclosure-on-request does not compose per sentence**
  (`disclosures.py::check_agent_turn` `AI_DISCLOSURE_REQUEST_IGNORED`). Disclosure in
  sentence 2 → sentence 1 blocked → abort. `_speak_verbatim` bypasses `observe_agent`,
  so `ai_disclosure_requested` never clears → **the call wedges permanently**: every
  later streamed turn emits only the fallback. `finalize_call` still scores
  `compliant=True`. The authors protected only the at-open rule (`open_call` left
  synchronous); they missed this one.
- **Two-sentence Mini-Miranda** — `fires_mini_miranda` needs both halves co-located.
  `"This is an attempt to collect a debt." / "Any information obtained will be used for
  that purpose."` → sentence 1 is substantive, fires nothing → `MINI_MIRANDA_NOT_FIRED`
  → abort. Passes whole. Falsifies the `stream_turn` docstring claim that the
  Mini-Miranda "does compose sentence by sentence": the *ordering* composes, the
  *detector* does not.
- **"over three months"** on an engine-authored $800/3-monthly settlement: offsets are
  0/30/60 → `duration_days=60`, so 90 days is unauthorized → BLOCK.

## Escalation false positives (A6 terminates the call)
15 of 16 ordinary haggling utterances escalate. `detect_escalation` has **no negation
handling at all**, unlike `scan_prohibited`'s `_is_negated`. Worst offenders:
`\bgiving\s+up\b` ("I'm not giving up"), `\bfalling\s+apart\b` ("my car is falling
apart"), `\bnothing\s+left\b`, `\bi'?m\s+drowning\b` (routes to the *suicide* script),
`\bbehind\s+on\s+everything\b`, `\b(?:got\s+)?laid\s+off\b` (no recency).

## Held under pressure — allocate less time next pass
- Tool argument boundary (`tools.py::Param.parse`): dict/float/negative/unknown-cadence
  all rejected or safely coerced; no smuggling path found.
- `_HARD_FLOORS` reject logic, tier monotonicity (A4), installment summation, cadence,
  duration: 0 violations across a 320-case sweep and 0 illegal acceptances below floor.
- Escalated call cannot resume: `_escalate` sets `ended=True`, `stream_turn` raises.
- No ReDoS: all patterns linear (<15ms on 30 KB adversarial inputs).
- Base `authorized_for` tightening works — $250/$800/20%/3-months/4-payments all blocked
  before any tool call.

## Lesser, still real
- `MAX_PROPOSED_PAYMENTS = 1000` (`tools.py:52`) does block the engine from ruling:
  "$5/week for twenty years" → 1042 → `ArgumentError`, no verdict, no counter.
- Strike counter leaks across paths: a blocked streamed sentence leaves `strikes=1`, so a
  later `turn()` hits the fallback after one block instead of two.
- Streaming fallback is appended to `self.messages` twice (`_speak_verbatim` then the
  joined `spoken`).
- Identity confirmation cannot be revoked (`agent.py::_perceive` only ever sets it true;
  no `with_identity_denied`). Agent quotes the balance after "that's not me". Pre-existing.
