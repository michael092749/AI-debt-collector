# Plan — caller-experience and commitment-integrity follow-ups

Source: review commentary handed to `/build` (repeat-back confirmation, post-call
artifact, latency/dead-air, Gemini certification).

## Task 1 — Gate commitment on an engine-authored repeat-back — IN PROGRESS

A commitment is recorded by `tools.py::_confirm_agreement` (`state.agree(offer)`,
tools.py:644). Nothing today requires the agent to have read the terms back to
the consumer first. Repeat-back is at best a prompt habit, and a habit is not a
control — `guardrails/disclosures.py:10-12` makes exactly this argument about
disclosures.

Mirror the disclosure pattern: the engine authors a canonical confirmation line
next to the detector that verifies it, and the tool is refused until the line
has actually been spoken.

Acceptance criteria:

1. `confirmation_line(offer)` renders a canonical, clock-free confirmation
   sentence from an engine-authored `Offer`, using only figures the numeric
   guard already authorizes for that offer (`numeric._from_offer`).
2. `repeats_back(text, offer)` is true only when the utterance states *every*
   installment amount and carries a confirmation cue. Numbers matched exactly
   (via the existing `extract_figures`), prose matched leniently.
3. `confirm_agreement` is refused — no verdict, no `agree()` transition, no
   agreement recorded — unless a previously *spoken* agent turn repeated back
   the standing offer.
4. The refusal payload hands the model the canonical line to say.
5. Tool payloads that put an offer on the table carry `you_must_confirm`
   alongside the existing `you_may_say`.

## Task 2 — Post-call artifact — PENDING

Emit a durable per-call artifact at call end. `audit/store.py` and
`CallSummary` already hold the material; scope is the artifact's shape and
where it is written, which is undecided.

## Task 3 — Acknowledgment filler for dead air — PENDING

A short "let me check that" before the LLM round trip, masking the 6.7s median.
Deferred deliberately: it changes what the caller hears on a collection call and
interacts with disclosure ordering (`check_agent_turn` runs on every candidate),
so it is not the free UX win it looks like.

## Task 4 — Certify the Gemini route — NOT APPLICABLE

The premise does not hold on this branch. `git log main..HEAD` is empty:
`gemini-route-switch` is identical to `main`, so there is no route switch here
to certify. The certification also already happened — 03217e2 pointed the eval
harness at the real route, Gemini failed numeric provenance and disclosure
ordering 3/3, and production was rolled back to `COLLECTOR_LLM=anthropic`.
Re-running `tests/evals/` against Gemini would re-derive a conclusion the repo
has already acted on.

## Note on the token-logging item

The commentary calls `agent.py:766-774` unfinished. It is not: `_ask_model`
already logs `elapsed_ms` with `turn` and `label`, and `_record_model_call`
already writes tokens, cost and stop reason to `ModelCalled`. No task opened.
