# Plan — caller-experience and commitment-integrity follow-ups

Source: review commentary handed to `/build` (repeat-back confirmation, post-call
artifact, latency/dead-air, Gemini certification).

## Task 1 — Gate commitment on an engine-authored repeat-back — DONE (129c6d0)

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

## Task 5 — Tighten the call opening — DONE

Three items from a second round of commentary, all describing what the *live*
model does rather than what the mock does. The mock already merged greeting,
AI disclosure and identity into one breath; `SYSTEM_PROMPT` never told the live
model to, and had no section about the opening at all. That was the hole.

* Greeting, AI disclosure and the identity question go out in one breath.
* The Mini-Miranda now rides out with the balance and the ask instead of
  costing a round trip of its own. `_run_tool` re-points `self.authorized`
  before the turn speaks, so the figure is authorized by the time the merged
  sentence is guarded.
* No more asking permission to ask. "Can we talk about getting that resolved?"
  is a yes/no gate whose "no" changes nothing, so it buys nothing and costs a
  round trip.

`test_disclosures_fire_in_order_and_before_any_substance` had to be repaired
rather than relaxed: it asserted the Mini-Miranda was in an *earlier turn* than
any figure, which was only ever a proxy for the real rule. Collapsing the turns
made a correct call fail it. It now compares offsets, which is what the guard
itself compares.

### Left open — needs a decision

Dropping "you can ask for a human at any time" from the opening was **not**
done. It would mean editing `AI_DISCLOSURE_TEXT`, which is a compliance script,
and `agent.py:1020` prepends that same constant on the safe-fallback path —
the path that speaks when the guard has already blocked the model twice.
Shortening the constant silently narrows what the fallback discloses, on
exactly the turns where the full text matters most. Asked separately.

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
