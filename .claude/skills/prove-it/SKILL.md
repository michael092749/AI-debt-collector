---
name: prove-it
description: 'Fix a bug by first writing a failing test that reproduces it, confirming it fails for the stated reason, then fixing until it passes. Use for any bug fix in this repo — "fix the bug", "this is broken", "X should not happen but does" — especially in guardrails/, decision_engine.py, or negotiation.py where an unproven fix is a compliance risk, not just a code smell.'
---

# Prove-It

This is the bug-fix workflow already implied by `CLAUDE.md` §4 (Goal-Driven Execution — "Fix the
bug" → "Write a test that reproduces it, then make it pass") and used explicitly in
`ADVERSARIAL_TESTING.md`'s recommended remedy process. This skill makes it a named, repeatable
step instead of something re-explained each time.

**Rule:** no fix without a test that proves both the bug and the fix. A fix without a reproducing
test is a guess, not a fix — the guess might make the reported symptom disappear while leaving
the actual defect in place, or introduce a regression the next merge doesn't catch.

## The loop

1. **Reproduce.** Write the smallest test that demonstrates the bug — the exact input sequence
   that triggers it, asserting the *correct* behavior (not the current buggy behavior). Put it in
   the relevant existing test file under `tests/` (e.g. `test_guardrails.py` for a guardrail
   issue, `test_decision_engine.py` for a decision-engine issue) rather than a new scratch file.

2. **Confirm red.** Run it and confirm it fails *for the reason you think it fails*, not for an
   unrelated reason (import error, wrong fixture, typo in the assertion).
   ```bash
   uv run pytest tests/<file>.py::<test_name> -v
   ```
   If it passes immediately, the bug isn't reproduced yet — the test is wrong, not the code.

3. **Fix.** Make the minimal change that addresses the root cause, not the symptom. Per
   `CLAUDE.md` §3 (Surgical Changes): touch only what the fix requires, match existing style,
   don't refactor adjacent code while you're in there.

4. **Confirm green.** Re-run the specific test, then the full suite:
   ```bash
   uv run pytest tests/<file>.py::<test_name> -v
   uv run pytest
   ```

5. **Check the rest of the toolchain**, since `pyproject.toml` enforces both:
   ```bash
   uv run ruff check src/collector
   uv run mypy --strict src/collector
   ```

## Why this matters more here than in a typical repo

This project's entire premise (`README.md`) is that every dollar figure and every compliance
statement is enforced by code, not by a transcript that "looks right" on manual review. A fix
that isn't backed by a test proving the specific failure mode is exactly the gap this project
exists to close. Don't special-case the fix process just because a bug feels obvious or small.

## When not to use this

For a change with no bug to reproduce — a new feature, a refactor, a doc update — this skill
doesn't apply. Use it specifically when something is broken and needs to stay fixed.
