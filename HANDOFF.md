# Handoff: merge, fix offers, deploy

**Goal**: Get the per-payment offer fix ($500×2 → $250/$750 bug), capacity guard, and refined opening merged into main and deployed to LiveKit Cloud.

**Done**: amount_each param on validate_consumer_offer, concede capacity guard + rotating mock capacity-asks, token counts on llm_respond log line, opening prompt rewritten as three beats and reconciled with main's hold-the-human-offer rule, merged 27-commit main line with 3-commit gemini-route-switch (kept main's streaming, dropped our duplicate), 852 tests + ruff + mypy green, two live collector-text --claude verifications, main fast-forwarded to aa378a6, deployed as h4fbdNfgqiWk (git_commit aa378a6 confirmed), post-deploy smoke showed warm cache hit (cache_read 4307) and zero blocks.

**Next**: Decide on durable audit storage — production still writes the SPEC §6 compliance record to ephemeral container disk, so every call is lost on redeploy; needs external Postgres (or equivalent) plus credentials via `lk agent update-secrets` (which itself restarts the agent).

**Watch out**: main is checked out in `.claude/worktrees/calm-churning-gray` with another session's uncommitted `rings.py` change — don't remove or force over that worktree; the primary checkout sits on `integration` (identical to main), and any 8 contiguous words copied verbatim from the system prompt into agent speech get blocked as CONFIDENTIAL_TEXT_LEAKED, so never paste example opening sentences into the prompt.
