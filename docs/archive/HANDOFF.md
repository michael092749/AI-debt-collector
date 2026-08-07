# Handoff: Voice quickstart + OpenRouter backend

**Goal**: Update `VOICE_QUICKSTART.md` off the deprecated `collector-voice dev` command, and add/test an OpenRouter backend option for the voice worker.

**Done**: `VOICE_QUICKSTART.md` step 5 now uses `lk agent dev src/collector/voice_app.py` (verified connects/registers cleanly); `voice_app.py` got a `_llm_client()` helper so `COLLECTOR_LLM=openrouter` routes through `OpenRouterClient` instead of `AnthropicClient` (lints clean, worker starts fine either way); OpenRouter reliability spot-tested via `collector-text --openrouter` — failed to produce the mandatory AI-disclosure opening line in 3 of 5 runs (guardrail fallback fired instead), documented as a caveat in the quickstart; all six `.env` keys plus `OPENROUTER_API_KEY` are filled in.

**Next**: Fixed — `_to_anthropic()` now appends a synthetic tagged `<call_started>` user turn when the mapped conversation comes out empty (i.e. `messages` held only the system prompt, as on `open_call()`'s first `respond()`). Proven with `tests/test_agent_loop.py::TestAnthropicMapping::test_the_opening_call_with_only_a_system_prompt_still_has_a_message` (red before the fix, green after); full suite (383 passed, 1 pre-existing unrelated failure in `test_openrouter_client.py::test_constructing_without_a_key_raises`), ruff, and mypy --strict all clean.

**Watch out**: Don't treat `COLLECTOR_LLM=openrouter` as a safe substitute for Anthropic — it's meaningfully less reliable on the compliance-critical disclosure line, not just a napkin-math latency tradeoff. Also, `lk agent dev` needs `LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET` exported into the actual shell env (`set -a; source .env; set +a`) — it does not read `.env` itself the way `voice_app.py` does via `python-dotenv`.
