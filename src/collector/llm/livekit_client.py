"""LiveKit Inference client — a third `LLMClient`, alongside Anthropic and OpenRouter.

Reaches LiveKit Cloud's inference gateway (`agent-gateway.livekit.cloud/v1`),
which is an OpenAI-compatible chat-completions endpoint, so the transcript and
tool mapping is the shared one in `openai_shape.py`. Three things differ from
the OpenRouter route:

1. **Auth is a locally minted JWT, not a static key.** LiveKit signs an access
   token carrying an inference grant from `LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`
   — the same credentials the voice worker already uses to join a room, so this
   route adds no third-party key to the deployment. The token is short-lived and
   re-minted on every `respond()` (what `livekit.agents.inference.LLM` does
   internally): a call can outlast one token's TTL, and a turn that failed
   mid-negotiation because a token aged out would be indistinguishable to the
   consumer from the agent hanging up.
2. **No reasoning/thinking parameter is sent.** `reasoning_effort` maps to
   thinking budgets of 1024 tokens and up on this gateway. One spoken reply can
   stack five sequential round trips (`stream_turn`) or seven (`turn`), so the
   right budget on a phone call is none, not a small one.
3. **Zero data retention by default.** Prompts and outputs pass through the
   gateway without being logged, stored, or trained on. That matters here for
   the same reason `voice_app.py` leaves LiveKit Cloud recording off:
   nothing on this call collects consent for a third-party copy of what
   the consumer said.

The gateway is reached over plain HTTP rather than through the SDK's
`inference.LLM` deliberately: that class is async, and the turn loop is
synchronous — `voice_app.py` already drives it from a worker thread.
Going through the raw endpoint also keeps `text_app.py` — which has no LiveKit
job context and no event loop — able to use this route unchanged.

Model default is Gemini 3 Flash, which is `-preview` on the gateway. See the
note in the README: this route is opt-in and has not been certified
against `tests/evals/` or the adversarial pass.
"""

from __future__ import annotations

import datetime
import os
import time
from collections.abc import Iterator
from typing import Any, cast

from collector.llm.base import LLMResponse, Message, StreamEvent
from collector.llm.openai_shape import (
    load_env,
    to_llm_response,
    to_openai_messages,
    to_stream_events,
    tool_definitions,
)

BASE_URL = "https://agent-gateway.livekit.cloud/v1"
MODEL = "google/gemini-3-flash-preview"

# A spoken turn is one or two sentences. The cap is a backstop against a
# runaway generation, not a style control — matches the other two clients.
MAX_TOKENS = 1024

# Matches the SDK's own inference client. Long enough that no single request
# can age out mid-flight, short enough to be worth re-minting per turn.
TOKEN_TTL = datetime.timedelta(minutes=10)


def _access_token(api_key: str, api_secret: str) -> str:
    """A JWT good for inference only — no room, no recording, no SIP grant."""
    from livekit.api import AccessToken, InferenceGrants

    return (
        AccessToken(api_key, api_secret)
        .with_identity("collector-agent")
        .with_inference_grants(InferenceGrants(perform=True))
        .with_ttl(TOKEN_TTL)
        .to_jwt()
    )


class LiveKitInferenceClient:
    """`LLMClient` backed by LiveKit Inference. Constructed lazily so importing
    this module never requires credentials — only calling it does."""

    def __init__(
        self,
        *,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        from openai import OpenAI

        load_env()
        key = api_key or os.environ.get("LIVEKIT_API_KEY")
        secret = api_secret or os.environ.get("LIVEKIT_API_SECRET")
        if not key or not secret:
            raise RuntimeError(
                "LIVEKIT_API_KEY and LIVEKIT_API_SECRET are not set. Run the "
                "scripted client instead (uv run collector-text) if you don't "
                "have a LiveKit Cloud project."
            )
        self._api_key = key
        self._api_secret = secret
        self._client = OpenAI(base_url=BASE_URL, api_key=_access_token(key, secret))
        self._model = model
        self._max_tokens = max_tokens
        self._tools: list[Any] = tool_definitions()

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        # Re-minted per turn rather than per client: signing is local and
        # costs nothing, and a call outliving TOKEN_TTL is ordinary.
        self._client.api_key = _access_token(self._api_key, self._api_secret)
        conversation = to_openai_messages(messages)
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=cast(Any, conversation),
            tools=self._tools,
        )
        return to_llm_response(response.choices[0].message)

    def stream(self, messages: tuple[Message, ...]) -> Iterator[StreamEvent]:
        """The same turn, emitted as it is written.

        Without this, ``stream_response`` treats the route as non-streaming and
        hands ``stream_turn`` the finished paragraph as a single delta: the
        per-sentence guard still runs, but it runs on everything at once and the
        voice path waits for the whole turn before any audio. So the streaming
        transport buys this route nothing until the route can stream — which is
        the only reason a faster model here would be a latency win rather than
        just a different model.

        Same request as ``respond`` plus ``stream=True``, deliberately: a route
        whose streaming and non-streaming paths ask for different things is a
        route where the guardrail behaviour certified on one is not the
        behaviour running on the other.

        **Still uncertified.** Streaming does not change what the README says
        about this route — ``MAX_TOOL_ROUNDS`` and the strike budget were tuned
        against Claude, and ``tests/evals/`` plus the adversarial pass have to
        be re-run before it carries a real call.
        """
        self._client.api_key = _access_token(self._api_key, self._api_secret)
        conversation = to_openai_messages(messages)
        started = time.monotonic()
        chunks = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=cast(Any, conversation),
            tools=self._tools,
            stream=True,
        )
        yield from to_stream_events(chunks, model=self._model, started=started)
