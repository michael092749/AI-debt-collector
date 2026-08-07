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
   thinking budgets of 1024 tokens and up on this gateway. `NegotiationAgent.turn()`
   can stack up to seven sequential round trips into one spoken reply, so the
   right budget on a phone call is none, not a small one.
3. **Zero data retention by default.** Prompts and outputs pass through the
   gateway without being logged, stored, or trained on. That matters here for
   the same reason `voice_app.py` leaves LiveKit Cloud recording off:
   nothing on this call collects consent for a third-party copy of what
   the consumer said.

The gateway is reached over plain HTTP rather than through the SDK's
`inference.LLM` deliberately: that class is async, and `NegotiationAgent.turn()`
is synchronous and already dispatched to a worker thread by `voice_app.py`.
Going through the raw endpoint also keeps `text_app.py` — which has no LiveKit
job context and no event loop — able to use this route unchanged.

The model is Gemini 3.6 Flash — the newest full Flash on the gateway, and GA
rather than `-preview`. ("Gemini 3.1 Flash", asked for originally, does not
exist there: 3.1 ships only as `flash-lite` and `pro-preview`.)

⚠️ **This route is now the default, and it is still uncertified.** It has not
been run against `tests/evals/` or the adversarial pass. `MAX_TOOL_ROUNDS` and
the strike budget were tuned against Claude, and the outbound guard already
trips on roughly three turns in eight *with* the model they were tuned for.
That was an acceptable footnote while this was opt-in; as the default it is a
shipped risk carried by every live call, and the per-call usage recorded below
is what makes it observable rather than merely hoped about.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import replace
from decimal import Decimal
from typing import Any

from collector.llm.base import LLMResponse, Message
from collector.llm.openai_shape import (
    MAX_RETRIES,
    TIMEOUT_SECONDS,
    chat_completion,
    load_env,
    to_openai_messages,
    tool_definitions,
)

BASE_URL = "https://agent-gateway.livekit.cloud/v1"
MODEL = "google/gemini-3.6-flash"

# A spoken turn is one or two sentences. The cap is a backstop against a
# runaway generation, not a style control — matches the other two clients.
MAX_TOKENS = 1024

# LiveKit Inference's published rates, USD per million tokens, as of 2026-08,
# keyed by the gateway's own namespaced model id. Three figures per model:
# input, output, cached input. A model absent from this table — or one whose
# cached rate LiveKit does not publish, on a call that actually read from cache
# — is reported with no cost rather than a guessed one.
#
# Note the cached rate is an absolute price here, not the multiple of the input
# rate that `anthropic_client` derives. Gemini 3.6 Flash and 3.5 Flash Lite
# publish no cached figure at all; `None` says so rather than inventing one.
PRICES: dict[str, tuple[Decimal, Decimal, Decimal | None]] = {
    "google/gemini-2.5-flash": (Decimal("0.30"), Decimal("2.50"), Decimal("0.03")),
    "google/gemini-2.5-flash-lite": (Decimal("0.10"), Decimal("0.40"), Decimal("0.01")),
    "google/gemini-2.5-pro": (Decimal("2.50"), Decimal("15.00"), Decimal("0.25")),
    "google/gemini-3-flash-preview": (Decimal("0.50"), Decimal("3.00"), Decimal("0.05")),
    "google/gemini-3.1-flash-lite": (Decimal("0.25"), Decimal("1.50"), Decimal("0.025")),
    "google/gemini-3.1-pro-preview": (Decimal("4.00"), Decimal("18.00"), Decimal("0.40")),
    "google/gemini-3.5-flash": (Decimal("1.50"), Decimal("9.00"), Decimal("0.15")),
    "google/gemini-3.5-flash-lite": (Decimal("0.30"), Decimal("2.50"), None),
    "google/gemini-3.6-flash": (Decimal("1.50"), Decimal("7.50"), None),
}

_PER_MILLION = Decimal(1_000_000)

# Matches the SDK's own inference client. Long enough that no single request
# can age out mid-flight, short enough to be worth re-minting per turn.
TOKEN_TTL = datetime.timedelta(minutes=10)


def estimate_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal | None:
    """USD for one call, or ``None`` when this table cannot price it honestly.

    Decimal throughout, like every other figure in this system: a float here
    would be a rounding error in a cost report rather than in a payment
    schedule, but the rule is the rule.

    Cached prompt tokens are a *subset* of ``prompt_tokens`` in the OpenAI
    shape, not an addition to it as they are on Anthropic's — so they are
    subtracted out and rebilled at the cached rate, never added on top. The
    clamp is defensive: a provider quirk must not be able to bill a negative.
    """
    rates = PRICES.get(model)
    if rates is None:
        return None
    input_rate, output_rate, cached_rate = rates
    if cache_write_tokens:
        # LiveKit publishes no cache-*write* rate for any model, so a call that
        # wrote to cache has a component this table cannot price.
        return None
    if cache_read_tokens and cached_rate is None:
        return None
    billable_input = max(input_tokens - cache_read_tokens, 0)
    billed = (
        input_rate * billable_input
        + output_rate * output_tokens
        + (cached_rate or 0) * cache_read_tokens
    )
    return billed / _PER_MILLION


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
        timeout: float = TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
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
        # Timeout and retries on the client, not hand-rolled around the call:
        # the SDK already backs off on the retryable statuses (408/409/429/5xx)
        # and reads ``retry-after``, and duplicating that is how the two drift.
        self._client = OpenAI(
            base_url=BASE_URL,
            api_key=_access_token(key, secret),
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._tools: list[Any] = tool_definitions()

    def respond(self, messages: tuple[Message, ...]) -> LLMResponse:
        # Re-minted per turn rather than per client: signing is local and
        # costs nothing, and a call outliving TOKEN_TTL is ordinary.
        self._client.api_key = _access_token(self._api_key, self._api_secret)
        response = chat_completion(
            self._client,
            model=self._model,
            messages=to_openai_messages(messages),
            tools=self._tools,
            max_tokens=self._max_tokens,
        )
        return _priced(response, self._model)


def _priced(response: LLMResponse, requested_model: str) -> LLMResponse:
    """Fill in ``cost_usd`` from the gateway's own rate card.

    Applied here rather than in ``openai_shape`` because the rates are LiveKit
    Inference's: the same model id billed through another gateway is another
    number, and a shared price table would quietly report one route's prices
    for the other's traffic.

    Priced against the model that was *asked for*, not the one the response
    echoes back. Gateways resolve a request to a dated or provider-qualified
    build (``…-002``, and the like), and that string is not a key in ``PRICES``
    — pricing off it would miss the table on every call and report ``None``.
    ``None`` is also the honest "no published rate" signal, so the two would be
    indistinguishable and the fault invisible. ``usage.model`` still carries
    whatever came back, which is what the audit trail wants.
    """
    usage = response.usage
    if usage is None:
        return response
    return replace(
        response,
        usage=replace(
            usage,
            cost_usd=estimate_cost(
                requested_model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            ),
        ),
    )
