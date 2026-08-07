"""Tests for `llm/livekit_client.py` — no credentials or network required.

The transcript mapping is shared with the OpenRouter route and pinned in
`test_openrouter_client.py`. What is specific to this client, and tested here,
is the request it builds and the token it signs: an inference-only grant,
re-minted every turn, and no thinking budget on a path that can spend seven
round trips on one spoken sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx
import jwt
import openai
import pytest

from collector.llm.base import Message, system_prompt
from collector.llm.livekit_client import (
    MODEL,
    PRICES,
    LiveKitInferenceClient,
    _access_token,
    estimate_cost,
)
from collector.tools import TOOL_NAMES


@dataclass
class _StubFunction:
    name: str
    arguments: str


@dataclass
class _StubToolCall:
    id: str
    function: _StubFunction


@dataclass
class _StubMessage:
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[_StubToolCall] = field(default_factory=list)


@dataclass
class _StubPromptDetails:
    """The gateway's `prompt_tokens_details`. `cache_write_tokens` is a LiveKit
    extension to the stock OpenAI shape, so it is present here and may be
    absent elsewhere."""

    cached_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class _StubUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: _StubPromptDetails | None = None


class _StubCompletions:
    """Stands in for `client.chat.completions`, recording what was asked for."""

    def __init__(
        self,
        message: _StubMessage,
        usage: Any = None,
        finish_reason: str | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.message = message
        self.usage = usage
        self.finish_reason = finish_reason
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        choice = type("Choice", (), {"message": self.message, "finish_reason": self.finish_reason})
        # `usage` omitted entirely when None: a response object with no usage
        # attribute at all is the shape a leaner gateway returns, and it must
        # not take the call down with it.
        body: dict[str, Any] = {"choices": [choice]}
        if self.usage is not None:
            body["usage"] = self.usage
        return type("Response", (), body)


class _StubClient:
    def __init__(self, completions: _StubCompletions) -> None:
        self.api_key = "initial"
        self.completions = completions
        self.chat = self


def _client(
    message: _StubMessage | None = None,
    *,
    usage: Any = None,
    finish_reason: str | None = None,
    raises: BaseException | None = None,
) -> tuple[LiveKitInferenceClient, _StubClient]:
    """A client with real (fake-credentialed) token signing and a stub transport."""
    client = LiveKitInferenceClient(api_key="devkey", api_secret="devsecret" * 4)
    stub = _StubClient(
        _StubCompletions(
            message or _StubMessage(content="Hello."),
            usage=usage,
            finish_reason=finish_reason,
            raises=raises,
        )
    )
    client._client = stub  # type: ignore[assignment]
    return client, stub


_ENDPOINT = "https://agent-gateway.livekit.cloud/v1/chat/completions"


def _status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", _ENDPOINT)
    return openai.APIStatusError(
        f"status {status}", response=httpx.Response(status, request=request), body=None
    )


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(
        message="connection reset", request=httpx.Request("POST", _ENDPOINT)
    )


def _preamble() -> tuple[Message, ...]:
    return (system_prompt(consumer_name="Dana", account_ref="A-1"),)


class TestAccessToken:
    def test_the_token_grants_inference_and_nothing_else(self) -> None:
        claims = jwt.decode(
            _access_token("devkey", "devsecret" * 4), options={"verify_signature": False}
        )
        assert claims["inference"] == {"perform": True}
        # No room join, no recording, no SIP: a leaked inference token must not
        # be usable to dial out or to pull a call's audio.
        assert "video" not in claims
        assert "sip" not in claims

    def test_the_token_expires(self) -> None:
        claims = jwt.decode(
            _access_token("devkey", "devsecret" * 4), options={"verify_signature": False}
        )
        assert 0 < claims["exp"] - claims["nbf"] <= 15 * 60


class TestRequestShape:
    def test_it_asks_for_gemini_and_the_whole_tool_whitelist(self) -> None:
        client, stub = _client()
        client.respond((system_prompt(consumer_name="Dana", account_ref="A-1"),))
        request = stub.completions.calls[0]
        assert request["model"] == MODEL
        assert {t["function"]["name"] for t in request["tools"]} == set(TOOL_NAMES)

    def test_no_thinking_budget_is_requested(self) -> None:
        """`turn()` can stack seven sequential round trips into one spoken
        reply. A per-call thinking budget — 1024 tokens at the lowest
        `reasoning_effort` this gateway accepts — is latency the consumer
        hears as silence, so none is asked for."""
        client, stub = _client()
        client.respond((system_prompt(consumer_name="Dana", account_ref="A-1"),))
        request = stub.completions.calls[0]
        assert "reasoning_effort" not in request
        assert "extra_body" not in request

    def test_the_token_is_reminted_on_every_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A call outliving one token's TTL is ordinary; a turn that failed
        because the token aged out would be indistinguishable, to the consumer,
        from the agent hanging up.

        Counted rather than compared: two tokens minted in the same second are
        byte-identical, so a client that signed once at construction and handed
        the same string back forever would pass any equality check here.
        """
        mints: list[tuple[str, str]] = []

        def _counting_token(api_key: str, api_secret: str) -> str:
            mints.append((api_key, api_secret))
            return f"token-{len(mints)}"

        client, stub = _client()
        monkeypatch.setattr("collector.llm.livekit_client._access_token", _counting_token)

        preamble = (system_prompt(consumer_name="Dana", account_ref="A-1"),)
        client.respond(preamble)
        assert stub.api_key == "token-1"
        client.respond((*preamble, Message(role="consumer", content="Hello?")))
        assert stub.api_key == "token-2"

    def test_the_signed_token_is_a_real_inference_grant(self) -> None:
        client, stub = _client()
        client.respond((system_prompt(consumer_name="Dana", account_ref="A-1"),))
        assert jwt.decode(stub.api_key, options={"verify_signature": False})["inference"] == {
            "perform": True
        }


class TestResponseMapping:
    def test_a_tool_call_comes_back_as_a_tool_call(self) -> None:
        message = _StubMessage(
            content=None,
            tool_calls=[
                _StubToolCall(
                    id="call_1",
                    function=_StubFunction(
                        name="validate_consumer_offer", arguments='{"payment_count": 3}'
                    ),
                )
            ],
        )
        client, _ = _client(message)
        response = client.respond((system_prompt(consumer_name="Dana", account_ref="A-1"),))
        assert response.wants_tools
        call = response.tool_calls[0]
        assert call.name == "validate_consumer_offer"
        assert call.arguments == {"payment_count": 3}
        assert call.call_id == "call_1"

    def test_a_refusal_speaks_nothing(self) -> None:
        client, _ = _client(_StubMessage(content="ignored", refusal="I can't help with that"))
        response = client.respond((system_prompt(consumer_name="Dana", account_ref="A-1"),))
        assert response.text == ""
        assert not response.wants_tools


class TestUsage:
    """Per-call tokens and latency are the only way anyone can tell whether
    routing the voice path here helped or hurt. A route that reports `None`
    is a route nobody can measure."""

    def test_usage_comes_back_from_an_openai_shaped_response(self) -> None:
        client, _ = _client(
            usage=_StubUsage(prompt_tokens=2100, completion_tokens=48),
            finish_reason="stop",
        )
        usage = client.respond(_preamble()).usage
        assert usage is not None
        assert usage.model == MODEL
        assert usage.input_tokens == 2100
        assert usage.output_tokens == 48
        assert usage.stop_reason == "stop"
        assert usage.latency_ms >= 0

    def test_cached_prompt_tokens_map_to_cache_read(self) -> None:
        client, _ = _client(
            usage=_StubUsage(
                prompt_tokens=2100,
                completion_tokens=48,
                prompt_tokens_details=_StubPromptDetails(cached_tokens=1800),
            )
        )
        usage = client.respond(_preamble()).usage
        assert usage is not None
        assert usage.cache_read_tokens == 1800
        # `prompt_tokens` is the total and `cached_tokens` a subset of it, so
        # the reported input count stays the total — the subtraction belongs to
        # the cost calculation, not to the counter.
        assert usage.input_tokens == 2100

    def test_the_livekit_cache_write_extension_is_read(self) -> None:
        client, _ = _client(
            usage=_StubUsage(
                prompt_tokens=2100,
                completion_tokens=48,
                prompt_tokens_details=_StubPromptDetails(cache_write_tokens=900),
            )
        )
        usage = client.respond(_preamble()).usage
        assert usage is not None
        assert usage.cache_write_tokens == 900

    def test_a_response_with_no_usage_block_does_not_raise(self) -> None:
        """The shape has grown before. A missing counter should cost a number
        in a report, not the call it came from."""
        client, _ = _client(usage=None)
        response = client.respond(_preamble())
        assert response.text == "Hello."
        assert response.usage is not None
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0
        # Counters default to zero and the cost follows them to $0.00, the same
        # way `AnthropicClient._usage` handles a usage block it cannot read.
        # The zero is a reporting artefact of an absent block, not a claim that
        # the call was free — but it costs a number in a report rather than the
        # call it came from, which is the tradeoff being made deliberately.
        assert response.usage.cost_usd == Decimal("0.00")

    def test_partial_usage_fields_do_not_raise(self) -> None:
        # No `completion_tokens`, no `prompt_tokens_details` — the counters
        # that are absent default, the ones present are read.
        partial = type("PartialUsage", (), {"prompt_tokens": 1200})
        client, _ = _client(usage=partial)
        usage = client.respond(_preamble()).usage
        assert usage is not None
        assert usage.input_tokens == 1200
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0

    def test_a_refusal_still_reports_what_it_spent(self) -> None:
        """A refused turn spent tokens and spent latency. Reporting neither
        would make it invisible to the per-call logging."""
        client, _ = _client(
            _StubMessage(content="ignored", refusal="I can't help with that"),
            usage=_StubUsage(prompt_tokens=2100, completion_tokens=5),
            finish_reason="stop",
        )
        response = client.respond(_preamble())
        assert response.text == ""
        assert response.usage is not None
        assert response.usage.input_tokens == 2100
        assert response.usage.output_tokens == 5


class TestCost:
    def test_the_shipped_model_is_priced(self) -> None:
        """A model absent from the table reports no cost. That is the right
        default, but the model this route actually runs should not be hitting
        it — an unpriced default route is an unbudgeted one."""
        assert MODEL in PRICES

    def test_cost_is_computed_from_published_rates(self) -> None:
        # gemini-3.6-flash: $1.50/1M input, $7.50/1M output.
        cost = estimate_cost(MODEL, input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == Decimal("9.00")

    def test_an_unknown_model_reports_no_cost_rather_than_a_guess(self) -> None:
        assert estimate_cost("google/gemini-9-imaginary", input_tokens=10, output_tokens=10) is None

    def test_cached_tokens_are_discounted_not_double_charged(self) -> None:
        """`cached_tokens` is a subset of `prompt_tokens`. Billing the full
        prompt *and* the cached portion would overstate every cached call."""
        model = "google/gemini-3-flash-preview"  # $0.50 in, $3.00 out, $0.05 cached
        cost = estimate_cost(
            model,
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=1_000_000,
        )
        # All of the input was cached, so it all bills at the cached rate.
        assert cost == Decimal("0.05")
        uncached = estimate_cost(model, input_tokens=1_000_000, output_tokens=0)
        assert uncached == Decimal("0.50")
        assert cost is not None and uncached is not None and cost < uncached

    def test_a_cache_read_with_no_published_cached_rate_reports_no_cost(self) -> None:
        # LiveKit publishes no cached-input figure for gemini-3.6-flash, so a
        # call that actually read from cache has a component this table cannot
        # price honestly.
        assert estimate_cost(MODEL, input_tokens=100, output_tokens=10) is not None
        assert (
            estimate_cost(MODEL, input_tokens=100, output_tokens=10, cache_read_tokens=50) is None
        )

    def test_a_resolved_model_id_in_the_response_still_gets_priced(self) -> None:
        """Gateways echo back a resolved build (`…-002`) rather than the id
        that was asked for. Pricing off that string would miss `PRICES` on
        every call and report no cost — which is also the honest "no published
        rate" signal, so the fault would be invisible."""
        client, stub = _client(usage=_StubUsage(prompt_tokens=1000, completion_tokens=100))

        # The stub's response class carries a *different* model id than the
        # request, the way a resolving gateway does.
        original = stub.completions.create

        def _with_echoed_model(**kwargs: Any) -> Any:
            body = original(**kwargs)
            body.model = f"{MODEL}-002"
            return body

        stub.completions.create = _with_echoed_model  # type: ignore[method-assign]

        usage = client.respond(_preamble()).usage
        assert usage is not None
        # Reported as it came back, for the audit trail...
        assert usage.model == f"{MODEL}-002"
        # ...but priced against what was asked for.
        assert usage.cost_usd is not None
        assert usage.cost_usd == estimate_cost(MODEL, input_tokens=1000, output_tokens=100)

    def test_a_cache_write_reports_no_cost(self) -> None:
        # No cache-write rate is published for any model on this gateway.
        assert (
            estimate_cost(MODEL, input_tokens=100, output_tokens=10, cache_write_tokens=50) is None
        )


class TestTransportFailures:
    """This is a live phone call. A blip must cost a turn's words, not the call."""

    @pytest.mark.parametrize("status", [408, 409, 429, 500, 503, 529])
    def test_a_transient_status_is_absorbed_into_the_response(self, status: int) -> None:
        client, _ = _client(raises=_status_error(status))
        response = client.respond(_preamble())
        assert response.error is not None
        assert str(status) in response.error
        assert response.text == ""
        assert not response.wants_tools
        # Still enough usage to see the failed turn in the latency log.
        assert response.usage is not None
        assert response.usage.model == MODEL

    def test_a_connection_drop_is_absorbed(self) -> None:
        client, _ = _client(raises=_connection_error())
        response = client.respond(_preamble())
        assert response.error is not None
        assert "APIConnectionError" in response.error

    def test_a_timeout_is_absorbed(self) -> None:
        client, _ = _client(raises=openai.APITimeoutError(request=httpx.Request("POST", _ENDPOINT)))
        response = client.respond(_preamble())
        assert response.error is not None

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 413, 422])
    def test_a_fatal_status_propagates(self, status: int) -> None:
        """A bad key or a model name that does not exist will fail identically
        on every turn. Swallowing one would leave the agent silently mute while
        the call reported itself compliant."""
        client, _ = _client(raises=_status_error(status))
        with pytest.raises(openai.APIStatusError):
            client.respond(_preamble())


class TestTimeoutConfiguration:
    def test_the_timeout_and_retry_budget_reach_the_openai_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The SDK default is ten minutes. On a phone call that is the whole
        call spent in silence, so the constants have to actually be passed."""
        captured: dict[str, Any] = {}

        def _spy(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("openai.OpenAI", _spy)
        LiveKitInferenceClient(api_key="devkey", api_secret="devsecret" * 4)
        assert captured["timeout"] == 6.0
        assert captured["max_retries"] == 1

    def test_the_wall_clock_worst_case_is_twelve_seconds(self) -> None:
        """Pinned, not endorsed. timeout x (retries + 1) is 12s — the standing
        tail risk from FINDINGS-2026-08-07 Part 2, inherited unchanged from the
        Anthropic client so the routes do not diverge by accident. It bounds a
        stall; it does not deliver the ~1.5s tolerance that motivates it.
        Retuning is a decision for all three clients at once."""
        from collector.llm.openai_shape import MAX_RETRIES, TIMEOUT_SECONDS

        assert TIMEOUT_SECONDS * (MAX_RETRIES + 1) == 12.0


class TestCredentials:
    def test_constructing_without_credentials_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A real .env may exist locally with real credentials; `load_env` would
        # reload them into os.environ right after delenv clears them below.
        monkeypatch.setattr("collector.llm.livekit_client.load_env", lambda: None)
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="LIVEKIT_API_KEY"):
            LiveKitInferenceClient()

    def test_a_key_without_a_secret_is_not_enough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("collector.llm.livekit_client.load_env", lambda: None)
        monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="LIVEKIT_API_SECRET"):
            LiveKitInferenceClient()
