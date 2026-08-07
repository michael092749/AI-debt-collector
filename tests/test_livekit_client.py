"""Tests for `llm/livekit_client.py` — no credentials or network required.

The transcript mapping is shared with the OpenRouter route and pinned in
`test_openrouter_client.py`. What is specific to this client, and tested here,
is the request it builds and the token it signs: an inference-only grant,
re-minted every turn, and no thinking budget on a path that can spend seven
round trips on one spoken sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jwt
import pytest

from collector.llm.base import Message, system_prompt
from collector.llm.livekit_client import MODEL, LiveKitInferenceClient, _access_token
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


class _StubCompletions:
    """Stands in for `client.chat.completions`, recording what was asked for."""

    def __init__(self, message: _StubMessage) -> None:
        self.message = message
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        choice = type("Choice", (), {"message": self.message})
        return type("Response", (), {"choices": [choice]})


class _StubClient:
    def __init__(self, message: _StubMessage) -> None:
        self.api_key = "initial"
        self.completions = _StubCompletions(message)
        self.chat = self


def _client(message: _StubMessage | None = None) -> tuple[LiveKitInferenceClient, _StubClient]:
    """A client with real (fake-credentialed) token signing and a stub transport."""
    client = LiveKitInferenceClient(api_key="devkey", api_secret="devsecret" * 4)
    stub = _StubClient(message or _StubMessage(content="Hello."))
    client._client = stub  # type: ignore[assignment]
    return client, stub


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
