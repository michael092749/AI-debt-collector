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

from collector.llm.base import (
    Message,
    StreamCompleted,
    StreamingLLMClient,
    TextDelta,
    ToolCall,
    system_prompt,
)
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


# --------------------------------------------------------------------------
# Streaming: the route has to be able to, or the streaming transport is moot
# --------------------------------------------------------------------------


@dataclass
class _StubFragment:
    """One tool call as it arrives on a stream: keyed by index, id and name in
    the first fragment, arguments a few characters at a time after it."""

    index: int
    id: str | None = None
    function: _StubFunction | None = None


@dataclass
class _StubDelta:
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[_StubFragment] | None = None


class _StubStreamCompletions:
    """Stands in for `client.chat.completions` on the streaming path."""

    def __init__(self, deltas: list[_StubDelta]) -> None:
        self.deltas = deltas
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return [
            type("Chunk", (), {"choices": [type("Choice", (), {"delta": delta})]})
            for delta in self.deltas
        ]


class _StubStreamClient:
    def __init__(self, deltas: list[_StubDelta]) -> None:
        self.api_key = "initial"
        self.completions = _StubStreamCompletions(deltas)
        self.chat = self


def _streaming_client(
    deltas: list[_StubDelta],
) -> tuple[LiveKitInferenceClient, _StubStreamClient]:
    client = LiveKitInferenceClient(api_key="devkey", api_secret="secret" * 4)
    stub = _StubStreamClient(deltas)
    client._client = stub  # type: ignore[assignment]
    return client, stub


class TestStreaming:
    """Why this exists at all: ``stream_response`` falls back to ``respond()``
    for a client without ``stream``, so an unstreamed route hands
    ``stream_turn`` the finished paragraph as one delta. The per-sentence guard
    still runs, but the voice path waits for the whole turn before any audio —
    the route gains nothing from the streaming transport, and a faster model on
    it is just a different model rather than a latency win.

    Note what these do *not* claim: nothing here certifies this route for a real
    call. ``MAX_TOOL_ROUNDS`` and the strike budget were tuned against Claude.
    """

    def test_it_conforms_to_the_streaming_protocol(self) -> None:
        """``stream_response`` dispatches on a runtime protocol check, so
        satisfying it is what actually routes a turn through ``stream()``
        instead of silently degrading to one delta."""
        client, _ = _streaming_client([_StubDelta(content="Hello.")])

        assert isinstance(client, StreamingLLMClient)

    def test_text_arrives_delta_by_delta(self) -> None:
        """The whole point: three fragments out, three fragments through, so the
        sentence guard sees a sentence the moment it completes rather than after
        the turn."""
        client, _ = _streaming_client(
            [
                _StubDelta(content="That works. "),
                _StubDelta(content="Let me confirm "),
                _StubDelta(content="the details."),
            ]
        )

        events = list(client.stream((Message(role="consumer", content="ok"),)))

        assert [e.text for e in events if isinstance(e, TextDelta)] == [
            "That works. ",
            "Let me confirm ",
            "the details.",
        ]
        completed = events[-1]
        assert isinstance(completed, StreamCompleted)
        assert completed.response.text == "That works. Let me confirm the details."

    def test_a_tool_call_is_reassembled_from_its_fragments(self) -> None:
        """The shape that has to be rebuilt rather than read. Arguments arrive a
        few characters at a time; a client that read only the first fragment
        would call the engine with an empty proposal."""
        client, _ = _streaming_client(
            [
                _StubDelta(
                    tool_calls=[
                        _StubFragment(
                            index=0,
                            id="call_1",
                            function=_StubFunction(name="validate_consumer_offer", arguments=""),
                        )
                    ]
                ),
                _StubDelta(
                    tool_calls=[
                        _StubFragment(index=0, function=_StubFunction(name="", arguments='{"pay'))
                    ]
                ),
                _StubDelta(
                    tool_calls=[
                        _StubFragment(
                            index=0, function=_StubFunction(name="", arguments='ment_count": 2}')
                        )
                    ]
                ),
            ]
        )

        events = list(client.stream((Message(role="consumer", content="two payments?"),)))

        completed = events[-1]
        assert isinstance(completed, StreamCompleted)
        assert completed.response.tool_calls == (
            ToolCall(
                name="validate_consumer_offer", arguments={"payment_count": 2}, call_id="call_1"
            ),
        )

    def test_parallel_tool_calls_keep_their_own_arguments(self) -> None:
        """Interleaved fragments are keyed by ``index``. Merging two calls'
        arguments would hand the engine a proposal the model never made."""
        client, _ = _streaming_client(
            [
                _StubDelta(
                    tool_calls=[
                        _StubFragment(
                            index=0,
                            id="a",
                            function=_StubFunction(name="record_refusal", arguments="{}"),
                        ),
                        _StubFragment(
                            index=1,
                            id="b",
                            function=_StubFunction(name="propose_offer", arguments="{}"),
                        ),
                    ]
                ),
            ]
        )

        events = list(client.stream((Message(role="consumer", content="no"),)))

        completed = events[-1]
        assert isinstance(completed, StreamCompleted)
        assert [c.name for c in completed.response.tool_calls] == [
            "record_refusal",
            "propose_offer",
        ]

    def test_a_refusal_streams_nothing(self) -> None:
        """Same rule as the non-streaming path: say nothing rather than
        something unvetted."""
        client, _ = _streaming_client([_StubDelta(refusal="I can't help with that.")])

        events = list(client.stream((Message(role="consumer", content="..."),)))

        assert not [e for e in events if isinstance(e, TextDelta)]
        completed = events[-1]
        assert isinstance(completed, StreamCompleted)
        assert completed.response.text == ""
        assert completed.response.tool_calls == ()

    def test_the_round_still_leaves_a_usage_record(self) -> None:
        """``_record_model_call`` writes nothing for a response with no usage,
        so a streamed round without one would be a model call missing from the
        audit trail. Token counts need ``stream_options``, which this route does
        not send — so latency and the model name are measured locally and the
        row exists."""
        client, _ = _streaming_client([_StubDelta(content="Hello.")])

        events = list(client.stream((Message(role="consumer", content="hi"),)))

        completed = events[-1]
        assert isinstance(completed, StreamCompleted)
        usage = completed.response.usage
        assert usage is not None
        assert usage.model == MODEL
        assert usage.latency_ms >= 0

    def test_the_streamed_request_matches_the_blocking_one(self) -> None:
        """A route whose two paths ask for different things is a route where the
        behaviour certified on one is not the behaviour running on the other."""
        client, stub = _streaming_client([_StubDelta(content="Hello.")])
        messages = (system_prompt(consumer_name="Dana", account_ref="A-1"),)

        list(client.stream(messages))
        streamed = dict(stub.completions.calls[0])

        blocking, blocking_stub = _client()
        blocking.respond(messages)
        asked = dict(blocking_stub.completions.calls[0])

        assert streamed.pop("stream") is True
        assert streamed == asked

    def test_the_token_is_reminted_for_a_streamed_turn_too(self) -> None:
        """The reason ``respond`` re-mints: a call outlives one token's TTL.
        Streaming does not change that, and a stream that started on an expired
        token fails mid-negotiation."""
        client, stub = _streaming_client([_StubDelta(content="Hello.")])

        list(client.stream((Message(role="consumer", content="hi"),)))

        assert stub.api_key != "initial"
