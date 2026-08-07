"""The real client's transport behaviour, without a key or a network.

``AnthropicClient`` constructs offline against a fake key and never dials until
a method is called, so its error partition, usage mapping and response assembly
are all testable here. What is *not* testable here — and is stated rather than
implied — is whether the live API accepts ``thinking={"type": "adaptive"}``
alongside ``output_config={"effort": ...}``, and whether mid-conversation
``system`` messages really are unsupported (the premise ``_to_anthropic`` is
built on). Those need a key.

The partition is the point. An earlier version listed exception *classes*
— ``(APIConnectionError, RateLimitError, InternalServerError)`` — and
``OverloadedError`` (529) turned out to be a sibling of ``InternalServerError``
rather than a subclass, so Anthropic's standard overload signal fell straight
through and killed the call. These tests pin the statuses, not the hierarchy.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from collector.llm.anthropic_client import AnthropicClient
from collector.llm.base import LLMResponse, Message, StreamCompleted

FAKE_KEY = "sk-ant-not-a-real-key"


def _client(**kwargs: Any) -> AnthropicClient:
    """Constructs fully offline: the SDK does not dial until a call is made."""
    return AnthropicClient(api_key=FAKE_KEY, **kwargs)


def _status_error(status: int) -> Exception:
    """The exception the SDK itself would raise for this status.

    Built through the SDK's own factory rather than by naming a class, so the
    test tracks whatever mapping the installed version actually has — which is
    the whole point, since naming classes is what went wrong the first time.
    """
    import anthropic

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request, json={"error": {"message": "nope"}})
    factory = anthropic.Anthropic(api_key=FAKE_KEY)
    return factory._make_status_error_from_response(response)  # noqa: SLF001


class _Raises:
    """Stands in for ``client.messages``, failing the way the SDK would."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def create(self, **kwargs: Any) -> Any:
        raise self._exc

    def stream(self, **kwargs: Any) -> Any:
        raise self._exc


class TestErrorPartition:
    @pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 529])
    def test_a_transient_status_is_spoken_through_not_raised(self, status: int) -> None:
        """529 is the one that matters most: it is Anthropic's overload signal
        and the most likely transient failure in production."""
        client = _client()
        client._client.messages = _Raises(_status_error(status))  # type: ignore[assignment]

        response = client.respond((Message(role="consumer", content="hello?"),))

        assert response.error is not None, f"status {status} should be absorbed"
        assert response.text == ""
        assert response.usage is not None and response.usage.latency_ms >= 0

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_a_misconfiguration_status_propagates_loudly(self, status: int) -> None:
        """A bad key or a model name that does not exist must not become an
        empty turn — that is a mute agent reporting itself compliant."""
        client = _client()
        client._client.messages = _Raises(_status_error(status))  # type: ignore[assignment]

        import anthropic

        with pytest.raises(anthropic.APIStatusError) as caught:
            client.respond((Message(role="consumer", content="hello?"),))
        assert caught.value.status_code == status

    def test_a_connection_failure_is_transient(self) -> None:
        import anthropic

        client = _client()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        client._client.messages = _Raises(  # type: ignore[assignment]
            anthropic.APITimeoutError(request=request)
        )

        response = client.respond(())
        assert response.error is not None
        assert "APITimeoutError" in response.error

    def test_the_streaming_path_partitions_the_same_way(self) -> None:
        """The except block only fires on iteration, so the stream must be
        drained for the test to mean anything."""
        transient = _client()
        transient._client.messages = _Raises(_status_error(529))  # type: ignore[assignment]
        (event,) = list(transient.stream(()))
        assert isinstance(event, StreamCompleted)
        assert event.response.error is not None

        import anthropic

        fatal = _client()
        fatal._client.messages = _Raises(_status_error(401))  # type: ignore[assignment]
        with pytest.raises(anthropic.AuthenticationError):
            list(fatal.stream(()))

    def test_the_error_detail_carries_no_credential(self) -> None:
        """It lands in the audit log and in a log line, so it must not carry
        the key or the request."""
        client = _client()
        client._client.messages = _Raises(_status_error(500))  # type: ignore[assignment]

        response = client.respond(())
        assert response.error is not None
        assert FAKE_KEY not in response.error
        assert "x-api-key" not in response.error.lower()


class _Block:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class _Usage:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class _Message:
    def __init__(self, content: list[Any], stop_reason: str, usage: Any, model: str) -> None:
        self.content, self.stop_reason, self.usage, self.model = content, stop_reason, usage, model


class _FakeStream:
    """The context manager ``client.messages.stream`` returns, drained empty."""

    def __init__(self, message: _Message) -> None:
        self._message = message
        self.text_stream: Any = iter(())

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> _Message:
        return self._message


class _Captures:
    """Stands in for ``client.messages``, recording the request kwargs."""

    def __init__(self, message: _Message) -> None:
        self._message = message
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self._message

    def stream(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return _FakeStream(self._message)


def _empty_reply() -> _Message:
    return _Message(
        content=[_Block(type="text", text="Hello.")],
        stop_reason="end_turn",
        usage=_Usage(input_tokens=1, output_tokens=1),
        model="claude-sonnet-5",
    )


class TestPromptCaching:
    """The static prefix must carry a cache breakpoint on every call.

    The API renders tools before system, so the single marker on the system
    block caches the tool schemas and the system prompt together — the
    ~2k-token prefix otherwise reprocessed uncached 2-3 times per turn.
    """

    CACHED_SYSTEM = [
        {
            "type": "text",
            "text": "You are a collections representative.",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    def test_respond_marks_the_system_block_for_caching(self) -> None:
        client = _client()
        captures = _Captures(_empty_reply())
        client._client.messages = captures  # type: ignore[assignment]

        client.respond(
            (
                Message(role="system", content="You are a collections representative."),
                Message(role="consumer", content="hello?"),
            )
        )
        assert captures.kwargs["system"] == self.CACHED_SYSTEM

    def test_stream_marks_the_system_block_for_caching(self) -> None:
        client = _client()
        captures = _Captures(_empty_reply())
        client._client.messages = captures  # type: ignore[assignment]

        list(
            client.stream(
                (
                    Message(role="system", content="You are a collections representative."),
                    Message(role="consumer", content="hello?"),
                )
            )
        )
        assert captures.kwargs["system"] == self.CACHED_SYSTEM

    def test_an_empty_system_prompt_passes_through_unmarked(self) -> None:
        """No prefix, nothing to cache — a breakpoint on an empty block would
        be a pointless cache write."""
        client = _client()
        captures = _Captures(_empty_reply())
        client._client.messages = captures  # type: ignore[assignment]

        client.respond((Message(role="consumer", content="hello?"),))
        assert captures.kwargs["system"] == ""


class TestResponseAssembly:
    def _replied(self, message: _Message) -> LLMResponse:
        client = _client()

        class _Returns:
            def create(self, **kwargs: Any) -> Any:
                return message

        client._client.messages = _Returns()  # type: ignore[assignment]
        return client.respond(())

    def test_text_and_tool_calls_are_both_carried(self) -> None:
        response = self._replied(
            _Message(
                content=[
                    _Block(type="text", text="Let me check that."),
                    _Block(
                        type="tool_use",
                        name="propose_offer",
                        input={"preferred_cadence": "weekly"},
                        id="toolu_1",
                    ),
                ],
                stop_reason="tool_use",
                usage=_Usage(input_tokens=900, output_tokens=42),
                model="claude-sonnet-5",
            )
        )
        assert response.text == "Let me check that."
        assert response.tool_calls[0].name == "propose_offer"
        assert response.tool_calls[0].arguments == {"preferred_cadence": "weekly"}
        assert response.tool_calls[0].call_id == "toolu_1"

    def test_a_refusal_yields_an_empty_turn_but_still_reports_usage(self) -> None:
        response = self._replied(
            _Message(
                content=[_Block(type="text", text="I won't help with that.")],
                stop_reason="refusal",
                usage=_Usage(input_tokens=10, output_tokens=0),
                model="claude-sonnet-5",
            )
        )
        assert response.text == ""
        assert response.tool_calls == ()
        assert response.usage is not None and response.usage.stop_reason == "refusal"

    def test_usage_maps_every_counter_and_prices_it(self) -> None:
        response = self._replied(
            _Message(
                content=[_Block(type="text", text="Hello.")],
                stop_reason="end_turn",
                usage=_Usage(
                    input_tokens=1_000_000,
                    output_tokens=0,
                    cache_read_input_tokens=7,
                    cache_creation_input_tokens=11,
                ),
                model="claude-sonnet-5",
            )
        )
        usage = response.usage
        assert usage is not None
        assert (usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens) == (
            1_000_000,
            7,
            11,
        )
        assert usage.cost_usd is not None and usage.cost_usd > 3

    def test_a_missing_usage_counter_costs_a_number_not_the_call(self) -> None:
        """The usage shape has grown before. A counter that disappears should
        cost a figure in a report, not the turn it came from."""
        response = self._replied(
            _Message(
                content=[_Block(type="text", text="Hello.")],
                stop_reason="end_turn",
                usage=_Usage(input_tokens=5),
                model="claude-sonnet-5",
            )
        )
        assert response.text == "Hello."
        assert response.usage is not None and response.usage.output_tokens == 0
