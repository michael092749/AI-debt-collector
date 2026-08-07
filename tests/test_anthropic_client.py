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

import json
from typing import Any

import httpx
import pytest

from collector.llm.anthropic_client import AnthropicClient, _cached_transcript, _to_anthropic
from collector.llm.base import LLMResponse, Message, StreamCompleted, system_prompt

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


class TestPromptCaching:
    """What a turn's second, third and fourth model calls stop paying for.

    ``stream_turn`` can make five sequential calls for one spoken reply, each
    re-sending the whole growing transcript plus the system prompt plus six tool
    schemas. Two ``cache_control`` breakpoints turn that into one full-price
    prefix per call instead of five.

    **These tests certify shape, not savings.** A cache hit is only observable
    as a non-zero ``cache_read_input_tokens`` from the live API, and a prefix
    under the model's 1024-token minimum caches nothing at all while reporting
    no error. So what is pinned here is everything that would silently cost
    every hit if it drifted: where the breakpoints sit, that there are few
    enough of them, and that the cached prefix is byte-identical between calls.
    The live signal is ``cache_read_tokens`` on the ``ModelCalled`` row, which
    ``_usage`` already reads and ``estimate_cost`` already prices.
    """

    def _sent(self, messages: tuple[Message, ...]) -> dict[str, Any]:
        """The kwargs the SDK would have received."""
        seen: dict[str, Any] = {}
        client = _client()

        class _Captures:
            def create(self, **kwargs: Any) -> Any:
                seen.update(kwargs)
                return _Message(content=[], stop_reason="end_turn", usage=_Usage(), model="m")

        client._client.messages = _Captures()  # type: ignore[assignment]
        client.respond(messages)
        return seen

    @staticmethod
    def _breakpoints(request: dict[str, Any]) -> int:
        blocks = list(request["system"])
        for message in request["messages"]:
            content = message["content"]
            if not isinstance(content, str):
                blocks.extend(content)
        return sum("cache_control" in block for block in blocks)

    def _conversation(self) -> tuple[Message, ...]:
        return (
            system_prompt(consumer_name="Dana", account_ref="A-1"),
            Message(role="consumer", content="Yes, speaking."),
            Message(role="agent", content="Thanks for confirming."),
            Message(role="consumer", content="I can do fifty a month."),
        )

    def test_the_system_prompt_carries_a_breakpoint_and_so_covers_the_tools(self) -> None:
        """Render order is tools, then system, then messages, so one breakpoint
        on the last system block caches the six tool schemas with it. That is
        why there is no separate breakpoint on the tools."""
        request = self._sent(self._conversation())

        assert isinstance(request["system"], list), "a bare string has nowhere to put cache_control"
        assert request["system"][-1]["cache_control"] == {"type": "ephemeral"}
        assert "collections representative" in request["system"][-1]["text"]

    def test_the_end_of_the_transcript_carries_a_breakpoint(self) -> None:
        """The one that compounds inside a turn: round two reads the prefix
        round one wrote."""
        request = self._sent(self._conversation())

        last = request["messages"][-1]["content"]
        assert not isinstance(last, str), "a string content block cannot be marked"
        assert last[-1]["cache_control"] == {"type": "ephemeral"}

    def test_the_breakpoints_stay_under_the_ceiling(self) -> None:
        """Four per request, and the API rejects a fifth outright — so a third
        breakpoint added later has to be a decision, not an accident."""
        assert self._breakpoints(self._sent(self._conversation())) <= 4

    def test_the_opening_call_is_marked_too(self) -> None:
        """``open_call`` sends only the system prompt. It is the request that
        *writes* the entry the first turn reads, so leaving it unmarked would
        make every call pay full price for its first two model calls."""
        request = self._sent((system_prompt(consumer_name="Dana", account_ref="A-1"),))

        assert request["system"][-1]["cache_control"] == {"type": "ephemeral"}
        assert self._breakpoints(request) >= 1

    def test_the_cached_prefix_is_byte_identical_between_calls(self) -> None:
        """The silent-invalidator guard. Caching is a prefix match, so anything
        non-deterministic in the tool schemas or the system prompt — an unsorted
        ``json.dumps``, a timestamp, a set iteration — costs every hit on every
        call and reports nothing. Compared as serialized bytes, because that is
        what the cache key is derived from."""
        first = self._sent(self._conversation())
        second = self._sent(self._conversation())

        assert json.dumps(first["tools"]) == json.dumps(second["tools"])
        assert json.dumps(first["system"]) == json.dumps(second["system"])

    def test_marking_the_request_does_not_reach_back_into_the_mapping(self) -> None:
        """``_to_anthropic`` is a pure mapping the loop's own tests read
        directly. The caching layer copies the entry it marks, so the mapping
        keeps returning the shape it documents."""
        _, conversation = _to_anthropic(self._conversation())
        before = json.dumps(conversation)

        _cached_transcript(conversation)

        assert json.dumps(conversation) == before

    def test_streaming_sends_the_same_cached_prefix(self) -> None:
        """The voice path streams, so an unmarked ``stream()`` would mean the
        caching only ever helped the terminal."""
        seen: dict[str, Any] = {}
        client = _client()

        class _Captures:
            def stream(self, **kwargs: Any) -> Any:
                seen.update(kwargs)
                raise RuntimeError("captured")

        client._client.messages = _Captures()  # type: ignore[assignment]
        with pytest.raises(RuntimeError):
            list(client.stream(self._conversation()))

        assert seen["system"][-1]["cache_control"] == {"type": "ephemeral"}
        assert seen["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
