"""Mapping tests for `llm/openai_shape.py` — no key or network required.

Mirrors `TestAnthropicMapping` in `test_agent_loop.py`: same transcript shapes,
asserted against OpenAI-style chat completions instead of the Anthropic
Messages API. The mapping is shared by every client that speaks that format —
OpenRouter and LiveKit Inference — so these pin both routes at once.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import openai
import pytest

from collector.llm.base import Message, ToolCall, system_prompt
from collector.llm.openai_shape import to_openai_messages, tool_definitions
from collector.llm.openrouter_client import MODEL, OpenRouterClient
from collector.tools import TOOL_NAMES

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class _StubCompletions:
    """Stands in for `client.chat.completions`, recording what was asked for."""

    def __init__(self, usage: Any = None, raises: BaseException | None = None) -> None:
        self.usage = usage
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        message = type("Msg", (), {"content": "Hello.", "refusal": None, "tool_calls": []})
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})
        body: dict[str, Any] = {"choices": [choice]}
        if self.usage is not None:
            body["usage"] = self.usage
        return type("Response", (), body)


class _StubClient:
    def __init__(self, completions: _StubCompletions) -> None:
        self.completions = completions
        self.chat = self


def _client(
    *, usage: Any = None, raises: BaseException | None = None
) -> tuple[OpenRouterClient, _StubClient]:
    client = OpenRouterClient(api_key="sk-test")
    stub = _StubClient(_StubCompletions(usage=usage, raises=raises))
    client._client = stub  # type: ignore[assignment]
    return client, stub


def _status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", _ENDPOINT)
    return openai.APIStatusError(
        f"status {status}", response=httpx.Response(status, request=request), body=None
    )


def _preamble() -> tuple[Message, ...]:
    return (system_prompt(consumer_name="Dana", account_ref="A-1"),)


class TestOpenRouterMapping:
    def test_the_opening_system_prompt_becomes_a_system_message(self) -> None:
        conversation = to_openai_messages(
            (
                system_prompt(consumer_name="Dana", account_ref="A-1"),
                Message(role="consumer", content="Hello?"),
            )
        )
        assert conversation[0]["role"] == "system"
        assert "collections representative" in conversation[0]["content"]
        assert conversation[1] == {"role": "user", "content": "Hello?"}

    def test_a_guardrail_note_rides_in_tagged_as_an_operator_note(self) -> None:
        conversation = to_openai_messages(
            (
                system_prompt(consumer_name="Dana", account_ref="A-1"),
                Message(role="agent", content="..."),
                Message(role="system", content="PROHIBITED_THREAT"),
            )
        )
        last = conversation[-1]
        assert last["role"] == "system"
        assert "<compliance_note>" in last["content"]

    def test_a_tool_result_is_preceded_by_the_call_that_asked_for_it(self) -> None:
        call = ToolCall(name="propose_offer", arguments={}, call_id="call_abc")
        conversation = to_openai_messages(
            (
                system_prompt(consumer_name="Dana", account_ref="A-1"),
                Message(role="consumer", content="What can you do?"),
                Message(role="tool", content='{"ok": true}', tool_call=call),
            )
        )
        request, result = conversation[-2], conversation[-1]
        assert request["role"] == "assistant"
        assert request["tool_calls"][0]["id"] == "call_abc"
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_abc"

    def test_a_missing_call_id_maps_the_same_way_every_run(self) -> None:
        message = Message(
            role="tool", content='{"ok": true}', tool_call=ToolCall(name="propose_offer")
        )
        preamble = (system_prompt(consumer_name="D", account_ref="A"),)
        first = to_openai_messages((*preamble, message))
        second = to_openai_messages((*preamble, message))
        assert first == second

    def test_tool_call_arguments_round_trip_through_json(self) -> None:
        call = ToolCall(
            name="validate_consumer_offer", arguments={"payment_count": 3}, call_id="c1"
        )
        message = Message(role="tool", content='{"ok": true}', tool_call=call)
        conversation = to_openai_messages((message,))
        encoded = conversation[0]["tool_calls"][0]["function"]["arguments"]
        assert json.loads(encoded) == {"payment_count": 3}

    def test_an_all_system_conversation_gets_a_call_started_nudge(self) -> None:
        """The opening `respond()` call (ring 1), and every regeneration
        retry off of it, has nothing but system-role turns. The chat
        completions format tolerates that (unlike Anthropic's Messages API,
        which rejects it outright), but a synthetic user turn is added anyway
        to mirror `_to_anthropic`'s `<call_started>` nudge — kicking a model
        trained on dialogue into producing one from an all-system context is
        not something to leave to chance on a compliance-critical opening
        line."""
        conversation = to_openai_messages((system_prompt(consumer_name="Dana", account_ref="A-1"),))
        assert conversation[-1]["role"] == "user"
        assert "<call_started>" in conversation[-1]["content"]

    def test_a_retry_still_all_system_also_gets_the_nudge(self) -> None:
        conversation = to_openai_messages(
            (
                system_prompt(consumer_name="Dana", account_ref="A-1"),
                Message(role="system", content="AI_DISCLOSURE_MISSING_AT_OPEN"),
            )
        )
        assert conversation[-1]["role"] == "user"
        assert "<call_started>" in conversation[-1]["content"]

    def test_a_real_consumer_turn_suppresses_the_nudge(self) -> None:
        conversation = to_openai_messages(
            (
                system_prompt(consumer_name="Dana", account_ref="A-1"),
                Message(role="consumer", content="Hello?"),
            )
        )
        assert all("<call_started>" not in c.get("content", "") for c in conversation)

    def test_the_tool_surface_the_model_sees_is_the_whitelist(self) -> None:
        assert {t["function"]["name"] for t in tool_definitions()} == set(TOOL_NAMES)

    def test_constructing_without_a_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A real .env may exist locally with a real key; `load_env` would
        # reload it into os.environ right after delenv clears it below.
        monkeypatch.setattr("collector.llm.openrouter_client.load_env", lambda: None)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            OpenRouterClient()


class TestOpenRouterUsage:
    """Usage capture is shared with the LiveKit route via `openai_shape`, so
    this route gets tokens and latency for free. Cost it does not get."""

    def test_tokens_and_latency_come_back_populated(self) -> None:
        usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": 2100,
                "completion_tokens": 48,
                "prompt_tokens_details": type("Details", (), {"cached_tokens": 1800}),
            },
        )
        client, _ = _client(usage=usage)
        result = client.respond(_preamble()).usage
        assert result is not None
        assert result.input_tokens == 2100
        assert result.output_tokens == 48
        assert result.cache_read_tokens == 1800
        assert result.stop_reason == "stop"
        assert result.latency_ms >= 0

    def test_a_details_block_without_the_livekit_extension_does_not_raise(self) -> None:
        """`cache_write_tokens` is a LiveKit addition to the OpenAI shape and is
        simply absent here. Reading it by attribute would take the call down."""
        usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": type("Details", (), {"cached_tokens": 0}),
            },
        )
        client, _ = _client(usage=usage)
        result = client.respond(_preamble()).usage
        assert result is not None
        assert result.cache_write_tokens == 0

    def test_cost_is_not_guessed_on_this_route(self) -> None:
        """OpenRouter bills its own margin over whichever provider it routes
        to. Reporting Anthropic's published rates for it would put a wrong
        number in a cost report, so no number is reported at all."""
        usage = type("Usage", (), {"prompt_tokens": 2100, "completion_tokens": 48})
        client, _ = _client(usage=usage)
        result = client.respond(_preamble()).usage
        assert result is not None
        assert result.model == MODEL
        assert result.cost_usd is None


class TestOpenRouterTransportFailures:
    @pytest.mark.parametrize("status", [429, 500, 503, 529])
    def test_a_transient_status_is_absorbed_into_the_response(self, status: int) -> None:
        client, _ = _client(raises=_status_error(status))
        response = client.respond(_preamble())
        assert response.error is not None
        assert response.text == ""
        assert response.usage is not None

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 413, 422])
    def test_a_fatal_status_propagates(self, status: int) -> None:
        client, _ = _client(raises=_status_error(status))
        with pytest.raises(openai.APIStatusError):
            client.respond(_preamble())

    def test_the_reasoning_extra_body_is_still_sent(self) -> None:
        """Task-scoped guard: absorbing errors must not have changed the
        request this route builds."""
        client, stub = _client()
        client.respond(_preamble())
        assert stub.completions.calls[0]["extra_body"] == {"reasoning": {"effort": "low"}}


class TestOpenRouterTimeoutConfiguration:
    def test_the_timeout_and_retry_budget_reach_the_openai_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def _spy(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("openai.OpenAI", _spy)
        OpenRouterClient(api_key="sk-test")
        assert captured["timeout"] == 6.0
        assert captured["max_retries"] == 1
