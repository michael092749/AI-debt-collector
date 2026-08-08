"""Mapping tests for `llm/openai_shape.py` — no key or network required.

Mirrors `TestAnthropicMapping` in `test_agent_loop.py`: same transcript shapes,
asserted against OpenAI-style chat completions instead of the Anthropic
Messages API. The mapping is shared by every client that speaks that format —
OpenRouter and LiveKit Inference — so these pin both routes at once.
"""

from __future__ import annotations

import json

import pytest

from collector.llm.base import Message, ToolCall, system_prompt
from collector.llm.openai_shape import to_openai_messages, tool_definitions
from collector.llm.openrouter_client import OpenRouterClient
from collector.tools import TOOL_NAMES


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
