"""X9c: cached-token usage (I1) read from the real OpenAI and Anthropic SDK types.

The fakes elsewhere build usage from SimpleNamespace; these pin the field names the
providers read to the pinned SDKs, which CI installs from requirements-ci.txt.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.llm_router_part2_inference import AnthropicProvider, OpenAIProvider


@pytest.mark.asyncio
async def test_openai_sdk_usage_reports_cached_and_reasoning_tokens(
    sample_query_request,
):
    pytest.importorskip("openai", minversion="1.66")
    from openai.types.responses import Response

    api_response = Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "gpt-5",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "done", "annotations": []}
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {
                "input_tokens": 1200,
                "input_tokens_details": {"cached_tokens": 1024},
                "output_tokens": 50,
                "output_tokens_details": {"reasoning_tokens": 20},
                "total_tokens": 1250,
            },
        }
    )

    with patch("src.llm_router_part2_inference.openai.AsyncOpenAI") as mock_openai:
        client = AsyncMock()
        client.responses.create.return_value = api_response
        mock_openai.return_value = client
        provider = OpenAIProvider({"api_key": "test-key"})
        await provider.initialize()
        response = await provider.generate_response(sample_query_request, "gpt-5")

    assert response.token_count_input == 1200
    assert response.token_count_output == 50
    assert response.total_tokens == 1250
    assert response.provider_cached_input_tokens == 1024
    assert response.reasoning_tokens == 20


@pytest.mark.asyncio
async def test_anthropic_sdk_usage_reports_cache_read_and_creation_tokens(
    sample_query_request,
):
    pytest.importorskip("anthropic", minversion="0.69")
    from anthropic.types import Message

    api_response = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 300,
                "cache_read_input_tokens": 2000,
                "output_tokens": 40,
            },
        }
    )

    with patch(
        "src.llm_router_part2_inference.anthropic.AsyncAnthropic"
    ) as mock_anthropic:
        client = AsyncMock()
        client.messages.create.return_value = api_response
        mock_anthropic.return_value = client
        provider = AnthropicProvider({"api_key": "test-key"})
        await provider.initialize()
        response = await provider.generate_response(
            sample_query_request, "claude-sonnet-4-6"
        )

    # Anthropic's input_tokens excludes the cached prefix; the router adds it back.
    assert response.token_count_input == 2310
    assert response.total_tokens == 2350
    assert response.provider_cached_input_tokens == 2000
    assert response.provider_cache_read_input_tokens == 2000
    assert response.provider_cache_creation_input_tokens == 300
