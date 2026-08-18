from contextlib import nullcontext
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic import (
    adapter as anthropic_module,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter import (
    AnthropicAdapter,
)


class _Response(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_structured_output_caches_system_prompt(monkeypatch):
    monkeypatch.setattr(anthropic_module, "llm_rate_limiter_context_manager", nullcontext)
    adapter = object.__new__(AnthropicAdapter)
    adapter.model = "claude-sonnet"
    adapter.name = "Anthropic"
    adapter.llm_args = {"max_tokens": 128}
    adapter.aclient = AsyncMock(return_value=_Response(value="ok"))

    result = await adapter.acreate_structured_output("variable input", "stable prompt", _Response)

    request = adapter.aclient.await_args.kwargs
    assert request["system"] == [
        {
            "type": "text",
            "text": "stable prompt",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert request["messages"][0]["content"].endswith("variable input")
    assert "stable prompt" not in request["messages"][0]["content"]
    assert result == _Response(value="ok")
