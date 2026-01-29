"""Tests for context overflow recovery in the agent loop."""

from agent.base import BaseAgent
from config import Config
from llm.base import LLMMessage, LLMResponse, StopReason


class OverflowMockLLM:
    """Mock LLM that raises a context length error on the first call."""

    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.call_count = 0

    async def call_async(self, messages, tools=None, max_tokens=4096, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            raise Exception("context_length_exceeded")
        return LLMResponse(content="ok", stop_reason=StopReason.STOP)

    def extract_text(self, response):
        return response.content or ""

    @property
    def supports_tools(self):
        return True


class DummyAgent(BaseAgent):
    """Minimal agent for testing overflow recovery."""

    def run(self, task: str) -> str:
        return ""


async def test_context_overflow_recovery_removes_oldest(monkeypatch):
    """Context overflow should trigger removal and retry."""
    monkeypatch.setattr(Config, "CONTEXT_OVERFLOW_MAX_RETRIES", 1)
    llm = OverflowMockLLM()
    agent = DummyAgent(llm=llm, tools=[])

    await agent.memory.add_message(LLMMessage(role="user", content="Message 1"))
    await agent.memory.add_message(LLMMessage(role="user", content="Message 2"))

    before = agent.memory.short_term.count()

    response = await agent._call_with_overflow_recovery()

    after = agent.memory.short_term.count()

    assert response.content == "ok"
    assert llm.call_count == 2
    assert after < before
