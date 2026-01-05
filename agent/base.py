"""Base agent class for all agent types."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import anthropic

from .tool_executor import ToolExecutor
from tools.base import BaseTool


class BaseAgent(ABC):
    """Abstract base class for all agent types."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_iterations: int = 10,
        tools: List[BaseTool] = None,
    ):
        """Initialize the agent.

        Args:
            api_key: Anthropic API key
            model: Claude model to use
            max_iterations: Maximum number of agent loop iterations
            tools: List of tools available to the agent
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.tool_executor = ToolExecutor(tools or [])

    @abstractmethod
    def run(self, task: str) -> str:
        """Execute the agent on a task and return final answer."""
        pass

    def _call_claude(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> anthropic.types.Message:
        """Helper to call Claude API with consistent parameters."""
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        return self.client.messages.create(**kwargs)

    def _extract_text(self, content) -> str:
        """Extract text from response content blocks."""
        texts = []
        for block in content:
            if hasattr(block, "text"):
                texts.append(block.text)
        return "\n".join(texts) if texts else ""
