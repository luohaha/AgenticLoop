"""ReAct (Reasoning + Acting) agent implementation."""
from llm import LLMMessage

from .base import BaseAgent


class ReActAgent(BaseAgent):
    """Agent using ReAct (Reasoning + Acting) pattern."""

    SYSTEM_PROMPT = """You are a helpful assistant that can use tools to accomplish tasks.

You should use the following loop:
1. Think about what to do next (reasoning)
2. Use a tool if needed (action)
3. Observe the result
4. Repeat until you can answer the user's question

When you have enough information, provide your final answer directly without using any more tools."""

    def run(self, task: str) -> str:
        """Execute ReAct loop until task is complete.

        Args:
            task: The task to complete

        Returns:
            Final answer as a string
        """
        # Initialize memory with system message and user task
        self.memory.add_message(LLMMessage(role="system", content=self.SYSTEM_PROMPT))
        self.memory.add_message(LLMMessage(role="user", content=task))

        tools = self.tool_executor.get_tool_schemas()

        # Use the generic ReAct loop implementation
        result = self._react_loop(
            messages=[],  # Not used when use_memory=True
            tools=tools,
            max_iterations=self.max_iterations,
            use_memory=True,
            save_to_memory=True,
            verbose=True
        )

        self._print_memory_stats()
        return result

    def _print_memory_stats(self):
        """Print memory usage statistics."""
        stats = self.memory.get_stats()
        total_used = stats['total_input_tokens'] + stats['total_output_tokens']
        print("\n--- Memory Statistics ---")
        print(f"Total used: {total_used} tokens (Input: {stats['total_input_tokens']}, Output: {stats['total_output_tokens']})")
        print(f"Current context: {stats['current_tokens']} tokens")
        print(f"Compressions: {stats['compression_count']}")
        print(f"Net savings: {stats['net_savings']} tokens")
        print(f"Total cost: ${stats['total_cost']:.4f}")
        print(f"Messages: {stats['short_term_count']} in memory, {stats['summary_count']} summaries")
