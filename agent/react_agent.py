"""ReAct (Reasoning + Acting) agent implementation."""
from typing import List, Dict, Any

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
        messages = [{"role": "user", "content": task}]
        tools = self.tool_executor.get_tool_schemas()

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            # Call Claude with tools
            response = self._call_claude(
                messages=messages, system=self.SYSTEM_PROMPT, tools=tools
            )

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response.content})

            # Check if we're done (no tool calls)
            if response.stop_reason == "end_turn":
                final_answer = self._extract_text(response.content)
                print(f"\nFinal answer received.")
                return final_answer

            # Execute tool calls and add results
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"Tool call: {block.name}")
                        print(f"Input: {block.input}")

                        result = self.tool_executor.execute_tool_call(
                            block.name, block.input
                        )
                        print(f"Result: {result[:200]}...")  # Print first 200 chars

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )

                # Add tool results to conversation
                messages.append({"role": "user", "content": tool_results})

        return "Max iterations reached without completion."
