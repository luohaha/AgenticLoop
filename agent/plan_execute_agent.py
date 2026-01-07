"""Plan-and-Execute agent implementation."""
from typing import List
import re

from llm import LLMMessage
from .base import BaseAgent


class PlanExecuteAgent(BaseAgent):
    """Agent using Plan-and-Execute pattern."""

    PLANNER_PROMPT = """You are a planning assistant. Given a task, create a step-by-step plan.

Output your plan as a numbered list of steps. Each step should be clear and actionable.
Do not execute the plan, just create it.

Task: {task}

Plan:"""

    EXECUTOR_PROMPT = """You are executing step {step_num} of a plan: {step}

Previous steps and results:
{history}

Use available tools to complete this specific step. When done, provide a brief summary of what you accomplished."""

    SYNTHESIZER_PROMPT = """You have completed a multi-step plan. Here are the results:

{results}

Original task: {task}

Provide a final answer to the user's original task based on these results."""

    def run(self, task: str) -> str:
        """Execute Plan-and-Execute loop.

        Args:
            task: The task to complete

        Returns:
            Final answer as a string
        """
        # Phase 1: Create plan
        print("\n" + "=" * 60)
        print("PHASE 1: PLANNING")
        print("=" * 60)
        plan = self._create_plan(task)
        print(f"\n{plan}")

        # Phase 2: Execute each step
        print("\n" + "=" * 60)
        print("PHASE 2: EXECUTION")
        print("=" * 60)
        step_results = []
        steps = self._parse_plan(plan)

        if not steps:
            return "Failed to parse plan into executable steps."

        for i, step in enumerate(steps, 1):
            print(f"\n--- Executing Step {i}/{len(steps)}: {step} ---")
            result = self._execute_step(step, i, step_results, task)
            step_results.append(f"Step {i}: {step}\nResult: {result}")
            print(f"✓ Step {i} completed")

        # Phase 3: Synthesize final answer
        print("\n" + "=" * 60)
        print("PHASE 3: SYNTHESIS")
        print("=" * 60)
        final_answer = self._synthesize_results(task, step_results)

        # Print memory statistics
        self._print_memory_stats()

        return final_answer

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

    def _create_plan(self, task: str) -> str:
        """Generate a plan without using tools."""
        messages = [
            LLMMessage(role="system", content="You are a planning expert. Create clear, actionable plans."),
            LLMMessage(role="user", content=self.PLANNER_PROMPT.format(task=task))
        ]
        response = self._call_llm(messages=messages)

        # Track token usage from planning phase
        if response.usage:
            self.memory.token_tracker.add_input_tokens(response.usage.get("input_tokens", 0))
            self.memory.token_tracker.add_output_tokens(response.usage.get("output_tokens", 0))

        return self._extract_text(response)

    def _parse_plan(self, plan: str) -> List[str]:
        """Parse plan into individual steps."""
        lines = plan.strip().split("\n")
        steps = []
        for line in lines:
            # Match numbered lists like "1. ", "1) ", etc.
            match = re.match(r"^\d+[\.)]\s+(.+)$", line.strip())
            if match:
                steps.append(match.group(1))
        return steps

    def _execute_step(
        self, step: str, step_num: int, previous_results: List[str], original_task: str
    ) -> str:
        """Execute a single step using tools (mini ReAct loop)."""
        history = "\n\n".join(previous_results) if previous_results else "None"

        # Initialize step-specific message list for mini-loop
        messages = [
            LLMMessage(
                role="user",
                content=self.EXECUTOR_PROMPT.format(
                    step_num=step_num, step=step, history=history
                )
            )
        ]

        tools = self.tool_executor.get_tool_schemas()

        # Use the generic ReAct loop for this step (mini-loop)
        result = self._react_loop(
            messages=messages,
            tools=tools,
            max_iterations=5,  # Limited iterations for each step
            use_memory=False,  # Use local messages list, not global memory
            save_to_memory=False,  # Don't auto-save to memory
            verbose=False  # Quieter output for sub-steps
        )

        # Save step result summary to main memory
        self.memory.add_message(
            LLMMessage(role="assistant", content=f"Step {step_num} completed: {result}")
        )

        return result

    def _synthesize_results(self, task: str, results: List[str]) -> str:
        """Combine step results into final answer."""
        messages = [
            LLMMessage(
                role="user",
                content=self.SYNTHESIZER_PROMPT.format(
                    results="\n\n".join(results), task=task
                )
            )
        ]
        response = self._call_llm(messages=messages)

        # Track token usage from synthesis phase
        if response.usage:
            self.memory.token_tracker.add_input_tokens(response.usage.get("input_tokens", 0))
            self.memory.token_tracker.add_output_tokens(response.usage.get("output_tokens", 0))

        return self._extract_text(response)
