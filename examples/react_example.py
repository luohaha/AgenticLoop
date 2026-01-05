"""Example usage of ReAct Agent."""
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from agent.react_agent import ReActAgent
from tools.calculator import CalculatorTool
from tools.file_ops import FileReadTool, FileWriteTool
from tools.web_search import WebSearchTool


def main():
    """Run ReAct Agent example."""
    print("=" * 60)
    print("ReAct Agent Example")
    print("=" * 60)

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # Initialize agent with tools
    agent = ReActAgent(
        api_key=Config.ANTHROPIC_API_KEY,
        model=Config.MODEL,
        max_iterations=10,
        tools=[
            CalculatorTool(),
            FileReadTool(),
            FileWriteTool(),
            WebSearchTool(),
        ],
    )

    # Example 1: Simple calculation
    print("\n--- Example 1: Simple Calculation ---")
    result1 = agent.run("What is 12345 multiplied by 67890?")
    print(f"\nResult: {result1}")

    # Example 2: File operations
    print("\n\n--- Example 2: File Operations ---")
    result2 = agent.run(
        "Create a file called 'test_output.txt' with the content 'Hello from ReAct Agent!', "
        "then read it back to verify."
    )
    print(f"\nResult: {result2}")

    # Example 3: Web search
    print("\n\n--- Example 3: Web Search ---")
    result3 = agent.run(
        "Search for 'Python agentic frameworks' and tell me the top 3 results"
    )
    print(f"\nResult: {result3}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
